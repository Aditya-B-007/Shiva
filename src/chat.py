import os
import sys
import io
import math
import base64
import threading
import webbrowser
from PIL import Image
from dataclasses import dataclass
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel, TextGenerator
from src.imageRecognitionForNandi import (
    VisionEncoder,
    VisionEncoderFactory,
    MLPProjector,
    VisionBridge,
    ImagePreprocessor
)
from src.dataIngestionPipeline import TokenSplicer, MultimodalInputBatch
from src.continuousLearningFromHumanFeedback import OnlineFeedbackTrainer
from src.interfaces import IMultimodalSplicer, ITokenizer, ITextGenerator, IImagePreprocessor
from src.config import (
    default_model_config,
    default_chat_generation_config,
    default_recognition_generation_config,
    default_live_learning_config,
    INFERENCE_MAX_SEQ_LEN,
    DEFAULT_PORT,
    HTML_PAGE,
)


@dataclass
class ApplicationState:
    """Strongly typed application state container replacing untyped global dictionaries."""
    languageModel: TransformerModel
    textGenerator: ITextGenerator
    visionBridge: VisionBridge
    visionEncoder: VisionEncoder
    imagePreprocessor: IImagePreprocessor
    tokenSplicer: IMultimodalSplicer
    tokenizer: ITokenizer
    imageTokenIdentifier: Optional[int]
    executionDevice: torch.device
    onlineTrainer: OnlineFeedbackTrainer


app = FastAPI(title="Nandi SLM Web UI")

appState: Optional[ApplicationState] = None


class ChatRequest(BaseModel):
    text: str


class RecognizeRequest(BaseModel):
    image: str


class FeedbackRequest(BaseModel):
    prompt: str
    corrected_response: str
    thought: Optional[str] = None
    image: Optional[str] = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_nandi():
    global appState
    device = get_device()
    tokenizer = TokenizerNandi()
    if os.path.exists(tokenizer.model_path):
        tokenizer.load()
        if tokenizer.tokenizer.token_to_id("<image>") is None:
            tokenizer.addSpecialTokens(["<image>"])
        vocab_size = tokenizer.getVocabSize()
    else:
        vocab_size = default_model_config.ntoken

    image_token_id = tokenizer.tokenizer.token_to_id("<image>") if hasattr(tokenizer, "tokenizer") else None

    # 1. Initialize Vision Bridge via Factory
    print("[+] Initializing SigLIP Vision Encoder via Factory and MLP Projector...")
    vision_encoder = VisionEncoderFactory.createVisionEncoder().to(device)
    mlp_projector = MLPProjector(visual_dim=vision_encoder.hiddenDim, language_dim=default_model_config.ninp).to(device)
    bridge = VisionBridge(encoder=vision_encoder, projector=mlp_projector).to(device)
    splicer: IMultimodalSplicer = TokenSplicer()
    image_preprocessor: IImagePreprocessor = ImagePreprocessor()

    # 2. Initialize Language Model Backbone & Text Generator
    model = TransformerModel(
        ntoken=vocab_size,
        ninp=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        nlayers=default_model_config.nlayers,
        dropout=0.0,
        max_seq_len=INFERENCE_MAX_SEQ_LEN
    ).to(device)
    text_generator: ITextGenerator = TextGenerator(model)

    # 3. Load Checkpoint (prefer nandi_vision_final.pt, fallback to chat/latest)
    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
    target_ckpt = os.path.join(checkpoint_dir, "nandi_vision_final.pt")
    if not os.path.exists(target_ckpt):
        target_ckpt = os.path.join(checkpoint_dir, "nandi_chat_final.pt")

    if os.path.exists(target_ckpt):
        print(f"[+] Loading model weights from: {target_ckpt}")
        ckpt = torch.load(target_ckpt, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "projector_state_dict" in ckpt:
            bridge.projector.load_state_dict(ckpt["projector_state_dict"])
            print("[+] Successfully loaded trained MLP Projector weights.")
    else:
        print("[!] No checkpoint found in model_artifacts/checkpoints.")

    model.eval()
    bridge.eval()

    online_trainer = OnlineFeedbackTrainer(
        languageModel=model,
        visionBridge=bridge,
        visionEncoder=vision_encoder,
        tokenSplicer=splicer,
        tokenizer=tokenizer,
        imageTokenIdentifier=image_token_id,
        executionDevice=device,
        learningRate=default_live_learning_config.learningRate
    )

    appState = ApplicationState(
        languageModel=model,
        textGenerator=text_generator,
        visionBridge=bridge,
        visionEncoder=vision_encoder,
        imagePreprocessor=image_preprocessor,
        tokenSplicer=splicer,
        tokenizer=tokenizer,
        imageTokenIdentifier=image_token_id,
        executionDevice=device,
        onlineTrainer=online_trainer
    )


@app.on_event("startup")
def startup_event():
    load_nandi()


@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML_PAGE


@app.post("/api/feedback")
@app.post("/feedback")
def handle_feedback(req: FeedbackRequest):
    if appState is None:
        raise RuntimeError("Application state is not initialized.")

    user_prompt = req.prompt.strip()
    corrected_response = req.corrected_response.strip()
    thought = (req.thought or "Clear, accurate, and helpful response.").strip()

    if req.image:
        img_data = req.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_data)
        result = appState.onlineTrainer.executeLiveMultimodalLearningStep(
            rawImageBytes=raw_bytes,
            userPrompt=user_prompt if user_prompt else "Describe what is happening in this picture: <image>",
            correctedResponse=corrected_response,
            reasoningThought=thought
        )
    else:
        result = appState.onlineTrainer.executeLiveTextLearningStep(
            userPrompt=user_prompt,
            correctedResponse=corrected_response,
            reasoningThought=thought
        )

    return result


@app.post("/api/chat")
@app.post("/chat")
def handle_chat(req: ChatRequest):
    if appState is None:
        raise RuntimeError("Application state is not initialized.")

    cfg = default_chat_generation_config
    formatted_prompt = f"User: {req.text.strip()}\nAssistant: "
    encoded = appState.tokenizer.encode(formatted_prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=appState.executionDevice)

    eos_id = getattr(appState.tokenizer, "eos_token_id", None)
    if eos_id is None and hasattr(appState.tokenizer, "tokenizer"):
        eos_id = appState.tokenizer.tokenizer.token_to_id("</s>")

    output_ids = appState.textGenerator.generateTokens(
        tokenIndices=input_ids,
        maxNewTokens=cfg.maxNewTokens,
        temperature=cfg.temperature,
        topK=cfg.topK,
        topP=cfg.topP,
        repetitionPenalty=cfg.repetitionPenalty,
        eosTokenId=eos_id
    )
    resp_tokens = output_ids[0][input_ids.size(1):].tolist()
    response = appState.tokenizer.decode(resp_tokens, skipSpecialTokens=True).strip()

    # Strip thoughts if model outputs internal reasoning
    if "<|thought|>" in response:
        parts = response.split("<|thought|>")
        response = parts[-1].strip() if len(parts) >= 3 else response.replace("<|thought|>", "").strip()

    return {"text": response if response else "[No response]"}


@app.post("/api/recognize")
@app.post("/recognize")
def handle_recognize(req: RecognizeRequest):
    if appState is None:
        raise RuntimeError("Application state is not initialized.")

    cfg = default_recognition_generation_config

    img_data = req.image
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pixel_values = appState.visionEncoder.processor(images=img, return_tensors="pt").pixel_values.to(appState.executionDevice)

    prompt = "User: Describe what is happening in this picture: <image>\nAssistant: <|thought|>\n"
    encoded = appState.tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=appState.executionDevice)

    with torch.no_grad():
        visual_tokens = appState.visionBridge(pixel_values)
        embedding_layer = appState.languageModel.get_input_embeddings()
        ninp = getattr(appState.languageModel, "ninp", visual_tokens.size(-1))
        scale = math.sqrt(ninp)
        text_embeddings = embedding_layer(input_ids) * scale

        batch_contract = MultimodalInputBatch(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            visual_tokens=visual_tokens,
            image_token_id=appState.imageTokenIdentifier,
        )
        spliced = appState.tokenSplicer.splice(batch_contract)

        eos_id = getattr(appState.tokenizer, "eos_token_id", None)
        if eos_id is None and hasattr(appState.tokenizer, "tokenizer"):
            eos_id = appState.tokenizer.tokenizer.token_to_id("</s>")

        output_tokens = appState.textGenerator.generateTokens(
            inputsEmbeds=spliced.embeddings,
            prefixLength=spliced.embeddings.size(1),
            maxNewTokens=cfg.maxNewTokens,
            temperature=cfg.temperature,
            topK=cfg.topK,
            topP=cfg.topP,
            repetitionPenalty=cfg.repetitionPenalty,
            eosTokenId=eos_id
        )

        if isinstance(output_tokens, torch.Tensor):
            token_list = output_tokens[0].tolist()
        else:
            token_list = list(output_tokens)

        result_text = appState.tokenizer.decode(token_list, skipSpecialTokens=True).strip()

        # Clean reasoning tokens out to present clean caption to the user
        if "<|thought|>" in result_text:
            parts = result_text.split("<|thought|>")
            result_text = parts[-1].strip() if len(parts) >= 2 else result_text.replace("<|thought|>", "").strip()

    return {"text": result_text if result_text else "[Model generated empty description]"}


def open_browser(port: int = DEFAULT_PORT):
    url = f"http://127.0.0.1:{port}"
    print(f"\nNandi SLM Web UI is running at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


if __name__ == "__main__":
    open_browser(DEFAULT_PORT)
    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT, log_level="info")

import os
import sys
import io
import math
import base64
import threading
import webbrowser
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.imageRecognitionForNandi import VisionEncoder, MLPProjector, VisionBridge
from src.dataIngestionPipeline import TokenSplicer, MultimodalInputBatch
from src.continousLearningFromHumanFeedback import LiveLearner
from src.config import default_model_config, HTML_PAGE


app = FastAPI(title="Nandi SLM Web UI")

# Global model state
state = {}


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
    device = get_device()
    tokenizer = TokenizerNandi()
    if os.path.exists(tokenizer.model_path):
        tokenizer.load()
        if tokenizer.tokenizer.token_to_id("<image>") is None:
            tokenizer.add_special_tokens(["<image>"])
        vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = default_model_config.ntoken

    image_token_id = tokenizer.tokenizer.token_to_id("<image>") if hasattr(tokenizer, "tokenizer") else None

    # 1. Initialize Vision Bridge
    print("[+] Initializing SigLIP Vision Encoder and MLP Projector...")
    vision_encoder = VisionEncoder().to(device)
    mlp_projector = MLPProjector(visual_dim=vision_encoder.hidden_dim, language_dim=default_model_config.ninp).to(device)
    bridge = VisionBridge(encoder=vision_encoder, projector=mlp_projector).to(device)
    splicer = TokenSplicer()

    # 2. Initialize Language Model Backbone (Match training max_seq_len=512)
    model = TransformerModel(
        ntoken=vocab_size,
        ninp=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        nlayers=default_model_config.nlayers,
        dropout=0.0,
        max_seq_len=512
    ).to(device)

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

    live_learner = LiveLearner(
        languageModel=model,
        visionBridge=bridge,
        visionEncoder=vision_encoder,
        tokenSplicer=splicer,
        tokenizer=tokenizer,
        imageTokenIdentifier=image_token_id,
        executionDevice=device,
        learningRate=1e-5
    )

    state["model"] = model
    state["bridge"] = bridge
    state["vision_encoder"] = vision_encoder
    state["splicer"] = splicer
    state["tokenizer"] = tokenizer
    state["image_token_id"] = image_token_id
    state["device"] = device
    state["live_learner"] = live_learner


@app.on_event("startup")
def startup_event():
    load_nandi()


@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML_PAGE


@app.post("/api/feedback")
@app.post("/feedback")
def handle_feedback(req: FeedbackRequest):
    live_learner: LiveLearner = state["live_learner"]
    user_prompt = req.prompt.strip()
    corrected_response = req.corrected_response.strip()
    thought = (req.thought or "Clear, accurate, and helpful response.").strip()

    if req.image:
        img_data = req.image
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_data)
        result = live_learner.executeLiveMultimodalLearningStep(
            rawImageBytes=raw_bytes,
            userPrompt=user_prompt if user_prompt else "Describe what is happening in this picture: <image>",
            correctedResponse=corrected_response,
            reasoningThought=thought
        )
    else:
        result = live_learner.executeLiveTextLearningStep(
            userPrompt=user_prompt,
            correctedResponse=corrected_response,
            reasoningThought=thought
        )

    return result


@app.post("/api/chat")
@app.post("/chat")
def handle_chat(req: ChatRequest):
    model = state["model"]
    tokenizer = state["tokenizer"]
    device = state["device"]

    formatted_prompt = f"User: {req.text.strip()}\nAssistant: "
    encoded = tokenizer.encode(formatted_prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None and hasattr(tokenizer, "tokenizer"):
        eos_id = tokenizer.tokenizer.token_to_id("</s>")

    output_ids = model.generate(
        idx=input_ids,
        max_new_tokens=150,
        temperature=0.6,
        top_k=25,
        top_p=0.85,
        repetition_penalty=1.3,
        eos_token_id=eos_id
    )
    resp_tokens = output_ids[0][input_ids.size(1):].tolist()
    response = tokenizer.decode(resp_tokens, skip_special_tokens=True).strip()
    
    # Strip thoughts if model outputs internal reasoning
    if "<|thought|>" in response:
        parts = response.split("<|thought|>")
        response = parts[-1].strip() if len(parts) >= 3 else response.replace("<|thought|>", "").strip()

    return {"text": response if response else "[No response]"}


@app.post("/api/recognize")
@app.post("/recognize")
def handle_recognize(req: RecognizeRequest):
    model = state["model"]
    bridge = state["bridge"]
    vision_encoder = state["vision_encoder"]
    splicer = state["splicer"]
    tokenizer = state["tokenizer"]
    image_token_id = state["image_token_id"]
    device = state["device"]

    img_data = req.image
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    pixel_values = vision_encoder.processor(images=img, return_tensors="pt").pixel_values.to(device)

    prompt = "User: Describe what is happening in this picture: <image>\nAssistant: <|thought|>\n"
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        visual_tokens = bridge(pixel_values)
        embedding_layer = model.get_input_embeddings()
        ninp = getattr(model, "ninp", visual_tokens.size(-1))
        scale = math.sqrt(ninp)
        text_embeddings = embedding_layer(input_ids) * scale

        batch_contract = MultimodalInputBatch(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            visual_tokens=visual_tokens,
            image_token_id=image_token_id,
        )
        spliced = splicer.splice(batch_contract)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None and hasattr(tokenizer, "tokenizer"):
            eos_id = tokenizer.tokenizer.token_to_id("</s>")

        output_tokens = model.generate(
            inputs_embeds=spliced.embeddings,
            prefix_len=spliced.embeddings.size(1),
            max_new_tokens=100,
            temperature=0.3,
            top_k=20,
            top_p=0.8,
            repetition_penalty=1.35,
            eos_token_id=eos_id
        )

        if isinstance(output_tokens, torch.Tensor):
            token_list = output_tokens[0].tolist()
        else:
            token_list = list(output_tokens)

        result_text = tokenizer.decode(token_list, skip_special_tokens=True).strip()

        # Clean reasoning tokens out to present clean caption to the user
        if "<|thought|>" in result_text:
            parts = result_text.split("<|thought|>")
            result_text = parts[-1].strip() if len(parts) >= 2 else result_text.replace("<|thought|>", "").strip()

    return {"text": result_text if result_text else "[Model generated empty description]"}


def open_browser(port: int = 7860):
    url = f"http://127.0.0.1:{port}"
    print(f"\nNandi SLM Web UI is running at {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()


if __name__ == "__main__":
    port = 7860
    open_browser(port)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


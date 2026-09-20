import os
import torch
import configparser
from dataclasses import dataclass
from typing import Tuple

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".config"))


# =============================================================================
# Model / Architecture Config
# =============================================================================

@dataclass
class ModelConfig:
    ntoken: int = 9714
    ninp: int = 512
    nhead: int = 8
    n_kv_heads: int = 2
    nhid: int = 2048
    nlayers: int = 9
    dropout: float = 0.1
    max_seq_len: int = 256


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    model: str = "google/siglip-base-patch16-224"


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 4
    stride: int = 512
    learning_rate: float = 1.5e-3
    min_lr: float = 1.5e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 1
    eval_interval: int = 250
    save_interval: int = 500


# =============================================================================
# Generation / Inference Configs
# =============================================================================

@dataclass
class GenerationConfig:
    """Default autoregressive generation parameters."""
    maxNewTokens: int = 250
    temperature: float = 0.5
    topK: int = 30
    topP: float = 0.85
    repetitionPenalty: float = 1.2


@dataclass
class ChatGenerationConfig:
    """Generation parameters for the /api/chat endpoint."""
    maxNewTokens: int = 150
    temperature: float = 0.6
    topK: int = 25
    topP: float = 0.85
    repetitionPenalty: float = 1.3


@dataclass
class RecognitionGenerationConfig:
    """Generation parameters for the /api/recognize endpoint."""
    maxNewTokens: int = 100
    temperature: float = 0.3
    topK: int = 20
    topP: float = 0.8
    repetitionPenalty: float = 1.35


# =============================================================================
# Online / Live-Learning Config
# =============================================================================

@dataclass
class LiveLearningConfig:
    """Parameters governing the online feedback training loop."""
    learningRate: float = 1e-5
    projectorLrMultiplier: float = 2.0
    weightDecay: float = 0.01
    maxSeqLen: int = 512
    gradClip: float = 1.0
    checkpointEveryNSteps: int = 3


# =============================================================================
# Supervised Fine-Tuning Config
# =============================================================================

@dataclass
class SFTConfig:
    """Hyperparameters for supervised fine-tuning (finetuningNandi.py)."""
    batchSize: int = 8
    gradAccumSteps: int = 4
    learningRate: float = 3e-4
    minLr: float = 3e-5
    weightDecay: float = 0.05
    gradClip: float = 1.0
    epochs: int = 10
    evalInterval: int = 100
    saveInterval: int = 250
    maxSeqLen: int = 512
    dropout: float = 0.05


# =============================================================================
# Multimodal Stage Training Configs
# =============================================================================

@dataclass
class Stage1TrainConfig:
    """Hyperparameters for Stage-1 projector alignment (stage1AlignmentNandi.py)."""
    batchSize: int = 2
    gradAccumSteps: int = 4
    learningRate: float = 1e-3
    minLr: float = 1e-6
    weightDecay: float = 0.01
    gradClip: float = 1.0
    epochs: int = 8
    evalInterval: int = 10
    saveInterval: int = 50
    maxSeqLen: int = 512
    dropout: float = 0.05


@dataclass
class Stage2TrainConfig:
    """Hyperparameters for Stage-2 joint multimodal fine-tuning (stage2MultimodalNandi.py)."""
    batchSize: int = 2
    gradAccumSteps: int = 4
    learningRate: float = 2e-5
    minLr: float = 1e-6
    weightDecay: float = 0.01
    gradClip: float = 1.0
    epochs: int = 8
    evalInterval: int = 10
    saveInterval: int = 50
    maxSeqLen: int = 512
    dropout: float = 0.05


# =============================================================================
# Module-level Constants
# (named, never raw literals anywhere else in the codebase)
# =============================================================================

# Tokenizer training
VOCAB_SIZE: int = 50257
MIN_FREQUENCY: int = 2

# Server
DEFAULT_PORT: int = 7860

# Transformer / RoPE
ROPE_BASE: float = 10000.0
MAX_GENERATE_CONTEXT: int = 8192       # max tokens kept in generate() rolling window
REPETITION_PENALTY_WINDOW: int = 64    # last N tokens checked for repetition penalty
TEMPERATURE_EPSILON: float = 1e-5      # floor to avoid division by zero in temperature scaling

# Inference
INFERENCE_MAX_SEQ_LEN: int = 512       # context length used at chat / inference time

# Vision / image preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Tokenizer fallbacks
DEFAULT_PAD_TOKEN_ID: int = 1


# =============================================================================
# Config loader (reads .config file at project root)
# =============================================================================

def load_config(config_path=CONFIG_PATH):
    model_cfg = ModelConfig()
    vision_cfg = VisionConfig()
    train_cfg = TrainConfig()

    if os.path.exists(config_path):
        parser = configparser.ConfigParser()
        try:
            parser.read(config_path)
        except configparser.MissingSectionHeaderError:
            with open(config_path, "r", encoding="utf-8") as f:
                content = "[default]\n" + f.read()
            parser.read_string(content)

        kv = {}
        for section in parser.sections():
            for k, v in parser[section].items():
                kv[k.lower()] = v.strip(' "\'')
        for k, v in parser.defaults().items():
            kv[k.lower()] = v.strip(' "\'')

        if "ntoken" in kv:
            model_cfg.ntoken = int(kv["ntoken"])
        if "ninp" in kv:
            model_cfg.ninp = int(kv["ninp"])
        if "nhead" in kv:
            model_cfg.nhead = int(kv["nhead"])
        if "n_kv_heads" in kv:
            model_cfg.n_kv_heads = int(kv["n_kv_heads"])
        if "nhid" in kv:
            model_cfg.nhid = int(kv["nhid"])
        if "nlayers" in kv:
            model_cfg.nlayers = int(kv["nlayers"])
        if "dropout" in kv:
            model_cfg.dropout = float(kv["dropout"])
        if "max_seq_len" in kv:
            model_cfg.max_seq_len = int(kv["max_seq_len"])

        if "image_size" in kv:
            vision_cfg.image_size = int(kv["image_size"])
        if "patch_size" in kv:
            vision_cfg.patch_size = int(kv["patch_size"])
        if "in_channels" in kv:
            vision_cfg.in_channels = int(kv["in_channels"])
        if "model" in kv:
            vision_cfg.model = str(kv["model"])

        if "batch_size" in kv:
            train_cfg.batch_size = int(kv["batch_size"])
        if "grad_accum_steps" in kv:
            train_cfg.grad_accum_steps = int(kv["grad_accum_steps"])
        if "stride" in kv:
            train_cfg.stride = int(kv["stride"])
        if "learning_rate" in kv:
            train_cfg.learning_rate = float(kv["learning_rate"])
        if "min_lr" in kv:
            train_cfg.min_lr = float(kv["min_lr"])
        if "weight_decay" in kv:
            train_cfg.weight_decay = float(kv["weight_decay"])
        if "grad_clip" in kv:
            train_cfg.grad_clip = float(kv["grad_clip"])
        if "epochs" in kv:
            train_cfg.epochs = int(kv["epochs"])
        if "eval_interval" in kv:
            train_cfg.eval_interval = int(kv["eval_interval"])
        if "save_interval" in kv:
            train_cfg.save_interval = int(kv["save_interval"])

    return model_cfg, vision_cfg, train_cfg


# =============================================================================
# Device / AMP utility  (centralised here so no script duplicates it)
# =============================================================================

def getDeviceAndDtype() -> Tuple[torch.device, bool, torch.dtype]:
    """
    Detect the best available compute device and return
    (device, use_amp, amp_dtype).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = True
        amp_dtype = torch.bfloat16
        print(">> Running on Apple Silicon Metal Performance Shaders (MPS) with AMP acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        use_amp = False
        amp_dtype = torch.float32
        print(">> Running on CPU.")
    return device, use_amp, amp_dtype


# =============================================================================
# Module-level singletons (imported by the rest of the codebase)
# =============================================================================

default_model_config, default_vision_config, default_train_config = load_config()

default_generation_config           = GenerationConfig()
default_chat_generation_config      = ChatGenerationConfig()
default_recognition_generation_config = RecognitionGenerationConfig()
default_live_learning_config        = LiveLearningConfig()
default_sft_config                  = SFTConfig()
default_stage1_config               = Stage1TrainConfig()
default_stage2_config               = Stage2TrainConfig()


# =============================================================================
# HTML_PAGE — kept in config so chat.py stays thin
# =============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Nandi SLM - ChatGPT UI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: #212121; color: #ececec; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 16px 24px; background: #171717; border-bottom: 1px solid #303030; display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 1.2rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .badge { background: #333; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; color: #888; }
  #chat-container { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; max-width: 800px; margin: 0 auto; width: 100%; }
  .msg { padding: 14px 18px; border-radius: 12px; line-height: 1.5; max-width: 80%; }
  .user { align-self: flex-end; background: #2f2f2f; color: #fff; }
  .assistant { align-self: flex-start; background: #181818; border: 1px solid #333; }
  .msg img { max-width: 100%; max-height: 250px; border-radius: 8px; margin-bottom: 8px; display: block; }
  #input-container { padding: 16px 24px; background: #171717; border-top: 1px solid #303030; }
  .input-box { max-width: 800px; margin: 0 auto; display: flex; gap: 12px; align-items: center; position: relative; }
  #text-input { flex: 1; padding: 14px 18px; background: #2f2f2f; border: 1px solid #444; border-radius: 24px; color: #fff; outline: none; font-size: 1rem; }
  #text-input:disabled { background: #1e1e1e; color: #666; cursor: not-allowed; border-color: #333; }
  .btn { background: #fff; color: #000; border: none; padding: 12px 20px; border-radius: 20px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  .btn:hover { background: #ddd; }
  .btn:disabled { background: #555; color: #888; cursor: not-allowed; }
  .upload-btn { background: #333; color: #fff; padding: 10px 16px; border-radius: 20px; cursor: pointer; display: flex; align-items: center; gap: 6px; font-size: 0.9rem; }
  .upload-btn:hover { background: #444; }
  #file-input { display: none; }
  #img-preview-bar { max-width: 800px; margin: 0 auto 8px auto; display: none; align-items: center; gap: 12px; background: #2a2a2a; padding: 8px 16px; border-radius: 12px; }
  #img-preview { width: 40px; height: 40px; object-fit: cover; border-radius: 6px; }
  .tag { font-size: 0.85rem; color: #ffb86c; font-weight: 500; }
  .remove-img { color: #ff5555; cursor: pointer; margin-left: auto; font-weight: bold; }
  .teach-btn { display: inline-block; margin-top: 8px; font-size: 0.8rem; color: #888; cursor: pointer; text-decoration: underline; background: none; border: none; padding: 0; }
  .teach-btn:hover { color: #ccc; }
  .teach-box { margin-top: 10px; padding: 10px; background: #222; border: 1px solid #444; border-radius: 8px; display: flex; flex-direction: column; gap: 8px; }
  .teach-input { width: 100%; padding: 8px 12px; background: #2f2f2f; border: 1px solid #555; border-radius: 6px; color: #fff; outline: none; font-size: 0.9rem; }
  .teach-submit-btn { align-self: flex-end; background: #444; color: #fff; border: none; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; cursor: pointer; }
  .teach-submit-btn:hover { background: #555; }
</style>
</head>
<body>
<header>
  <h1>Nandi SLM</h1>
  <span class="badge">nandi2 text+image</span>
</header>
<div id="chat-container">
  <div class="msg assistant">Hello! I am Nandi SLM. You can chat with me, or click <b>Image</b> to upload a photo for pure image recognition.</div>
</div>
<div id="input-container">
  <div id="img-preview-bar">
    <img id="img-preview" src="">
    <span class="tag">Image attached: Text input locked for pure image recognition.</span>
    <span class="remove-img" onclick="clearImage()">Remove</span>
  </div>
  <div class="input-box">
    <label class="upload-btn" for="file-input">Image</label>
    <input type="file" id="file-input" accept="image/*" onchange="handleImageSelect(event)">
    <input type="text" id="text-input" placeholder="Type a message..." onkeydown="if(event.key==='Enter') send()">
    <button class="btn" id="send-btn" onclick="send()">Send</button>
  </div>
</div>
<script>
let currentImageBase64 = null;
let turnHistory = {};
let turnCounter = 0;

function handleImageSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    currentImageBase64 = evt.target.result;
    document.getElementById("img-preview").src = currentImageBase64;
    document.getElementById("img-preview-bar").style.display = "flex";
    const textInput = document.getElementById("text-input");
    textInput.value = "";
    textInput.disabled = true;
    textInput.placeholder = "Image attached. Click Recognize to continue.";
    document.getElementById("send-btn").innerText = "Recognize";
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentImageBase64 = null;
  document.getElementById("file-input").value = "";
  document.getElementById("img-preview-bar").style.display = "none";
  const textInput = document.getElementById("text-input");
  textInput.disabled = false;
  textInput.placeholder = "Type a message...";
  document.getElementById("send-btn").innerText = "Send";
}

function openTeachBox(turnId) {
  const container = document.getElementById("teach-container-" + turnId);
  if (!container) return;
  container.style.display = (container.style.display === "none" || !container.style.display) ? "flex" : "none";
}

async function submitCorrection(turnId) {
  const inputEl = document.getElementById("teach-input-" + turnId);
  const correctedResponse = inputEl ? inputEl.value.trim() : "";
  if (!correctedResponse) return;

  const btnEl = document.getElementById("teach-submit-btn-" + turnId);
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.innerText = "Learning...";
  }

  const turnData = turnHistory[turnId] || {};
  const payload = {
    prompt: turnData.prompt || "User input",
    corrected_response: correctedResponse,
    thought: "Clear, accurate, and helpful response.",
    image: turnData.image || null
  };

  try {
    const res = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error("HTTP error " + res.status);
    }
    const data = await res.json();
    console.log("Feedback recorded:", data);

    const toggleBtn = document.getElementById("teach-toggle-" + turnId);
    if (toggleBtn) {
      toggleBtn.innerText = "Learned (Loss: " + (data.loss ? data.loss.toFixed(3) : "OK") + ")";
      toggleBtn.style.color = "#4caf50";
      toggleBtn.disabled = true;
    }
    const container = document.getElementById("teach-container-" + turnId);
    if (container) container.style.display = "none";
  } catch (err) {
    console.error("Failed to submit feedback:", err);
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerText = "Retry Teach";
    }
  }
}

async function send() {
  const chat = document.getElementById("chat-container");
  const sendBtn = document.getElementById("send-btn");
  turnCounter++;
  const thisTurn = turnCounter;

  if (currentImageBase64) {
    const imgData = currentImageBase64;
    const promptText = "Describe what is happening in this picture: <image>";
    turnHistory[thisTurn] = { prompt: promptText, image: imgData };

    chat.innerHTML += `<div class="msg user"><img src="${imgData}"><br><i>[Image Recognition Request]</i></div>`;
    clearImage();
    sendBtn.disabled = true;
    chat.scrollTop = chat.scrollHeight;

    try {
      const res = await fetch("/api/recognize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imgData })
      });
      const data = await res.json();
      const botText = data.text || "[No description generated]";

      chat.innerHTML += `
        <div class="msg assistant">
          <div>${botText}</div>
          <button id="teach-toggle-${thisTurn}" class="teach-btn" onclick="openTeachBox(${thisTurn})">Teach Nandi</button>
          <div id="teach-container-${thisTurn}" class="teach-box" style="display:none;">
            <input id="teach-input-${thisTurn}" class="teach-input" type="text" placeholder="Enter the correct description...">
            <button id="teach-submit-btn-${thisTurn}" class="teach-submit-btn" onclick="submitCorrection(${thisTurn})">Teach</button>
          </div>
        </div>`;
    } catch (err) {
      chat.innerHTML += `<div class="msg assistant" style="color:#ff5555;">Error contacting server: ${err.message}</div>`;
    }

    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  } else {
    const textInput = document.getElementById("text-input");
    const text = textInput.value.trim();
    if (!text) return;
    textInput.value = "";

    turnHistory[thisTurn] = { prompt: text, image: null };

    chat.innerHTML += `<div class="msg user">${text}</div>`;
    sendBtn.disabled = true;
    chat.scrollTop = chat.scrollHeight;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      });
      const data = await res.json();
      const botText = data.text || "[No response]";

      chat.innerHTML += `
        <div class="msg assistant">
          <div>${botText}</div>
          <button id="teach-toggle-${thisTurn}" class="teach-btn" onclick="openTeachBox(${thisTurn})">Teach Nandi</button>
          <div id="teach-container-${thisTurn}" class="teach-box" style="display:none;">
            <input id="teach-input-${thisTurn}" class="teach-input" type="text" placeholder="Enter what Nandi should have said...">
            <button id="teach-submit-btn-${thisTurn}" class="teach-submit-btn" onclick="submitCorrection(${thisTurn})">Teach</button>
          </div>
        </div>`;
    } catch (err) {
      chat.innerHTML += `<div class="msg assistant" style="color:#ff5555;">Error contacting server: ${err.message}</div>`;
    }

    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}
</script>
</body>
</html>"""

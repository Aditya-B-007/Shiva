import os
import configparser
from dataclasses import dataclass

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".config"))

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

@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 2
    stride: int = 128
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 2
    eval_interval: int = 50
    save_interval: int = 250

def load_config(config_path=CONFIG_PATH):
    model_cfg = ModelConfig()
    vision_cfg = VisionConfig()
    train_cfg = TrainConfig()

    if os.path.exists(config_path):
        parser = configparser.ConfigParser()
        parser.read(config_path)

        if "model" in parser:
            m = parser["model"]
            if "ntoken" in m:
                model_cfg.ntoken = int(m["ntoken"])
            if "ninp" in m:
                model_cfg.ninp = int(m["ninp"])
            if "nhead" in m:
                model_cfg.nhead = int(m["nhead"])
            if "n_kv_heads" in m:
                model_cfg.n_kv_heads = int(m["n_kv_heads"])
            if "nhid" in m:
                model_cfg.nhid = int(m["nhid"])
            if "nlayers" in m:
                model_cfg.nlayers = int(m["nlayers"])
            if "dropout" in m:
                model_cfg.dropout = float(m["dropout"])
            if "max_seq_len" in m:
                model_cfg.max_seq_len = int(m["max_seq_len"])

        if "vision" in parser:
            v = parser["vision"]
            if "image_size" in v:
                vision_cfg.image_size = int(v["image_size"])
            if "patch_size" in v:
                vision_cfg.patch_size = int(v["patch_size"])
            if "in_channels" in v:
                vision_cfg.in_channels = int(v["in_channels"])

        if "training" in parser:
            t = parser["training"]
            if "batch_size" in t:
                train_cfg.batch_size = int(t["batch_size"])
            if "grad_accum_steps" in t:
                train_cfg.grad_accum_steps = int(t["grad_accum_steps"])
            if "stride" in t:
                train_cfg.stride = int(t["stride"])
            if "learning_rate" in t:
                train_cfg.learning_rate = float(t["learning_rate"])
            if "min_lr" in t:
                train_cfg.min_lr = float(t["min_lr"])
            if "weight_decay" in t:
                train_cfg.weight_decay = float(t["weight_decay"])
            if "grad_clip" in t:
                train_cfg.grad_clip = float(t["grad_clip"])
            if "epochs" in t:
                train_cfg.epochs = int(t["epochs"])
            if "eval_interval" in t:
                train_cfg.eval_interval = int(t["eval_interval"])
            if "save_interval" in t:
                train_cfg.save_interval = int(t["save_interval"])

    return model_cfg, vision_cfg, train_cfg

default_model_config, default_vision_config, default_train_config = load_config()

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
</style>
</head>
<body>
<header>
  <h1>🦬 Nandi SLM</h1>
  <span class="badge">Gemma 4 Encoder-Free Vision • ~30M Params</span>
</header>
<div id="chat-container">
  <div class="msg assistant">👋 Hello! I am Nandi SLM. You can chat with me, or click <b>📎 Image</b> to upload a photo for pure image recognition (Note: if an image is uploaded, text input is locked).</div>
</div>
<div id="input-container">
  <div id="img-preview-bar">
    <img id="img-preview" src="">
    <span class="tag">⚠️ Image attached: Text input locked for pure image recognition.</span>
    <span class="remove-img" onclick="clearImage()">✕ Remove</span>
  </div>
  <div class="input-box">
    <label class="upload-btn" for="file-input">📎 Image</label>
    <input type="file" id="file-input" accept="image/*" onchange="handleImageSelect(event)">
    <input type="text" id="text-input" placeholder="Type a message..." onkeydown="if(event.key==='Enter') send()">
    <button class="btn" id="send-btn" onclick="send()">Send</button>
  </div>
</div>
<script>
let currentImageBase64 = null;
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
    textInput.placeholder = "Image attached (no text allowed). Click Recognize ->";
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
async function send() {
  const chat = document.getElementById("chat-container");
  const sendBtn = document.getElementById("send-btn");
  if (currentImageBase64) {
    const imgData = currentImageBase64;
    chat.innerHTML += `<div class="msg user"><img src="${imgData}"><br><i>[Image Recognition Request]</i></div>`;
    clearImage();
    sendBtn.disabled = true;
    chat.scrollTop = chat.scrollHeight;

    const res = await fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imgData })
    });
    const data = await res.json();
    chat.innerHTML += `<div class="msg assistant">${data.text || "[No description generated]"}</div>`;
    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  } else {
    const textInput = document.getElementById("text-input");
    const text = textInput.value.trim();
    if (!text) return;
    textInput.value = "";
    chat.innerHTML += `<div class="msg user">${text}</div>`;
    sendBtn.disabled = true;
    chat.scrollTop = chat.scrollHeight;

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    const data = await res.json();
    chat.innerHTML += `<div class="msg assistant">${data.text || "[No response]"}</div>`;
    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}
</script>
</body>
</html>"""

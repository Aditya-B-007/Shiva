import os
import sys
import io
import json
import base64
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.imageRecognitionForNandi import imageRecognitionEmbedder
from src.config import default_model_config, default_vision_config, HTML_PAGE


class ChatRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        self.POST_ROUTES = {
            "/recognize": self.handle_recognize,
            "/chat": self.handle_chat
        }
        super().__init__(*args, **kwargs)

    def _send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def handle_recognize(self, data):
        img_data = data.get("image", "")
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        img_bytes = base64.b64decode(img_data)
        img_stream = io.BytesIO(img_bytes)

        result = self.server.embedder.recognize_image(
            model=self.server.model,
            tokenizer=self.server.tokenizer,
            image_input=img_stream,
            device=self.server.device
        )
        self._send_json({"text": result if result else "[Model generated empty description]"})

    def handle_chat(self, data):
        text = data.get("text", "")
        formatted_prompt = f"User: {text}\nAssistant: "
        encoded = self.server.tokenizer.encode(formatted_prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.server.device)
        eos_id = getattr(self.server.tokenizer, "eos_token_id", None)
        if eos_id is None and hasattr(self.server.tokenizer, "tokenizer"):
            eos_id = self.server.tokenizer.tokenizer.token_to_id("</s>")

        output_ids = self.server.model.generate(
            idx=input_ids,
            max_new_tokens=250,
            temperature=0.7,
            eos_token_id=eos_id
        )
        resp_tokens = output_ids[0][input_ids.size(1):].tolist()
        response = self.server.tokenizer.decode(resp_tokens, skip_special_tokens=True)
        self._send_json({"text": response if response else "[No response]"})

    

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = json.loads(body.decode("utf-8")) if body else {}

        handler = self.POST_ROUTES.get(self.path)
        if not handler:
            self.send_error(404, "Endpoint not found")
            return

        handler(self, data)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_nandi():
    device = get_device()
    tokenizer = TokenizerNandi()
    if os.path.exists(tokenizer.model_path):
        tokenizer.load()
        vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = default_model_config.ntoken

    model = TransformerModel(
        ntoken=vocab_size,
        ninp=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        nlayers=default_model_config.nlayers,
        dropout=0.0
    ).to(device)

    embedder = imageRecognitionEmbedder(
        image_size=default_vision_config.image_size,
        patch_size=default_vision_config.patch_size,
        in_channels=default_vision_config.in_channels,
        d_model=default_model_config.ninp
    ).to(device)

    checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])
        if checkpoints:
            latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
            try:
                ckpt = torch.load(latest_ckpt, map_location=device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
            except Exception:
                pass

    model.eval()
    embedder.eval()
    return model, embedder, tokenizer, device


def start_server_and_open_ui(port=7860):
    print("\nInitializing Nandi SLM...")
    model, embedder, tokenizer, device = load_nandi()
    print("Model loaded successfully on device:", device)

    server = HTTPServer(("127.0.0.1", port), ChatRequestHandler)
    server.model = model
    server.embedder = embedder
    server.tokenizer = tokenizer
    server.device = device

    url = f"http://127.0.0.1:{port}"
    print(f"\nNandi SLM Web UI is running at {url}")
    print("Automatically opening your web browser...")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nNandi SLM Web UI stopped.")


if __name__ == "__main__":
    start_server_and_open_ui()

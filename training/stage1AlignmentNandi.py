import os
import sys
import json
import time
import math
import argparse
import torch
import torch.nn as nn
from PIL import Image
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import default_model_config, default_vision_config
from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.imageRecognitionForNandi import VisionEncoder, MLPProjector, VisionBridge
from src.dataIngestionPipeline import TokenSplicer, MultimodalInputBatch, IMultimodalSplicer


class Stage1Config:
    DEFAULT_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "multimodalTrain.jsonl"))
    DEFAULT_CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
    BASE_CHAT_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "nandi_chat_final.pt")
    OUTPUT_STAGE1_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "nandi_stage1_projector.pt")
    
    NINP = default_model_config.ninp
    NHEAD = default_model_config.nhead
    N_KV_HEADS = default_model_config.n_kv_heads
    NHID = default_model_config.nhid
    NLAYERS = default_model_config.nlayers
    DROPOUT = 0.05
    MAX_SEQ_LEN = 512
    
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 4
    LR = 1e-3           # Projector-only warmup alignment
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    EPOCHS = 8
    EVAL_INTERVAL = 10
    SAVE_INTERVAL = 50


class ILossEvaluator(ABC):
    @abstractmethod
    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


class MaskedCausalLMLoss(ILossEvaluator):
    def __init__(self, ignore_index: int = -100, shift_labels: bool = True):
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.shift_labels = shift_labels

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.shift_labels:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        else:
            shift_logits = logits
            shift_labels = labels

        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )


class ITrainingStageConfigurator(ABC):
    @abstractmethod
    def configure(self, bridge: VisionBridge, slm: nn.Module, lr: float) -> Optimizer:
        pass


class Stage1AlignmentConfigurator(ITrainingStageConfigurator):
    def configure(self, bridge: VisionBridge, slm: nn.Module, lr: float) -> Optimizer:
        for param in slm.parameters():
            param.requires_grad = False
        for param in bridge.encoder.parameters():
            param.requires_grad = False
        for param in bridge.projector.parameters():
            param.requires_grad = True

        print("[+] Stage 1 Configuration: SLM backbone & Vision encoder FROZEN. Training MLP Projector only.")
        return AdamW(bridge.projector.parameters(), lr=lr, weight_decay=Stage1Config.WEIGHT_DECAY)


class MultimodalTrainer:
    def __init__(
        self,
        slm_model: nn.Module,
        vision_bridge: VisionBridge,
        splicer: IMultimodalSplicer,
        loss_evaluator: ILossEvaluator,
        optimizer: Optimizer,
        image_token_id: int
    ):
        self.slm = slm_model
        self.bridge = vision_bridge
        self.splicer = splicer
        self.loss_evaluator = loss_evaluator
        self.optimizer = optimizer
        self.image_token_id = image_token_id

    def execute_step(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        self.bridge.projector.train()
        visual_tokens = self.bridge(pixel_values)
        embedding_layer = self.slm.get_input_embeddings()
        ninp = getattr(self.slm, "ninp", visual_tokens.size(-1))
        scale = math.sqrt(ninp)
        text_embeddings = embedding_layer(input_ids) * scale

        batch_contract = MultimodalInputBatch(
            input_ids=input_ids,
            text_embeddings=text_embeddings,
            visual_tokens=visual_tokens,
            image_token_id=self.image_token_id,
            labels=labels
        )
        spliced = self.splicer.splice(batch_contract)
        outputs = self.slm(inputs_embeds=spliced.embeddings)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = self.loss_evaluator.evaluate(logits, spliced.labels)

        return loss


class MultimodalDataset(Dataset):
    def __init__(self, data_path: Optional[str], tokenizer: TokenizerNandi, image_processor, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []

        self.pad_token_id = tokenizer.tokenizer.token_to_id("<pad>") or 1
        self.image_token_id = tokenizer.tokenizer.token_to_id("<image>")
        if self.image_token_id is None:
            tokenizer.add_special_tokens(["<image>"])
            self.image_token_id = tokenizer.tokenizer.token_to_id("<image>")

        if data_path and os.path.exists(data_path):
            print(f"[+] Loading multimodal dataset from: {data_path}")
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
            print(f"[+] Loaded {len(self.samples)} multimodal samples.")
        else:
            print("[!] No multimodal dataset file found. Generating synthetic samples for alignment & verification.")
            self._generate_synthetic_samples()

    def _generate_synthetic_samples(self, count: int = 16):
        colors = [
            ((255, 0, 0), "red"),
            ((0, 255, 0), "green"),
            ((0, 0, 255), "blue"),
            ((255, 255, 0), "yellow"),
        ]
        for i in range(count):
            rgb, color_name = colors[i % len(colors)]
            self.samples.append({
                "synthetic_rgb": rgb,
                "prompt": f"User: What color is displayed in this image: <image>?\nAssistant: <|thought|>\nAnalyzing visual features.\n<|thought|>\nThis image depicts the color {color_name}.</s>"
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record = self.samples[idx]
        if "synthetic_rgb" in record:
            img = Image.new("RGB", (224, 224), color=record["synthetic_rgb"])
        elif "image_path" in record and os.path.exists(record["image_path"]):
            img = Image.open(record["image_path"]).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        pixel_values = self.image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        prompt_text = record.get("prompt", "User: Describe this: <image>\nAssistant: An image.</s>")
        encoded = self.tokenizer.encode(prompt_text)
        token_ids = encoded.ids

        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]

        # In unshifted sequence modeling, labels = token_ids (MaskedCausalLMLoss will shift by 1)
        labels = list(token_ids)

        # Mask user prompt from loss calculation if formatted with Assistant:
        if "Assistant: " in prompt_text:
            user_part = prompt_text.split("Assistant: ")[0] + "Assistant: "
            user_len = len(self.tokenizer.encode(user_part).ids)
            for j in range(min(user_len, len(labels))):
                labels[j] = -100

        pad_len = self.max_seq_len - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            pixel_values,
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )


def get_device_and_dtype():
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


def train_stage1(data_path: Optional[str] = None, base_checkpoint: Optional[str] = None, epochs: int = Stage1Config.EPOCHS, lr: Optional[float] = None):
    device, use_amp, amp_dtype = get_device_and_dtype()
    print("\n" + "=" * 65)
    print("  STAGE 1: NANDI SLM PROJECTOR ALIGNMENT TRAINING")
    print("  (SLM backbone & Vision Encoder FROZEN, MLP Projector TRAINED)")
    print("=" * 65)

    tokenizer = TokenizerNandi()
    tokenizer.load()
    if tokenizer.tokenizer.token_to_id("<image>") is None:
        tokenizer.add_special_tokens(["<image>"])
    vocab_size = tokenizer.get_vocab_size()
    image_token_id = tokenizer.tokenizer.token_to_id("<image>")
    print(f"[+] Loaded Tokenizer | Vocab Size: {vocab_size:,} | '<image>' ID: {image_token_id}")

    # 1. Initialize Vision Bridge
    print("[+] Loading SigLIP Vision Encoder & MLP Projector...")
    vision_encoder = VisionEncoder().to(device)
    mlp_projector = MLPProjector(visual_dim=vision_encoder.hidden_dim, language_dim=Stage1Config.NINP).to(device)
    bridge = VisionBridge(encoder=vision_encoder, projector=mlp_projector).to(device)

    # 2. Initialize Language Model Backbone
    model = TransformerModel(
        ntoken=vocab_size,
        ninp=Stage1Config.NINP,
        nhead=Stage1Config.NHEAD,
        n_kv_heads=Stage1Config.N_KV_HEADS,
        nhid=Stage1Config.NHID,
        nlayers=Stage1Config.NLAYERS,
        dropout=Stage1Config.DROPOUT,
        max_seq_len=Stage1Config.MAX_SEQ_LEN
    ).to(device)

    # 3. Load pre-trained weights if provided
    ckpt_to_load = base_checkpoint or Stage1Config.BASE_CHAT_CHECKPOINT
    if not os.path.exists(ckpt_to_load):
        fallback_ckpts = sorted([
            os.path.join(Stage1Config.DEFAULT_CHECKPOINT_DIR, f)
            for f in os.listdir(Stage1Config.DEFAULT_CHECKPOINT_DIR)
            if f.endswith(".pt")
        ]) if os.path.exists(Stage1Config.DEFAULT_CHECKPOINT_DIR) else []
        if fallback_ckpts:
            ckpt_to_load = fallback_ckpts[-1]
            print(f"[!] Target chat checkpoint not found. Falling back to latest available: {ckpt_to_load}")
        else:
            print("[!] No prior language checkpoint found. Initializing with base weights.")
            ckpt_to_load = None

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"[+] Loading language checkpoint weights from: {ckpt_to_load}")
        checkpoint = torch.load(ckpt_to_load, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    # 4. Configure Stage 1 Optimizer
    configurator = Stage1AlignmentConfigurator()
    learning_rate = lr or Stage1Config.LR
    optimizer = configurator.configure(bridge, model, lr=learning_rate)
    loss_evaluator = MaskedCausalLMLoss(ignore_index=-100, shift_labels=True)
    splicer = TokenSplicer()

    trainer = MultimodalTrainer(
        slm_model=model,
        vision_bridge=bridge,
        splicer=splicer,
        loss_evaluator=loss_evaluator,
        optimizer=optimizer,
        image_token_id=image_token_id
    )

    # 5. Dataset & DataLoader
    resolved_data_path = data_path or Stage1Config.DEFAULT_DATA_PATH
    dataset = MultimodalDataset(
        data_path=resolved_data_path,
        tokenizer=tokenizer,
        image_processor=vision_encoder.processor,
        max_seq_len=Stage1Config.MAX_SEQ_LEN
    )
    dataloader = DataLoader(dataset, batch_size=Stage1Config.BATCH_SIZE, shuffle=True, drop_last=False)
    total_opt_steps = (len(dataloader) // Stage1Config.GRAD_ACCUM_STEPS) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_opt_steps, 1), eta_min=Stage1Config.MIN_LR)

    print(f"[+] Dataset size: {len(dataset)} samples | Batch Size: {Stage1Config.BATCH_SIZE} | Epochs: {epochs}")
    print(f"[+] Effective Batch Size: {Stage1Config.BATCH_SIZE * Stage1Config.GRAD_ACCUM_STEPS}")

    # 6. Training Loop
    os.makedirs(Stage1Config.DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    global_step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"\n--- Stage 1 Projector Alignment | Epoch {epoch + 1}/{epochs} ---")

        for step, (pixel_values, input_ids, labels) in enumerate(dataloader):
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = trainer.execute_step(pixel_values, input_ids, labels)
                    loss = loss / Stage1Config.GRAD_ACCUM_STEPS
            else:
                loss = trainer.execute_step(pixel_values, input_ids, labels)
                loss = loss / Stage1Config.GRAD_ACCUM_STEPS

            loss.backward()

            if (step + 1) % Stage1Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    bridge.projector.parameters(),
                    max_norm=Stage1Config.GRAD_CLIP
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if device.type == "mps" and global_step % 20 == 0:
                    torch.mps.empty_cache()

                if global_step % Stage1Config.EVAL_INTERVAL == 0 or global_step == 1:
                    curr_loss = loss.item() * Stage1Config.GRAD_ACCUM_STEPS
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    speed = global_step / max(elapsed, 0.001)
                    print(f"Step {global_step:04d} | Loss: {curr_loss:.4f} | LR: {current_lr:.2e} | Speed: {speed:.2f} opt_steps/s")

            epoch_loss += loss.item() * Stage1Config.GRAD_ACCUM_STEPS

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f}")

    # 7. Save Final Stage 1 Checkpoint
    final_out = Stage1Config.OUTPUT_STAGE1_CHECKPOINT
    torch.save({
        "stage": 1,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "projector_state_dict": bridge.projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, final_out)
    print(f"\n[+] Stage 1 Alignment Complete! Checkpoint saved to: {final_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nandi SLM Stage 1 Projector Alignment Training")
    parser.add_argument("--data_path", type=str, default=Stage1Config.DEFAULT_DATA_PATH, help="Path to multimodal training JSONL dataset.")
    parser.add_argument("--base_checkpoint", type=str, default=None, help="Path to base language checkpoint (e.g. nandi_chat_final.pt).")
    parser.add_argument("--epochs", type=int, default=Stage1Config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    args = parser.parse_args()

    train_stage1(
        data_path=args.data_path,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        lr=args.lr
    )


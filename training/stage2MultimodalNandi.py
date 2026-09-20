import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    default_model_config,
    default_stage2_config,
    getDeviceAndDtype,
)
from src.interfaces import ITrainingStageConfigurator
from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.imageRecognitionForNandi import VisionEncoder, MLPProjector, VisionBridge
from src.dataIngestionPipeline import TokenSplicer
from training.multimodalTrainer import MultimodalTrainer, MaskedCausalLMLoss
from training.multimodalDataset import MultimodalDataset


# =============================================================================
# Paths
# =============================================================================

DEFAULT_DATA_PATH        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "multimodalTrain.jsonl"))
DEFAULT_CHECKPOINT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
STAGE1_PROJECTOR_CHECKPOINT = os.path.join(DEFAULT_CHECKPOINT_DIR, "nandi_stage1_projector.pt")
BASE_CHAT_CHECKPOINT     = os.path.join(DEFAULT_CHECKPOINT_DIR, "nandi_chat_final.pt")
OUTPUT_FINAL_CHECKPOINT  = os.path.join(DEFAULT_CHECKPOINT_DIR, "nandi_vision_final.pt")


# =============================================================================
# Stage 2 configurator — freezes encoder, trains SLM + projector jointly
# =============================================================================

class Stage2InstructionConfigurator(ITrainingStageConfigurator):
    def configure(self, bridge: VisionBridge, slm: nn.Module, lr: float) -> Optimizer:
        for param in bridge.encoder.parameters():
            param.requires_grad = False
        for param in bridge.projector.parameters():
            param.requires_grad = True
        for param in slm.parameters():
            param.requires_grad = True

        print("[+] Stage 2 Configuration: Vision encoder FROZEN. Jointly fine-tuning SLM + MLP Projector.")
        trainable_params = [
            {"params": bridge.projector.parameters(), "lr": lr},
            {"params": slm.parameters(), "lr": lr * 0.5}
        ]
        return AdamW(trainable_params, weight_decay=default_stage2_config.weightDecay)


# =============================================================================
# Main training function
# =============================================================================

def train_stage2(
    data_path=None,
    base_checkpoint=None,
    epochs=None,
    lr=None
):
    cfg = default_stage2_config
    epochs = epochs or cfg.epochs
    learning_rate = lr or cfg.learningRate

    device, use_amp, amp_dtype = getDeviceAndDtype()
    print("\n" + "=" * 65)
    print("  STAGE 2: NANDI SLM JOINT MULTIMODAL FINE-TUNING")
    print("  (Vision Encoder FROZEN, SLM backbone + MLP Projector JOINTLY TRAINED)")
    print("=" * 65)

    tokenizer = TokenizerNandi()
    tokenizer.load()
    if tokenizer.tokenizer.token_to_id("<image>") is None:
        tokenizer.addSpecialTokens(["<image>"])
    vocab_size = tokenizer.getVocabSize()
    image_token_id = tokenizer.tokenizer.token_to_id("<image>")
    print(f"[+] Loaded Tokenizer | Vocab Size: {vocab_size:,} | '<image>' ID: {image_token_id}")

    # 1. Vision Bridge
    print("[+] Loading SigLIP Vision Encoder & MLP Projector...")
    vision_encoder = VisionEncoder().to(device)
    mlp_projector = MLPProjector(visual_dim=vision_encoder.hiddenDim, language_dim=default_model_config.ninp).to(device)
    bridge = VisionBridge(encoder=vision_encoder, projector=mlp_projector).to(device)

    # 2. Language Model Backbone
    model = TransformerModel(
        ntoken=vocab_size,
        ninp=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        nlayers=default_model_config.nlayers,
        dropout=cfg.dropout,
        max_seq_len=cfg.maxSeqLen
    ).to(device)

    # 3. Load pre-trained weights
    # Priority: Stage1 projector checkpoint → base chat checkpoint → fallback
    ckpt_to_load = base_checkpoint or STAGE1_PROJECTOR_CHECKPOINT
    if not os.path.exists(ckpt_to_load):
        ckpt_to_load = BASE_CHAT_CHECKPOINT

    if not os.path.exists(ckpt_to_load):
        fallback_ckpts = sorted([
            os.path.join(DEFAULT_CHECKPOINT_DIR, f)
            for f in os.listdir(DEFAULT_CHECKPOINT_DIR)
            if f.endswith(".pt")
        ]) if os.path.exists(DEFAULT_CHECKPOINT_DIR) else []
        if fallback_ckpts:
            ckpt_to_load = fallback_ckpts[-1]
            print(f"[!] Target checkpoint not found. Falling back to latest: {ckpt_to_load}")
        else:
            print("[!] No prior checkpoint found. Initializing with base weights.")
            ckpt_to_load = None

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"[+] Loading checkpoint weights from: {ckpt_to_load}")
        checkpoint = torch.load(ckpt_to_load, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        if "projector_state_dict" in checkpoint:
            bridge.projector.load_state_dict(checkpoint["projector_state_dict"])
            print("[+] Successfully loaded pre-trained MLP Projector weights.")

    # 4. Configure Stage 2 optimizer
    configurator = Stage2InstructionConfigurator()
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
    resolved_data_path = data_path or DEFAULT_DATA_PATH
    dataset = MultimodalDataset(
        data_path=resolved_data_path,
        tokenizer=tokenizer,
        image_processor=vision_encoder.processor,
        max_seq_len=cfg.maxSeqLen
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batchSize, shuffle=True, drop_last=False)
    total_opt_steps = (len(dataloader) // cfg.gradAccumSteps) * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_opt_steps, 1), eta_min=cfg.minLr)

    print(f"[+] Dataset size: {len(dataset)} samples | Batch Size: {cfg.batchSize} | Epochs: {epochs}")
    print(f"[+] Effective Batch Size: {cfg.batchSize * cfg.gradAccumSteps}")

    # 6. Training loop
    os.makedirs(DEFAULT_CHECKPOINT_DIR, exist_ok=True)
    global_step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"\n--- Stage 2 Joint Fine-Tuning | Epoch {epoch + 1}/{epochs} ---")

        for step, (pixel_values, input_ids, labels) in enumerate(dataloader):
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = trainer.execute_step(pixel_values, input_ids, labels)
                    loss = loss / cfg.gradAccumSteps
            else:
                loss = trainer.execute_step(pixel_values, input_ids, labels)
                loss = loss / cfg.gradAccumSteps

            loss.backward()

            if (step + 1) % cfg.gradAccumSteps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    list(bridge.projector.parameters()) + list(model.parameters()),
                    max_norm=cfg.gradClip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if device.type == "mps" and global_step % 20 == 0:
                    torch.mps.empty_cache()

                if global_step % cfg.evalInterval == 0 or global_step == 1:
                    curr_loss = loss.item() * cfg.gradAccumSteps
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    speed = global_step / max(elapsed, 0.001)
                    print(f"Step {global_step:04d} | Loss: {curr_loss:.4f} | LR: {current_lr:.2e} | Speed: {speed:.2f} opt_steps/s")

            epoch_loss += loss.item() * cfg.gradAccumSteps

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f}")

    # 7. Save final multimodal checkpoint
    torch.save({
        "stage": 2,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "projector_state_dict": bridge.projector.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, OUTPUT_FINAL_CHECKPOINT)
    print(f"\n[+] Multimodal Training Complete! Checkpoint saved to: {OUTPUT_FINAL_CHECKPOINT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nandi SLM Stage 2 Joint Multimodal Fine-Tuning")
    parser.add_argument("--data_path",       type=str,   default=DEFAULT_DATA_PATH,   help="Path to multimodal training JSONL dataset.")
    parser.add_argument("--base_checkpoint", type=str,   default=None,                help="Path to base checkpoint (defaults to nandi_stage1_projector.pt).")
    parser.add_argument("--epochs",          type=int,   default=default_stage2_config.epochs, help="Number of training epochs.")
    parser.add_argument("--lr",              type=float, default=None,                help="Learning rate override.")
    args = parser.parse_args()

    train_stage2(
        data_path=args.data_path,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
        lr=args.lr
    )

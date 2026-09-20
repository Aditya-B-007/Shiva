import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.dataIngestionPipeline import get_data_loader
from src.config import default_model_config, default_train_config, getDeviceAndDtype


# =============================================================================
# Paths (stage-specific; all numerics come from default_train_config)
# =============================================================================

CORPUS_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.txt"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))


def train():
    cfg = default_train_config
    device, use_amp, amp_dtype = getDeviceAndDtype()
    print(f"Starting Training for Nandi SLM on Device: {device}")

    tokenizer = TokenizerNandi()
    if not os.path.exists(tokenizer.model_path):
        print("Tokenizer model not found! Please train tokenizer first via python3 src/tokenization.py")
        sys.exit(1)
    tokenizer.load()
    if tokenizer.tokenizer.token_to_id("<image>") is None:
        tokenizer.addSpecialTokens(["<image>"])
    vocab_size = tokenizer.getVocabSize()
    print(f"Loaded Tokenizer with Vocab Size: {vocab_size:,}")

    model = TransformerModel(
        ntoken=vocab_size,
        ninp=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        nlayers=default_model_config.nlayers,
        dropout=default_model_config.dropout,
        max_seq_len=default_model_config.max_seq_len
    ).to(device)

    unique_params = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)
    print(f"Model Parameters: {total_params:,} ({total_params / 1e6:.2f} Million)")

    print(f"Building DataLoader from: {CORPUS_PATH}...")
    dataloader = get_data_loader(
        corpus_path=CORPUS_PATH,
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        max_length=default_model_config.max_seq_len,
        stride=cfg.stride,
        shuffle=True
    )
    total_optimizer_steps = (len(dataloader) // cfg.grad_accum_steps) * cfg.epochs
    print(f"Batches per Epoch: {len(dataloader):,}")
    print(f"Effective Batch Size: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"Total Optimizer Steps: {total_optimizer_steps:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=cfg.min_lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    global_step = 0
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        print(f"\n--- Epoch {epoch + 1}/{cfg.epochs} ---")

        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                    loss = loss / cfg.grad_accum_steps
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / cfg.grad_accum_steps

            loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if device.type == "mps" and global_step % 50 == 0:
                    torch.mps.empty_cache()

                if global_step % cfg.eval_interval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    curr_loss = loss.item() * cfg.grad_accum_steps
                    print(f"Step {global_step:05d}/{total_optimizer_steps:05d} | Loss: {curr_loss:.4f} | LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} opt_steps/s")

                if global_step % cfg.save_interval == 0:
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"nandi_step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item() * cfg.grad_accum_steps,
                    }, checkpoint_path)
                    print(f">> Saved Checkpoint: {checkpoint_path}")

            epoch_loss += loss.item() * cfg.grad_accum_steps

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_epoch_loss:.4f}")

    final_path = os.path.join(CHECKPOINT_DIR, "nandi_final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_epoch_loss,
    }, final_path)
    print(f"\nTraining Complete! Final model weights saved to: {final_path}")


if __name__ == "__main__":
    train()

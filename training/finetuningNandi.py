import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenization import TokenizerNandi
from src.transformer import TransformerModel
from src.config import default_model_config, default_sft_config, getDeviceAndDtype, DEFAULT_PAD_TOKEN_ID


# =============================================================================
# Paths
# =============================================================================

DATA_PATH        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "nandiTrain.jsonl"))
BASE_CHECKPOINT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints", "nandi_final.pt"))
OUTPUT_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))


# =============================================================================
# SFT Dataset
# =============================================================================

class NandiQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=None):
        cfg = default_sft_config
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len or cfg.maxSeqLen
        self.samples = []

        pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        self.pad_token_id = pad_token_id if pad_token_id is not None else DEFAULT_PAD_TOKEN_ID

        print(f"Loading Q&A dataset from {jsonl_path}...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                self._process_record(record)

        print(f"Loaded {len(self.samples)} valid Q&A instruction pairs.")

    def _process_record(self, record):
        prompt = record.get("human_prompt", "").strip()
        thought = record.get("thought", "").strip()
        response_text = record.get("assistant_response", "").strip()

        if not response_text.endswith("</s>"):
            response_text = response_text + "</s>"

        user_prompt_text = f"User: {prompt}\nAssistant: <|thought|>\n"
        assistant_full_response = f"{thought}\n<|thought|>\n{response_text}"

        user_prompt_ids = self.tokenizer.encode(user_prompt_text).ids
        assistant_ids = self.tokenizer.encode(assistant_full_response).ids

        full_ids = user_prompt_ids + assistant_ids
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[:self.max_seq_len]
        prompt_len = min(len(user_prompt_ids), len(full_ids))
        target_ids = [-100] * prompt_len + full_ids[prompt_len:]
        input_chunk = full_ids[:-1]
        target_chunk = target_ids[1:]
        pad_len = (self.max_seq_len - 1) - len(input_chunk)
        if pad_len > 0:
            input_chunk = input_chunk + [self.pad_token_id] * pad_len
            target_chunk = target_chunk + [-100] * pad_len

        self.samples.append((
            torch.tensor(input_chunk, dtype=torch.long),
            torch.tensor(target_chunk, dtype=torch.long)
        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Fine-tuning entry point
# =============================================================================

def finetune():
    cfg = default_sft_config
    device, use_amp, amp_dtype = getDeviceAndDtype()
    print("Starting Supervised Fine-Tuning (SFT) on Q&A Data")

    tokenizer = TokenizerNandi()
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
        dropout=cfg.dropout,
        max_seq_len=cfg.maxSeqLen
    ).to(device)

    ckpt_path = BASE_CHECKPOINT
    if not os.path.exists(ckpt_path):
        candidates = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("nandi_") and f.endswith(".pt")])
        if candidates:
            ckpt_path = os.path.join(OUTPUT_DIR, candidates[-1])
        else:
            print(f"Error: No pre-trained base model found in {OUTPUT_DIR}")
            sys.exit(1)

    print(f"Loading base model weights from: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Successfully transferred pre-trained weights to Q&A model!")

    dataset = NandiQADataset(DATA_PATH, tokenizer, max_seq_len=cfg.maxSeqLen)
    dataloader = DataLoader(dataset, batch_size=cfg.batchSize, shuffle=True, drop_last=True)

    total_optimizer_steps = (len(dataloader) // cfg.gradAccumSteps) * cfg.epochs
    print(f"Total Batches per Epoch: {len(dataloader):,}")
    print(f"Total SFT Optimization Steps: {total_optimizer_steps:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learningRate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weightDecay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=cfg.minLr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        print(f"\n--- SFT Epoch {epoch + 1}/{cfg.epochs} ---")

        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                    loss = loss / cfg.gradAccumSteps
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / cfg.gradAccumSteps

            loss.backward()

            if (step + 1) % cfg.gradAccumSteps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradClip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if device.type == "mps" and global_step % 25 == 0:
                    torch.mps.empty_cache()

                if global_step % cfg.evalInterval == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    curr_loss = loss.item() * cfg.gradAccumSteps
                    print(f"Step {global_step:04d}/{total_optimizer_steps:04d} | SFT Loss: {curr_loss:.4f} | LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s")

                if global_step % cfg.saveInterval == 0:
                    ckpt_out = os.path.join(OUTPUT_DIR, f"nandi_qa_step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item() * cfg.gradAccumSteps,
                    }, ckpt_out)
                    print(f">> Saved SFT Checkpoint: {ckpt_out}")

            epoch_loss += loss.item() * cfg.gradAccumSteps

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f}")

    final_path = os.path.join(OUTPUT_DIR, "nandi_chat_final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, final_path)
    print(f"\nFine-Tuning Complete! Chat model saved to: {final_path}")


if __name__ == "__main__":
    finetune()

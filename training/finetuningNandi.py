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
from src.config import default_model_config

class SFTConfig:
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "nandiTrain.jsonl"))
    BASE_CHECKPOINT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints", "nandi_final.pt"))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
    
    # Model Specs from config
    NINP = default_model_config.ninp
    NHEAD = default_model_config.nhead
    N_KV_HEADS = default_model_config.n_kv_heads
    NHID = default_model_config.nhid
    NLAYERS = default_model_config.nlayers
    DROPOUT = 0.05
    MAX_SEQ_LEN = 512 
    
    # Fine-Tuning Hyperparameters
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 4 
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-5
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    EPOCHS = 3 
    EVAL_INTERVAL = 25
    SAVE_INTERVAL = 100

class NandiQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        if pad_token_id is None:
            pad_token_id = 1
        self.pad_token_id = pad_token_id

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

def get_device_and_dtype():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = True
        amp_dtype = torch.bfloat16
        print(">> Fine-tuning on Apple Silicon Metal Performance Shaders (MPS) with AMP!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        use_amp = False
        amp_dtype = torch.float32
        print(">> Fine-tuning on CPU.")
    return device, use_amp, amp_dtype

def finetune():
    device, use_amp, amp_dtype = get_device_and_dtype()
    print("Starting Supervised Fine-Tuning (SFT) on Q&A Data")

    tokenizer = TokenizerNandi()
    tokenizer.load()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded Tokenizer with Vocab Size: {vocab_size:,}")

    model = TransformerModel(
        ntoken=vocab_size,
        ninp=SFTConfig.NINP,
        nhead=SFTConfig.NHEAD,
        n_kv_heads=SFTConfig.N_KV_HEADS,
        nhid=SFTConfig.NHID,
        nlayers=SFTConfig.NLAYERS,
        dropout=SFTConfig.DROPOUT,
        max_seq_len=SFTConfig.MAX_SEQ_LEN
    ).to(device)

    ckpt_path = SFTConfig.BASE_CHECKPOINT
    if not os.path.exists(ckpt_path):
        candidates = sorted([f for f in os.listdir(SFTConfig.OUTPUT_DIR) if f.startswith("nandi_") and f.endswith(".pt")])
        if candidates:
            ckpt_path = os.path.join(SFTConfig.OUTPUT_DIR, candidates[-1])
        else:
            print(f"Error: No pre-trained base model found in {SFTConfig.OUTPUT_DIR}")
            sys.exit(1)

    print(f"Loading base model weights from: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Successfully transferred pre-trained weights to Q&A model!")

    dataset = NandiQADataset(SFTConfig.DATA_PATH, tokenizer, max_seq_len=SFTConfig.MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=SFTConfig.BATCH_SIZE, shuffle=True, drop_last=True)

    total_optimizer_steps = (len(dataloader) // SFTConfig.GRAD_ACCUM_STEPS) * SFTConfig.EPOCHS
    print(f"Total Batches per Epoch: {len(dataloader):,}")
    print(f"Total SFT Optimization Steps: {total_optimizer_steps:,}")
    optimizer = AdamW(
        model.parameters(),
        lr=SFTConfig.LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=SFTConfig.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=SFTConfig.MIN_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(SFTConfig.EPOCHS):
        epoch_loss = 0.0
        print(f"\n--- SFT Epoch {epoch + 1}/{SFTConfig.EPOCHS} ---")

        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                    loss = loss / SFTConfig.GRAD_ACCUM_STEPS
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / SFTConfig.GRAD_ACCUM_STEPS

            loss.backward()

            if (step + 1) % SFTConfig.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=SFTConfig.GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if device.type == "mps" and global_step % 25 == 0:
                    torch.mps.empty_cache()

                if global_step % SFTConfig.EVAL_INTERVAL == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    curr_loss = loss.item() * SFTConfig.GRAD_ACCUM_STEPS
                    print(f"Step {global_step:04d}/{total_optimizer_steps:04d} | SFT Loss: {curr_loss:.4f} | LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s")

                if global_step % SFTConfig.SAVE_INTERVAL == 0:
                    ckpt_out = os.path.join(SFTConfig.OUTPUT_DIR, f"nandi_qa_step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item() * SFTConfig.GRAD_ACCUM_STEPS,
                    }, ckpt_out)
                    print(f">> Saved SFT Checkpoint: {ckpt_out}")

            epoch_loss += loss.item() * SFTConfig.GRAD_ACCUM_STEPS

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_loss:.4f}")

    final_path = os.path.join(SFTConfig.OUTPUT_DIR, "nandi_chat_final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
    }, final_path)
    print(f"\nFine-Tuning Complete! Chat model saved to: {final_path}")

if __name__ == "__main__":
    finetune()

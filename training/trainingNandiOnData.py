import os
import sys
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenization import TokenizerNandi, Config as TokenizerConfig
from src.transformer import TransformerModel
from src.dataIngestionPipeline import get_data_loader

class TrainConfig:
    CORPUS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.txt"))
    CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_artifacts", "checkpoints"))
    
    # Model Hyperparameters (Scaled down for cool & fast Mac training: ~28M params)
    NINP = 512           
    NHEAD = 8           
    NHID = 2048          
    NLAYERS = 8        
    DROPOUT = 0.1
    MAX_SEQ_LEN = 256    
    
    # Training Loop Parameters
    BATCH_SIZE = 8      
    GRAD_ACCUM_STEPS = 2 
    STRIDE = 128
    LEARNING_RATE = 5e-4
    MIN_LR = 5e-5
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    EPOCHS = 2         
    EVAL_INTERVAL = 50
    SAVE_INTERVAL = 250

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

def train():
    device, use_amp, amp_dtype = get_device_and_dtype()
    print(f"Starting Training for Nandi SLM on Device: {device}")

    tokenizer = TokenizerNandi()
    if not os.path.exists(tokenizer.model_path):
        print("Tokenizer model not found! Please train tokenizer first via python3 src/tokenization.py")
        sys.exit(1)
    tokenizer.load()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded Tokenizer with Vocab Size: {vocab_size:,}")

    model = TransformerModel(
        ntoken=vocab_size,
        ninp=TrainConfig.NINP,
        nhead=TrainConfig.NHEAD,
        nhid=TrainConfig.NHID,
        nlayers=TrainConfig.NLAYERS,
        dropout=TrainConfig.DROPOUT,
        max_seq_len=TrainConfig.MAX_SEQ_LEN
    ).to(device)

    unique_params = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)
    print(f"Model Parameters: {total_params:,} ({total_params / 1e6:.2f} Million)")

    print(f"Building DataLoader from: {TrainConfig.CORPUS_PATH}...")
    dataloader = get_data_loader(
        corpus_path=TrainConfig.CORPUS_PATH,
        tokenizer=tokenizer,
        batch_size=TrainConfig.BATCH_SIZE,
        max_length=TrainConfig.MAX_SEQ_LEN,
        stride=TrainConfig.STRIDE,
        shuffle=True
    )
    total_optimizer_steps = (len(dataloader) // TrainConfig.GRAD_ACCUM_STEPS) * TrainConfig.EPOCHS
    print(f"Batches per Epoch: {len(dataloader):,}")
    print(f"Effective Batch Size: {TrainConfig.BATCH_SIZE * TrainConfig.GRAD_ACCUM_STEPS}")
    print(f"Total Optimizer Steps: {total_optimizer_steps:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=TrainConfig.LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=TrainConfig.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=TrainConfig.MIN_LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(TrainConfig.CHECKPOINT_DIR, exist_ok=True)

    global_step = 0
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(TrainConfig.EPOCHS):
        epoch_loss = 0.0
        print(f"\n--- Epoch {epoch + 1}/{TrainConfig.EPOCHS} ---")

        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                    loss = loss / TrainConfig.GRAD_ACCUM_STEPS
            else:
                logits = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                loss = loss / TrainConfig.GRAD_ACCUM_STEPS

            loss.backward()

            if (step + 1) % TrainConfig.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainConfig.GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if device.type == "mps" and global_step % 50 == 0:
                    torch.mps.empty_cache()

                if global_step % TrainConfig.EVAL_INTERVAL == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    curr_loss = loss.item() * TrainConfig.GRAD_ACCUM_STEPS
                    print(f"Step {global_step:05d}/{total_optimizer_steps:05d} | Loss: {curr_loss:.4f} | LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} opt_steps/s")

                if global_step % TrainConfig.SAVE_INTERVAL == 0:
                    checkpoint_path = os.path.join(TrainConfig.CHECKPOINT_DIR, f"nandi_step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item() * TrainConfig.GRAD_ACCUM_STEPS,
                    }, checkpoint_path)
                    print(f">> Saved Checkpoint: {checkpoint_path}")

            epoch_loss += loss.item() * TrainConfig.GRAD_ACCUM_STEPS

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete | Average Loss: {avg_epoch_loss:.4f}")

    final_path = os.path.join(TrainConfig.CHECKPOINT_DIR, "nandi_final.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_epoch_loss,
    }, final_path)
    print(f"\nTraining Complete! Final model weights saved to: {final_path}")

if __name__ == "__main__":
    train()

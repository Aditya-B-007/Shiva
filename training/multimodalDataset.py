"""
training/multimodalDataset.py
==============================
Shared multimodal Dataset used by both Stage 1 and Stage 2 training scripts.
Avoids code duplication between stage scripts.
"""

import os
import json
import torch
from PIL import Image
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset

from src.interfaces import ITokenizer
from src.config import DEFAULT_PAD_TOKEN_ID


class MultimodalDataset(Dataset):
    """
    Loads vision+language training samples from a JSONL file.
    Falls back to synthetic colour-patch samples when no file is present
    (useful for smoke-testing the pipeline without real data).
    """

    def __init__(
        self,
        data_path: Optional[str],
        tokenizer: ITokenizer,
        image_processor,
        max_seq_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, Any]] = []

        self.pad_token_id = tokenizer.tokenizer.token_to_id("<pad>") or DEFAULT_PAD_TOKEN_ID
        self.image_token_id = tokenizer.tokenizer.token_to_id("<image>")
        if self.image_token_id is None:
            tokenizer.addSpecialTokens(["<image>"])
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
                "prompt": (
                    f"User: What color is displayed in this image: <image>?\n"
                    f"Assistant: <|thought|>\nAnalyzing visual features.\n"
                    f"<|thought|>\nThis image depicts the color {color_name}.</s>"
                )
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

        # Labels mirror token_ids; MaskedCausalLMLoss will shift by 1.
        labels = list(token_ids)

        # Mask the user prompt portion so loss is computed on assistant text only.
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

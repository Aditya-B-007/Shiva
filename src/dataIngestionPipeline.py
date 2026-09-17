import os
import torch
from torch.utils.data import Dataset, DataLoader
try:
    from .tokenization import TokenizerNandi, Config as TokenizerConfig
except ImportError:
    from tokenization import TokenizerNandi, Config as TokenizerConfig
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import torch

class Config:
    CORPUS_PATH = os.getenv("CORPUS_PATH", TokenizerConfig.CORPUS_PATH)
    MAX_LENGTH = 256
    STRIDE = 128
    BATCH_SIZE = 4
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


@dataclass(frozen=True)
class MultimodalInputBatch:
    input_ids: torch.Tensor
    text_embeddings: torch.Tensor
    visual_tokens: torch.Tensor
    image_token_id: int
    labels: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class SplicedMultimodalOutput:
    embeddings: torch.Tensor
    labels: Optional[torch.Tensor] = None


class IMultimodalSplicer(ABC):
    @abstractmethod
    def splice(self, batch: MultimodalInputBatch) -> SplicedMultimodalOutput:
        pass


class TokenSplicer(IMultimodalSplicer):
    def splice(self, batch: MultimodalInputBatch) -> SplicedMultimodalOutput:
        batch_size = batch.input_ids.shape[0]
        num_patches = batch.visual_tokens.shape[1]
        spliced_embeds = []
        spliced_labels = []

        for b in range(batch_size):
            ids = batch.input_ids[b]
            embeds = batch.text_embeddings[b]
            img_indices = torch.where(ids == batch.image_token_id)[0]

            if len(img_indices) == 0:
                spliced_embeds.append(embeds)
                if batch.labels is not None:
                    spliced_labels.append(batch.labels[b])
                continue

            idx = img_indices[0].item()

            fused_emb = torch.cat(
                [embeds[:idx], batch.visual_tokens[b], embeds[idx + 1:]], 
                dim=0
            )
            spliced_embeds.append(fused_emb)

            if batch.labels is not None:
                lbls = batch.labels[b]
                mask = torch.full((num_patches,), -100, dtype=torch.long, device=ids.device)
                fused_lbl = torch.cat([lbls[:idx], mask, lbls[idx + 1:]], dim=0)
                spliced_labels.append(fused_lbl)

        stacked_embeds = torch.stack(spliced_embeds, dim=0)
        stacked_labels = torch.stack(spliced_labels, dim=0) if batch.labels is not None else None

        return SplicedMultimodalOutput(embeddings=stacked_embeds, labels=stacked_labels)

    
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length=Config.MAX_LENGTH, stride=Config.STRIDE):
        self.input_ids = []
        self.target_ids = []
        self.tokenizer = tokenizer
        self.txt = txt
        self.max_length = max_length
        self.stride = stride

        self._build_dataset()

    def _build_dataset(self):
        encoded = self.tokenizer.encode(self.txt)
        token_ids = encoded.ids if hasattr(encoded, "ids") else encoded

        self._create_chunks(token_ids, self.max_length, self.stride)

    def _create_chunks(self, tokenized_input, max_length, stride):
        for i in range(0, len(tokenized_input) - max_length, stride):
            input_chunk = tokenized_input[i:i + max_length]
            target_chunk = tokenized_input[i + 1:i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    tokenizer,
    batch_size=Config.BATCH_SIZE,
    max_length=Config.MAX_LENGTH,
    stride=Config.STRIDE,
    shuffle=Config.SHUFFLE,
    drop_last=Config.DROP_LAST,
    num_workers=Config.NUM_WORKERS,
):
    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def get_data_loader(
    corpus_path=Config.CORPUS_PATH,
    tokenizer=None,
    batch_size=Config.BATCH_SIZE,
    max_length=Config.MAX_LENGTH,
    stride=Config.STRIDE,
    shuffle=Config.SHUFFLE,
    drop_last=Config.DROP_LAST,
):
    if tokenizer is None:
        tokenizer = TokenizerNandi()
        if os.path.exists(tokenizer.model_path):
            tokenizer.load()
        else:
            tokenizer.train(corpus_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return create_dataloader(
        txt=raw_text,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    tokenizer = TokenizerNandi()
    if os.path.exists(tokenizer.model_path):
        tokenizer.load()
    else:
        tokenizer.train(Config.CORPUS_PATH)

    if os.path.exists(Config.CORPUS_PATH):
        with open(Config.CORPUS_PATH, "r", encoding="utf-8") as f:
            sample_txt = f.read()

        dataloader = create_dataloader(
            txt=sample_txt,
            tokenizer=tokenizer,
            batch_size=Config.BATCH_SIZE,
            max_length=Config.MAX_LENGTH,
            stride=Config.STRIDE,
        )
        for x, y in dataloader:
            break
    else:
        print("Path not there!!")
import os
import torch
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from typing import Optional

try:
    from src.interfaces import IMultimodalSplicer, ITokenizer, MultimodalInputBatch, SplicedMultimodalOutput
    from src.tokenization import TokenizerNandi, TokenizerConfig
    from src.config import DEFAULT_PAD_TOKEN_ID
except ImportError:
    from interfaces import IMultimodalSplicer, ITokenizer, MultimodalInputBatch, SplicedMultimodalOutput
    from tokenization import TokenizerNandi, TokenizerConfig
    from config import DEFAULT_PAD_TOKEN_ID


class DataPipelineConfig:
    """Runtime configuration for the text data pipeline."""
    CORPUS_PATH = os.getenv("CORPUS_PATH", TokenizerConfig.CORPUS_PATH)
    MAX_LENGTH = 256
    STRIDE = 128
    BATCH_SIZE = 4
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


# =============================================================================
# Multimodal token splicer  (implements IMultimodalSplicer)
# =============================================================================

class TokenSplicer(IMultimodalSplicer):
    def splice(self, batch: MultimodalInputBatch) -> SplicedMultimodalOutput:
        batchSize = batch.input_ids.shape[0]
        numPatches = batch.visual_tokens.shape[1]
        splicedEmbeds = []
        splicedLabels = []

        for b in range(batchSize):
            ids = batch.input_ids[b]
            embeds = batch.text_embeddings[b]
            imgIndices = torch.where(ids == batch.image_token_id)[0]

            if len(imgIndices) == 0:
                splicedEmbeds.append(embeds)
                if batch.labels is not None:
                    splicedLabels.append(batch.labels[b])
                continue

            idx = imgIndices[0].item()

            fusedEmb = torch.cat(
                [embeds[:idx], batch.visual_tokens[b], embeds[idx + 1:]],
                dim=0
            )
            splicedEmbeds.append(fusedEmb)

            if batch.labels is not None:
                lbls = batch.labels[b]
                mask = torch.full((numPatches,), -100, dtype=torch.long, device=ids.device)
                fusedLbl = torch.cat([lbls[:idx], mask, lbls[idx + 1:]], dim=0)
                splicedLabels.append(fusedLbl)

        stackedEmbeds = torch.stack(splicedEmbeds, dim=0)
        stackedLabels = torch.stack(splicedLabels, dim=0) if batch.labels is not None else None

        return SplicedMultimodalOutput(embeddings=stackedEmbeds, labels=stackedLabels)


# =============================================================================
# Text dataset and dataloader
# =============================================================================

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer: ITokenizer, maxLength=DataPipelineConfig.MAX_LENGTH, stride=DataPipelineConfig.STRIDE):
        self.input_ids = []
        self.target_ids = []
        self.tokenizer = tokenizer
        self.txt = txt
        self.maxLength = maxLength
        self.stride = stride

        self._buildDataset()

    def _buildDataset(self):
        encoded = self.tokenizer.encode(self.txt)
        tokenIds = encoded.ids if hasattr(encoded, "ids") else encoded
        self._createChunks(tokenIds, self.maxLength, self.stride)

    def _createChunks(self, tokenizedInput, maxLength, stride):
        for i in range(0, len(tokenizedInput) - maxLength, stride):
            inputChunk = tokenizedInput[i:i + maxLength]
            targetChunk = tokenizedInput[i + 1:i + maxLength + 1]
            self.input_ids.append(torch.tensor(inputChunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(targetChunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def get_data_loader(
    corpus_path=DataPipelineConfig.CORPUS_PATH,
    tokenizer: ITokenizer = None,
    batch_size=DataPipelineConfig.BATCH_SIZE,
    max_length=DataPipelineConfig.MAX_LENGTH,
    stride=DataPipelineConfig.STRIDE,
    shuffle=DataPipelineConfig.SHUFFLE,
    drop_last=DataPipelineConfig.DROP_LAST,
):
    if tokenizer is None:
        tokenizer = TokenizerNandi()
        if os.path.exists(tokenizer.model_path):
            tokenizer.load()
        else:
            tokenizer.train(corpus_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        rawText = f.read()

    dataset = GPTDataset(rawText, tokenizer, maxLength=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=DataPipelineConfig.NUM_WORKERS,
    )


if __name__ == "__main__":
    tokenizer = TokenizerNandi()
    if os.path.exists(tokenizer.model_path):
        tokenizer.load()
    else:
        tokenizer.train(DataPipelineConfig.CORPUS_PATH)

    if os.path.exists(DataPipelineConfig.CORPUS_PATH):
        dataloader = get_data_loader(tokenizer=tokenizer)
        for x, y in dataloader:
            break
    else:
        print("Corpus path not found!")
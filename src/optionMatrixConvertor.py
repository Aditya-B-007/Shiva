from typing import List, Union
import numpy as np
from src.dtos import CandidateOptionDTO, OptionMatrixRequestDTO
from src.tokenizer import BPETokenizer, EmbeddingTable
from src.transformer import BidirectionalEncoderStack, mean_pooling

__all__ = ["CandidateOptionDTO", "OptionMatrixRequestDTO", "OptionMatrixConverter"]


class OptionMatrixConverter:
    def __init__(
        self,
        tokenizer: BPETokenizer,
        embedder: EmbeddingTable,
        encoder: BidirectionalEncoderStack,
    ):
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.encoder = encoder

    def convert(self, options: Union[List[str], List[CandidateOptionDTO], OptionMatrixRequestDTO]) -> np.ndarray:
        if isinstance(options, OptionMatrixRequestDTO):
            texts = [opt.text for opt in options.options]
        elif isinstance(options, list):
            texts = [opt.text if isinstance(opt, CandidateOptionDTO) else str(opt) for opt in options]
        else:
            raise TypeError("Unsupported options format for OptionMatrixConverter")

        if not texts:
            return np.empty((0, self.embedder.hidden_dim), dtype=np.float32)

        input_ids, attention_mask = self.tokenizer.encode_batch(texts)
        embeddings = self.embedder.forward(input_ids)
        encoder_out = self.encoder.forward(embeddings, attention_mask=attention_mask)
        pooled = mean_pooling(encoder_out, attention_mask=attention_mask)
        return pooled
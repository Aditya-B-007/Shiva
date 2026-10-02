import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    from src.config.config import (
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        DEFAULT_VOCAB_SIZE,
        MIN_MERGE_FREQUENCY,
    )
    from src.transformerAndRL.embeddingTable import EmbeddingTable
except ImportError:
    from Shiva.src.config.config import (
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        DEFAULT_VOCAB_SIZE,
        MIN_MERGE_FREQUENCY,
    )
    from Shiva.src.transformerAndRL.embeddingTable import EmbeddingTable

__all__ = ["BPETokenizer", "EmbeddingTable"]


# =====================================================================
# BYTE-PAIR ENCODING (BPE) TOKENIZER
# =====================================================================

class BPETokenizer:

    def __init__(self):
        self._pad_token: str = PAD_TOKEN
        self._unk_token: str = UNK_TOKEN
        self._bos_token: str = BOS_TOKEN
        self._eos_token: str = EOS_TOKEN

        self._special_tokens: List[str] = [
            self._pad_token,
            self._unk_token,
            self._bos_token,
            self._eos_token,
        ]
        self._token_to_id: Dict[bytes, int] = {}
        self._id_to_token: Dict[int, bytes] = {}
        self._merges: Dict[Tuple[bytes, bytes], bytes] = {}

        self._initialize_base_vocab()

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self._token_to_id)

    @property
    def pad_token_id(self) -> int:
        """Integer ID for the padding token."""
        return self._token_to_id[self._pad_token.encode("utf-8")]

    @property
    def unk_token_id(self) -> int:
        """Integer ID for the unknown token."""
        return self._token_to_id[self._unk_token.encode("utf-8")]

    @property
    def bos_token_id(self) -> int:
        """Integer ID for the beginning-of-sequence token."""
        return self._token_to_id[self._bos_token.encode("utf-8")]

    @property
    def eos_token_id(self) -> int:
        """Integer ID for the end-of-sequence token."""
        return self._token_to_id[self._eos_token.encode("utf-8")]

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_tokens)

    # -----------------------------------------------------------------
    # Private Vocabulary Initialization & BPE Operations
    # -----------------------------------------------------------------

    def _initialize_base_vocab(self) -> None:
        self._token_to_id.clear()
        self._id_to_token.clear()

        for idx, token_str in enumerate(self._special_tokens):
            b_token = token_str.encode("utf-8")
            self._token_to_id[b_token] = idx
            self._id_to_token[idx] = b_token

        offset = len(self._special_tokens)
        for b in range(256):
            b_token = bytes([b])
            self._token_to_id[b_token] = offset + b
            self._id_to_token[offset + b] = b_token

    @staticmethod
    def _get_stats(token_sequences: List[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
        pairs: Dict[Tuple[bytes, bytes], int] = {}
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    @staticmethod
    def _merge_sequence(
        seq: List[bytes],
        pair: Tuple[bytes, bytes],
        replacement: bytes
    ) -> List[bytes]:
        new_seq: List[bytes] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(replacement)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    # -----------------------------------------------------------------
    # Training & Tokenization
    # -----------------------------------------------------------------

    def train(self, corpus: List[str], target_vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        token_sequences: List[List[bytes]] = [
            [bytes([b]) for b in text.encode("utf-8")]
            for text in corpus
        ]

        num_merges = target_vocab_size - self.vocab_size
        for _ in range(num_merges):
            stats = self._get_stats(token_sequences)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < MIN_MERGE_FREQUENCY:
                break

            merged_token = best_pair[0] + best_pair[1]
            new_id = self.vocab_size

            self._merges[best_pair] = merged_token
            self._token_to_id[merged_token] = new_id
            self._id_to_token[new_id] = merged_token

            token_sequences = [
                self._merge_sequence(seq, best_pair, merged_token)
                for seq in token_sequences
            ]

    def train_from_file(self, file_path: str, target_vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
        self.train(corpus, target_vocab_size=target_vocab_size)

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        tokens: List[bytes] = [bytes([b]) for b in text.encode("utf-8")]

        for pair, replacement in self._merges.items():
            tokens = self._merge_sequence(tokens, pair, replacement)

        unk_id = self.unk_token_id
        return [self._token_to_id.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        raw_bytes = bytearray()
        special_bytes = {s.encode("utf-8") for s in self._special_tokens}
        for tid in token_ids:
            if tid in self._id_to_token:
                b_val = self._id_to_token[tid]
                if b_val in special_bytes:
                    continue
                raw_bytes.extend(b_val)
        return raw_bytes.decode("utf-8", errors="replace")

    def encode_batch(
        self,
        texts: List[str],
        max_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        tokenized_batch = [self.encode(text) for text in texts]
        batch_size = len(texts)
        if batch_size == 0:
            return np.empty((0, 0), dtype=np.int64), np.empty((0, 0), dtype=np.float32)

        actual_max_len = max((len(seq) for seq in tokenized_batch), default=0)
        seq_len = max_len if max_len is not None else actual_max_len
        seq_len = max(seq_len, 1)

        pad_id = self.pad_token_id
        input_ids = np.full((batch_size, seq_len), fill_value=pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i, seq in enumerate(tokenized_batch):
            trunc_seq = seq[:seq_len]
            length = len(trunc_seq)
            if length > 0:
                input_ids[i, :length] = trunc_seq
                attention_mask[i, :length] = 1.0

        return input_ids, attention_mask

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, filepath: str) -> None:
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        serialized_merges = [
            [[p[0].hex(), p[1].hex()], m.hex()]
            for p, m in self._merges.items()
        ]
        serialized_vocab = {
            tok.hex(): idx for tok, idx in self._token_to_id.items()
        }
        data = {
            "special_tokens": self._special_tokens,
            "vocab_size": self.vocab_size,
            "merges": serialized_merges,
            "token_to_id": serialized_vocab
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._special_tokens = data["special_tokens"]
        self._pad_token = self._special_tokens[0]
        self._unk_token = self._special_tokens[1]
        self._bos_token = self._special_tokens[2]
        self._eos_token = self._special_tokens[3]

        self._merges = {}
        for (p0_hex, p1_hex), m_hex in data["merges"]:
            pair = (bytes.fromhex(p0_hex), bytes.fromhex(p1_hex))
            self._merges[pair] = bytes.fromhex(m_hex)

        self._token_to_id = {
            bytes.fromhex(k_hex): idx for k_hex, idx in data["token_to_id"].items()
        }
        self._id_to_token = {
            idx: tok for tok, idx in self._token_to_id.items()
        }

    def load_from_json(self, filepath: str) -> None:
        self.load(filepath)

    @classmethod
    def from_json(cls, filepath: str) -> "BPETokenizer":
        tokenizer = cls()
        tokenizer.load(filepath)
        return tokenizer
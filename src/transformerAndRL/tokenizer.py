import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    from src.config.config import (
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        PAD_TOKEN_ID,
        HIDDEN_DIM,
        DEFAULT_VOCAB_SIZE,
        MIN_MERGE_FREQUENCY,
        TokenizerConfig,
    )
except ImportError:
    from Shiva.src.config.config import (
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        PAD_TOKEN_ID,
        HIDDEN_DIM,
        DEFAULT_VOCAB_SIZE,
        MIN_MERGE_FREQUENCY,
        TokenizerConfig,
    )


# =====================================================================
# 1. BYTE-PAIR ENCODING (BPE) TOKENIZER
# =====================================================================

class BPETokenizer:
    def __init__(self):
        # Reserved Special Tokens
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}
        
        self.merges: Dict[Tuple[bytes, bytes], bytes] = {}
        
        self._initialize_base_vocab()

    def _initialize_base_vocab(self):
        for idx, token_str in enumerate(self.special_tokens):
            b_token = token_str.encode("utf-8")
            self.token_to_id[b_token] = idx
            self.id_to_token[idx] = b_token
        offset = len(self.special_tokens)
        for b in range(256):
            b_token = bytes([b])
            self.token_to_id[b_token] = offset + b
            self.id_to_token[offset + b] = b_token

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.pad_token.encode("utf-8")]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _get_stats(self, token_sequences: List[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
        pairs: Dict[Tuple[bytes, bytes], int] = {}
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_sequence(
        self, 
        seq: List[bytes], 
        pair: Tuple[bytes, bytes], 
        replacement: bytes
    ) -> List[bytes]:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(replacement)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        return new_seq

    def train(self, corpus: List[str], target_vocab_size: int = DEFAULT_VOCAB_SIZE):
        token_sequences: List[List[bytes]] = [
            [bytes([b]) for b in text.encode("utf-8")]
            for text in corpus
        ]

        num_merges = target_vocab_size - self.vocab_size
        for step in range(num_merges):
            stats = self._get_stats(token_sequences)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < MIN_MERGE_FREQUENCY:
                # No common pairs remaining
                break

            merged_token = best_pair[0] + best_pair[1]
            new_id = self.vocab_size
            
            self.merges[best_pair] = merged_token
            self.token_to_id[merged_token] = new_id
            self.id_to_token[new_id] = merged_token

            token_sequences = [
                self._merge_sequence(seq, best_pair, merged_token)
                for seq in token_sequences
            ]

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        tokens: List[bytes] = [bytes([b]) for b in text.encode("utf-8")]

        for pair, replacement in self.merges.items():
            tokens = self._merge_sequence(tokens, pair, replacement)

        unk_id = self.token_to_id[self.unk_token.encode("utf-8")]
        return [self.token_to_id.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        raw_bytes = bytearray()
        for tid in token_ids:
            if tid in self.id_to_token:
                b_val = self.id_to_token[tid]
                if b_val in [s.encode("utf-8") for s in self.special_tokens]:
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

        actual_max_len = max(len(seq) for seq in tokenized_batch) if tokenized_batch else 0
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


# =====================================================================
# 2. EMBEDDING TABLE
# =====================================================================

class EmbeddingTable:
    def __init__(self, vocab_size: int, hidden_dim: int = HIDDEN_DIM, pad_token_id: int = PAD_TOKEN_ID):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        scale = 1.0 / np.sqrt(hidden_dim)
        self.weights = np.random.normal(loc=0.0, scale=scale, size=(vocab_size, hidden_dim)).astype(np.float32)
        if pad_token_id is not None:
            self.weights[pad_token_id] = 0.0

        self.grad_weights = np.zeros_like(self.weights)
        self.last_input_ids = None

    @classmethod
    def from_tokenizer(cls, tokenizer: BPETokenizer, hidden_dim: int = HIDDEN_DIM) -> "EmbeddingTable":
        return cls(vocab_size=tokenizer.vocab_size, hidden_dim=hidden_dim, pad_token_id=tokenizer.pad_token_id)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        self.last_input_ids = input_ids
        return self.weights[input_ids]

    def backward(self, grad_output: np.ndarray) -> None:
        self.grad_weights.fill(0.0)
        np.add.at(self.grad_weights, self.last_input_ids, grad_output)
        
        if self.pad_token_id is not None:
            self.grad_weights[self.pad_token_id] = 0.0
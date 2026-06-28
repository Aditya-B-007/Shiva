import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from dotenv import load_dotenv
from base.PositionalEncodingInterface import PositionalEncodingInterface

load_dotenv()

###########################################################################
# CONFIGURATION
###########################################################################

@dataclass
class TransformerConfig:
    vocab_size: int = int(os.getenv("VOCAB_SIZE", 32000))
    max_sequence_length: int = int(os.getenv("MAX_SEQUENCE_LENGTH", 2048))
    vector_size: int = int(os.getenv("VECTOR_SIZE", 2048))
    num_heads: int = int(os.getenv("NUM_HEADS", 8))
    num_layers: int = int(os.getenv("NUM_LAYERS", 18))
    feed_forward_dimension: int = int(os.getenv("FFN_DIM", 8192))
    dropout: float = float(os.getenv("DROPOUT", 0.1))
    device: str = os.getenv("DEVICE", "cpu")
    dtype: str = os.getenv("DTYPE", "float16")
    positional_encoding: str = os.getenv("POSITIONAL_ENCODING", "rope") 
    rope_theta: float = float(os.getenv("ROPE_THETA", 10000.0))

    @classmethod
    def from_env(cls):
        return cls()


###########################################################################
# POSITIONAL ENCODING PROVIDER
###########################################################################
class PositionalEncodingProvider:
    @staticmethod
    def create(config) -> PositionalEncodingInterface:
        """Returns the specific strategy strongly typed to the interface definition."""
        pe_type = config.positional_encoding.lower()
        
        if pe_type == "sinusoidal":
            from .SinusoidalPositionalEncoding import SinusoidalPositionalEncoding
            return SinusoidalPositionalEncoding(config)
        elif pe_type == "rope":
            from .RoPEPositionalEncoding import RotaryPositionalEncoding
            return RotaryPositionalEncoding(config)
        elif pe_type == "alibi":
            from .AliBiPositionalEncoding import ALiBiPositionalEncoding
            return ALiBiPositionalEncoding(config)
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}")


###########################################################################
# EMBEDDING / UTILS
###########################################################################
class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.vector_size = config.vector_size
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.vector_size,
            padding_idx=0
        )
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.long:
            raise TypeError("Input tokens must be of dtype torch.long.")
        if tokens.dim() != 2:
            raise ValueError("Input tensor must have shape (batch_size, sequence_length).")
        return self.embedding(tokens)

class Dropout(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.vector_size)
    def forward(self, x):
        return self.layer_norm(x)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.vector_size, config.feed_forward_dimension)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.feed_forward_dimension, config.vector_size)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


###########################################################################
# MULTI HEAD ATTENTION
###########################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, config, use_positional: bool = True):
        super().__init__()
        self.vector_size = config.vector_size
        self.num_heads = config.num_heads
        self.head_dimension = self.vector_size // self.num_heads
        self.position_encoding = (
            PositionalEncodingProvider.create(config) if use_positional else None
        )

        self.query_projection = nn.Linear(self.vector_size, self.vector_size)
        self.key_projection = nn.Linear(self.vector_size, self.vector_size)
        self.value_projection = nn.Linear(self.vector_size, self.vector_size)
        self.output_projection = nn.Linear(self.vector_size, self.vector_size)

    def forward(self, query, key, value, attention_mask=None):
        batch_size, seq_len, _ = query.shape
        q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dimension).transpose(1, 2)
        k = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
        v = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
        
        if self.position_encoding:
            q, k = self.position_encoding.apply_qk(q, k)

        scale = self.head_dimension ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.position_encoding:
            scores = self.position_encoding.apply_attention_scores(scores)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.vector_size)
        
        return self.output_projection(context)


###########################################################################
# STRUCTURAL BLOCKS (ENCODER / DECODER)
###########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config, use_positional=True)
        self.dropout1 = Dropout(config)
        self.norm1 = LayerNormalization(config)
        self.feed_forward = FeedForward(config)
        self.dropout2 = Dropout(config)
        self.norm2 = LayerNormalization(config)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.self_attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.causal_attention = MultiHeadAttention(config, use_positional=True)
        self.dropout1 = Dropout(config)
        self.norm1 = LayerNormalization(config)
        self.cross_attention = MultiHeadAttention(config, use_positional=False)
        self.dropout2 = Dropout(config)
        self.norm2 = LayerNormalization(config)

        self.feed_forward = FeedForward(config)
        self.dropout3 = Dropout(config)
        self.norm3 = LayerNormalization(config)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        x = self.norm1(x + self.dropout1(self.causal_attention(x, x, x, target_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attention(x, encoder_output, encoder_output, source_mask)))
        x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return x


###########################################################################
# ORCHESTRATOR MODEL
###########################################################################
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = PositionalEncodingProvider.create(config)
        
        self.dropout = Dropout(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, source, source_mask=None):
        x = self.token_embedding(source)
        x = self.position_encoding.apply_embedding(x)
        return self.encoder(self.dropout(x), source_mask)

    def decode(self, target, encoder_output, source_mask=None, target_mask=None):
        x = self.token_embedding(target)
        x = self.position_encoding.apply_embedding(x)
        return self.decoder(self.dropout(x), encoder_output, source_mask, target_mask)

    def forward(self, source, target, source_mask=None, target_mask=None):
        return self.decode(target, self.encode(source, source_mask), source_mask, target_mask)

import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from dotenv import load_dotenv
from .PositionalEncoding import PositionalEncodingInterface
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
    positional_encoding_type: str = os.getenv("POSITIONAL_ENCODING_TYPE", "rope") 
    rope_theta: float = float(os.getenv("ROPE_THETA", 10000.0))

    @classmethod
    def from_env(cls):
        return cls()


###########################################################################
# EMBEDDING
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
        #Swaps the integer id of every word with the vector, no neural network involved.
        if tokens.dtype != torch.long:
            raise TypeError(
                "Input tokens must be of dtype torch.long."
            )
        if tokens.dim() != 2:
            raise ValueError(
                "Input tensor must have shape "
                "(batch_size, sequence_length)."
            )
        embeddings = self.embedding(tokens)
        return embeddings # This might blow up


###########################################################################
# POSITIONAL ENCODING ROUTER
###########################################################################
class PositionalEncodingRouter:
    @staticmethod
    def create(config) -> nn.Module:
        pe_type = config.positional_encoding_type.lower()
        
        if pe_type == "sinusoidal":
            from brain.PositionalEncoding.SinusoidalPositionalEncoding import SinusoidalPositionalEncoding
            return SinusoidalPositionalEncoding(config)
        elif pe_type == "rope":
            from brain.PositionalEncoding.RoPEPositionalEncoding import RotaryPositionalEncoding
            return RotaryPositionalEncoding(config)
        elif pe_type == "alibi":
            from brain.PositionalEncoding.AliBiPositionalEncoding import ALiBiPositionalEncoding
            return ALiBiPositionalEncoding(config)
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}")


###########################################################################
# DROPOUT
###########################################################################

class Dropout(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        return self.dropout(x)


###########################################################################
# LAYER NORMALIZATION
###########################################################################

class LayerNormalization(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.layer_norm = nn.LayerNorm(config.vector_size)

    def forward(self, x):

        return self.layer_norm(x)


###########################################################################
# FEED FORWARD NETWORK
###########################################################################

class FeedForward(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.linear1 = nn.Linear(config.vector_size,config.feed_forward_dimension)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.feed_forward_dimension,config.vector_size)

    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


###########################################################################
# MULTI HEAD ATTENTION
###########################################################################

class MultiHeadAttention(nn.Module):

    def __init__(self, config, position_encoding: nn.Module = None):
        super().__init__()
        self.vector_size = config.vector_size
        self.num_heads = config.num_heads
        self.head_dimension = self.vector_size // self.num_heads
        self.position_encoding = position_encoding

        self.query_projection = nn.Linear(self.vector_size, self.vector_size)
        self.key_projection = nn.Linear(self.vector_size, self.vector_size)
        self.value_projection = nn.Linear(self.vector_size, self.vector_size)
        self.output_projection = nn.Linear(self.vector_size, self.vector_size)

    def forward(self, query, key, value, attention_mask=None):
        batch_size, seq_len, _ = query.shape
        q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dimension).transpose(1, 2)
        k = self.key_projection(key).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
        v = self.value_projection(value).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)

        if self.position_encoding and self.position_encoding.__class__.__name__ == "RotaryPositionalEncoding":
            q = self.position_encoding(q)
            k = self.position_encoding(k)

        scale = self.head_dimension ** 0.5
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.position_encoding and self.position_encoding.__class__.__name__ == "ALiBiPositionalEncoding":
            attention_scores = self.position_encoding(attention_scores)
            
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float("-inf"))
            
        attention_probs = torch.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, v) 
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.vector_size)
        
        return self.output_projection(context)


###########################################################################
# TRANSFORMER BLOCK
###########################################################################

class TransformerBlock(nn.Module):

    def __init__(self, config, position_encoding: nn.Module = None):
        super().__init__()
        self.self_attention = MultiHeadAttention(config, position_encoding)
        self.dropout1 = Dropout(config)
        self.norm1 = LayerNormalization(config)
        self.feed_forward = FeedForward(config)
        self.dropout2 = Dropout(config)
        self.norm2 = LayerNormalization(config)

    def forward(self, x, mask=None):
        attention = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attention)
        x = self.norm1(x)
        ff = self.feed_forward(x)
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x

###########################################################################
# ENCODER
###########################################################################

class Encoder(nn.Module):

    def __init__(self, config, position_encoding: nn.Module = None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config, position_encoding)
            for _ in range(config.num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

###########################################################################
# DECODER
###########################################################################
class DecoderBlock(nn.Module):

    def __init__(self, config, position_encoding: nn.Module = None):
        super().__init__()
        # Causal attention handles the generation stream
        self.causal_attention = MultiHeadAttention(config, position_encoding)
        self.dropout1 = Dropout(config)
        self.norm1 = LayerNormalization(config)

        # Cross attention interacts with encoder representations
        self.cross_attention = MultiHeadAttention(config, position_encoding=None) # Positional embeddings are handled by encoder/causal steps
        self.dropout2 = Dropout(config)
        self.norm2 = LayerNormalization(config)

        self.feed_forward = FeedForward(config)
        self.dropout3 = Dropout(config)
        self.norm3 = LayerNormalization(config)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        # Phase 1: Masked Causal Self-Attention
        attn_target = self.causal_attention(x, x, x, target_mask)
        x = x + self.dropout1(attn_target)
        x = self.norm1(x)

        # Phase 2: Cross-Attention (Query from Decoder, Key & Value from Encoder)
        attn_cross = self.cross_attention(query=x, key=encoder_output, value=encoder_output, attention_mask=source_mask)
        x = x + self.dropout2(attn_cross)
        x = self.norm2(x)

        # Phase 3: Feed-Forward Network
        ff_out = self.feed_forward(x)
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)
        
        return x


class Decoder(nn.Module):

    def __init__(self, config, position_encoding: nn.Module = None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(config, position_encoding)
            for _ in range(config.num_layers)
        ])

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return x

###########################################################################
# TRANSFORMER
###########################################################################

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = PositionalEncodingRouter.create(config)
        self.dropout = Dropout(config)
        self.encoder = Encoder(config, self.position_encoding)
        self.decoder = Decoder(config, self.position_encoding)

    def encode(self, source, source_mask=None):
        x = self.token_embedding(source)
        if self.position_encoding.__class__.__name__ == "SinusoidalPositionalEncoding":
            x = self.position_encoding(x)
        x = self.dropout(x)
        return self.encoder(x, source_mask)

    def decode(self, target, encoder_output, source_mask=None, target_mask=None):
        x = self.token_embedding(target)
        if self.position_encoding.__class__.__name__ == "SinusoidalPositionalEncoding":
            x = self.position_encoding(x)

        x = self.dropout(x)
        return self.decoder(x, encoder_output, source_mask, target_mask)

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoder_output = self.encode(source, source_mask)
        decoder_output = self.decode(target, encoder_output, source_mask, target_mask)
        return decoder_output

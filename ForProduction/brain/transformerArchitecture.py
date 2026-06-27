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

    vocab_size= os.getenv("VOCAB_SIZE",32000)
    max_sequence_length= os.getenv("MAX_SEQUENCE_LENGTH",2048)
    vector_size=os.getenv("VECTOR_SIZE",2048)
    num_heads=os.getenv("NUM_HEADS",8)
    num_layers=os.getenv("NUM_LAYERS",18)
    feed_forward_dimension=os.getenv("FFN_DIM",8192)
    dropout=os.getenv("DROPOUT",0.1)
    device=os.getenv("DEVICE", "cpu")
    dtype=os.getenv("DTYPE", "float16")


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
# POSITIONAL ENCODING
###########################################################################



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

        self.linear1 = nn.Linear(

            config.vector_size,

            config.feed_forward_dimension

        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(

            config.feed_forward_dimension,

            config.vector_size

        )

    def forward(self, x):

        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        return x


###########################################################################
# MULTI HEAD ATTENTION
###########################################################################

class MultiHeadAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.vector_size = config.vector_size

        self.num_heads = config.num_heads

        self.head_dimension = (

            self.vector_size //

            self.num_heads

        )

        self.query_projection = nn.Linear(

            self.vector_size,

            self.vector_size

        )

        self.key_projection = nn.Linear(

            self.vector_size,

            self.vector_size

        )

        self.value_projection = nn.Linear(

            self.vector_size,

            self.vector_size

        )

        self.output_projection = nn.Linear(

            self.vector_size,

            self.vector_size

        )

    def forward(

            self,

            query,

            key,

            value,

            attention_mask=None

    ):

        #
        # TODO
        #
        # Q projection
        # K projection
        # V projection
        #
        # Split heads
        #
        # Scaled Dot Product
        #
        # Softmax
        #
        # Weighted Sum
        #
        # Merge Heads
        #
        # Output Projection
        #

        return query


###########################################################################
# TRANSFORMER BLOCK
###########################################################################

class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.self_attention = MultiHeadAttention(config)

        self.dropout1 = Dropout(config)

        self.norm1 = LayerNormalization(config)

        self.feed_forward = FeedForward(config)

        self.dropout2 = Dropout(config)

        self.norm2 = LayerNormalization(config)

    def forward(self, x, mask=None):

        attention = self.self_attention(

            x,

            x,

            x,

            mask

        )

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

    def __init__(self, config):

        super().__init__()

        self.layers = nn.ModuleList([

            TransformerBlock(config)

            for _ in range(config.num_layers)

        ])

    def forward(self, x, mask=None):

        for layer in self.layers:

            x = layer(x, mask)

        return x


###########################################################################
# DECODER
###########################################################################

class Decoder(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.layers = nn.ModuleList([

            TransformerBlock(config)

            for _ in range(config.num_layers)

        ])

    def forward(

            self,

            x,

            encoder_output,

            source_mask=None,

            target_mask=None

    ):

        #
        # TODO
        #
        # Masked Self Attention
        #
        # Cross Attention
        #
        # Feed Forward
        #

        return x


###########################################################################
# TRANSFORMER
###########################################################################

class Transformer(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.token_embedding = TokenEmbedding(config)

        self.position_encoding = PositionalEncoding(config)

        self.dropout = Dropout(config)

        self.encoder = Encoder(config)

        self.decoder = Decoder(config)

    def encode(

            self,

            source,

            source_mask=None

    ):

        x = self.token_embedding(source)

        x = self.position_encoding(x)

        x = self.dropout(x)

        return self.encoder(

            x,

            source_mask

        )

    def decode(

            self,

            target,

            encoder_output,

            source_mask=None,

            target_mask=None

    ):

        x = self.token_embedding(target)

        x = self.position_encoding(x)

        x = self.dropout(x)

        return self.decoder(

            x,

            encoder_output,

            source_mask,

            target_mask

        )

    def forward(

            self,

            source,

            target,

            source_mask=None,

            target_mask=None

    ):

        encoder_output = self.encode(

            source,

            source_mask

        )

        decoder_output = self.decode(

            target,

            encoder_output,

            source_mask,

            target_mask

        )

        return decoder_output


###########################################################################
# ENTRY POINT
###########################################################################

if __name__ == "__main__":

    config = TransformerConfig.from_env()

    model = Transformer(config)

    print(model)

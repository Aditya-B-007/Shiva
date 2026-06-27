import torch
from  PositionalEncodingInterface import PositionalEncoding
class RotaryPositionalEncoding(PositionalEncoding):

    def __init__(self, config):

        super().__init__()

        self.max_sequence_length = config.max_sequence_length
        self.vector_size = config.vector_size
        self.num_heads = config.num_heads
        self.head_dimension = self.vector_size // self.num_heads

        if self.head_dimension % 2 != 0:
            raise ValueError("Head dimension must be even.")

        self.theta = getattr(config,"rope_theta", 10000.0)

        self._build_cache(
            self.max_sequence_length
        )

    def _build_cache(self, sequence_length):

        device = (
            self.cos_cache.device
            if hasattr(self, "cos_cache")
            else torch.device("cpu")
        )

        inv_frequency = 1.0 / (

            self.theta ** (torch.arange(0,self.head_dimension,2,device=device)/ self.head_dimension))

        positions = torch.arange(sequence_length,device=device)

        frequencies = torch.outer(positions.float(),inv_frequency.float())

        frequencies = torch.cat((frequencies,frequencies),dim=-1)

        self.register_buffer(
            "cos_cache",
            torch.cos(frequencies)
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False
        )

        self.register_buffer(
            "sin_cache",
            torch.sin(frequencies)
            .unsqueeze(0)
            .unsqueeze(0),
            persistent=False
        )

    @staticmethod
    def rotate_half(x):

        half = x.shape[-1] // 2

        x1 = x[..., :half]

        x2 = x[..., half:]

        return torch.cat(
            (-x2, x1),
            dim=-1
        )

    def forward(self, x):

        seq_len = x.size(-2)

        if seq_len > self.cos_cache.size(2):
            self._build_cache(seq_len)

        cos = self.cos_cache[:, :, :seq_len]

        sin = self.sin_cache[:, :, :seq_len]

        return (
            x * cos
            +
            self.rotate_half(x) * sin
        )

import torch
class SinusoidalPositionalEncoding(PositionalEncoding):

    def __init__(self, config):

        super().__init__()

        self.vector_size = config.vector_size

        self.max_sequence_length = (
            config.max_sequence_length
        )

        positions = torch.arange(
            self.max_sequence_length
        ).unsqueeze(1)

        div_term = torch.exp(

            torch.arange( 0,self.vector_size,2)*(-torch.log(torch.tensor(10000.0))/ self.vector_size))

        encoding = torch.zeros(self.max_sequence_length,self.vector_size)

        encoding[:, 0::2] = torch.sin(positions * div_term)

        encoding[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer("encoding",encoding.unsqueeze(0),persistent=False)

    def forward(self, x):

        seq_len = x.size(1)

        return (x+self.encoding[:, :seq_len])

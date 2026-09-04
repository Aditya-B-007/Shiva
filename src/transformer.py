import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, ntoken=50257, ninp=1024, nhead=16, nhid=4096, nlayers=20, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ninp, 
            nhead=nhead, 
            dim_feedforward=nhid, 
            dropout=dropout,
            activation='relu',
            norm_first=True,
            batch_first=False  
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(ninp, ntoken)
        self.src_mask = None
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

if __name__ == "__main__":
    model = TransformerModel(ntoken=50257, ninp=1024, nhead=16, nhid=4096, nlayers=20)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Target Parameter Fit: {total_params / 1_000_000:.2f} Million Parameters")

    dummy_input = torch.randint(0, 50257, (32, 4)) 
    dummy_output = model(dummy_input)
    print(f"Output shape successfully verified: {dummy_output.shape}")

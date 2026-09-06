import torch
import torch.nn as nn
import math

#To make the following additions:
#1. Use orthogonal initialization for the weights of the encoder and decoder layers.
#2. Use Pre Layer Normalization in the TransformerEncoderLayer. With final layer normalization.
#3. We can use weight tying. It reduces the number of parameters and can improve performance. We can tie the weights of the encoder and decoder layers.
#We can use the remaining parameters for increasing the number of layers or the hidden size of the model. This can improve the model's capacity and performance.
#4. Use GELU activation function instead of ReLU for better performance in deep networks.

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
            activation='gelu',
            norm_first=True,
            batch_first=False  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=nlayers,
            norm=nn.LayerNorm(ninp)
        )
        self.decoder = nn.Linear(ninp, ntoken, bias=False)
        self.decoder.weight = self.encoder.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.orthogonal_(p)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src):
        sz = src.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(sz, device=src.device)

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, mask=mask, is_causal=True)
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
    unique_params = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)
    trainable_params = sum(p.numel() for p in unique_params if p.requires_grad)
    
    print(f"Total Parameters (unique):     {total_params:,}")
    print(f"Trainable Parameters (unique): {trainable_params:,}")
    print(f"Target Parameter Fit:          {total_params / 1_000_000:.2f} Million Parameters")

    dummy_input = torch.randint(0, 50257, (32, 4)) 
    dummy_output = model(dummy_input)
    print(f"Output shape successfully verified: {dummy_output.shape}")



"""
Transformer encoder counterpart to ResNetPredictorV2, operating on the same
8-channel, 200-step feature sequences. Kept small (CPU-trainable): 2 layers,
4 heads, d_model=64. A learned [SCORE] token is prepended and pooled at the
output, analogous to a CLS token.
"""
import math
import torch
import torch.nn as nn

IN_CH = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIM_FF = 128
MAX_LEN = 201  # 200 steps + 1 score token


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    def __init__(self, in_channels=IN_CH, d_model=D_MODEL, nhead=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.score_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True, norm_first=True)  # Pre-LN, like RoadFury
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        b = x.size(0)
        h = self.input_proj(x)  # [b, 200, d_model]
        tok = self.score_token.expand(b, -1, -1)
        h = torch.cat([tok, h], dim=1)  # [b, 201, d_model]
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.head(h[:, 0])  # score-token output

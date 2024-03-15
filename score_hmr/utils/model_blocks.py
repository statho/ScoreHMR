import math
import torch
from torch import nn
from .utils import exists


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, cond_emb_dim=None):
        super().__init__()

        inp_dim = input_dim
        output_dim = input_dim
        if cond_emb_dim is not None:
            inp_dim += cond_emb_dim

        self.layer_norm1 = nn.LayerNorm(inp_dim)
        self.linear_layer1 = nn.Linear(inp_dim, hidden_dim)
        self.act1 = nn.GELU()

        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.linear_layer2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.linear_layer1(self.layer_norm1(x)))
        x = self.act2(self.linear_layer2(self.layer_norm2(x)))
        return x


class ResMLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, time_emb_dim=None, cond_emb_dim=None):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, input_dim * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block = MLPBlock(input_dim, hidden_dim, cond_emb_dim)

    def forward(self, x, time_emb=None, cond_emb=None):
        # Scale and shift based on time embedding.
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim=1)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        residual = x

        # Concatenate params and image features.
        if cond_emb is not None:
            x = torch.cat((x, cond_emb), dim=1)

        x = residual + self.block(x)

        return x

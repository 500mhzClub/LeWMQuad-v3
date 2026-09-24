"""Training-supervised distance between two frozen dense visual states."""
import torch
from torch import nn

from lewm.dense_visual_motion_readout_development import pool_tokens


class DenseGoalMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1024, 32), nn.GELU(),
            nn.Flatten(1), nn.Linear(192*32, 64, bias=False))

    def embed(self, pooled):
        if pooled.ndim != 3 or pooled.shape[1:] != (192, 1024):
            raise ValueError('B,192,1024 pooled normalized visual tokens required')
        return self.embedding(pooled)

    def forward(self, first, second):
        # Shared embedding guarantees nonnegativity, symmetry and exact zero
        # for identical features. No action, body state or goal pose is input.
        return (self.embed(first)-self.embed(second)).square().mean(-1)

    def dense_cost(self, first, second):
        return self(pool_tokens(first), pool_tokens(second))

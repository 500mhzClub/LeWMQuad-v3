"""C3 motion-decoder input variants for development (Andrew, 30 Sep 2026).

The base decoder, `DenseVisualMotionReadout`, sees only the current and predicted-future pooled
features. The variants add inputs without disturbing its starting point: every new weight
starts at zero, so a variant initialised from the base behaves exactly like the base until
trained.

- `past_frames` (a): each token also gets [current - frame 0.5 s ago, current - frame 1 s ago],
  pooled from the same V-JEPA features. Its inputs stay visual.
- `history` (b): an embedding of the 1.5-s applied-command history (15 steps of forward and
  yaw, normalised as for C4) is added to the decoder's hidden layer.
"""
import copy

import torch
from torch import nn


class ReadoutVariant(nn.Module):
    def __init__(self, base, past_frames=False, history=False):
        super().__init__()
        self.past_frames, self.history = past_frames, history
        base = copy.deepcopy(base)
        self.register_buffer('target_mean', base.target_mean.clone())
        self.register_buffer('target_scale', base.target_scale.clone())
        linear = base.project[0]
        width = 4096 if past_frames else 2048
        self.project = nn.Sequential(nn.Linear(width, linear.out_features), nn.GELU())
        with torch.no_grad():
            self.project[0].weight.zero_()
            self.project[0].weight[:, :2048] = linear.weight
            self.project[0].bias.copy_(linear.bias)
        self.flatten, self.hidden, self.act, self.out = base.decode[0], base.decode[1], base.decode[2], base.decode[3]
        if history:
            self.history_embedding = nn.Sequential(nn.Linear(30, 64), nn.GELU(), nn.Linear(64, self.hidden.out_features))
            with torch.no_grad():
                self.history_embedding[2].weight.zero_()
                self.history_embedding[2].bias.zero_()

    def normalized(self, current, future, past05=None, past10=None, history=None):
        parts = [current, future-current]
        if self.past_frames:
            parts += [current-past05, current-past10]
        h = self.hidden(self.flatten(self.project(torch.cat(parts, dim=-1))))
        if self.history:
            h = h+self.history_embedding(history.reshape(len(history), -1))
        return self.out(self.act(h))

    def forward(self, current, future, past05=None, past10=None, history=None):
        return self.normalized(current, future, past05, past10, history)*self.target_scale+self.target_mean

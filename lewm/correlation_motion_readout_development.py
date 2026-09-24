"""Anchored linear motion readout from spatial feature correlations.

The fixed encoder remains frozen. A repeated current image has exactly zero
motion output; absolute scene features and commands do not enter the regression.
"""
import torch
from torch import nn
from torch.nn import functional as F


def correlation_difference(current, future):
    if current.shape != future.shape or current.ndim != 3 or current.shape[1:] != (192, 1024):
        raise ValueError('matching pooled current/future dense features required')
    current = F.normalize(current.float(), dim=-1)
    future = F.normalize(future.float(), dim=-1)
    return (current @ future.transpose(1, 2)-current @ current.transpose(1, 2)).flatten(1)


class CorrelationMotionReadout(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        coefficient = torch.as_tensor(coefficient, dtype=torch.float32)
        if coefficient.shape != (192*192, 3) or not torch.isfinite(coefficient).all():
            raise ValueError('finite spatial correlation to XY/yaw coefficient required')
        self.register_buffer('coefficient', coefficient.clone())

    def forward(self, current, future):
        return correlation_difference(current, future) @ self.coefficient

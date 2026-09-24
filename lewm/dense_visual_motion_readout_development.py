"""Common supervised motion probe; frozen visual features are its only inputs."""
import torch
from torch import nn
from torch.nn import functional as F


def pool_tokens(tokens):
    """Keep a 12x16 spatial grid by averaging adjacent 2x2 V-JEPA patches."""
    if tokens.ndim != 3 or tokens.shape[1:] != (768,1024):
        raise ValueError('B,768,1024 normalized dense tokens required')
    grid=tokens.transpose(1,2).reshape(-1,1024,24,32)
    return F.avg_pool2d(grid,2).flatten(2).transpose(1,2).contiguous()


class DenseVisualMotionReadout(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(2048,32),nn.GELU())
        self.decode=nn.Sequential(nn.Flatten(1),nn.Linear(192*32,128),nn.GELU(),nn.Linear(128,3))
        self.register_buffer('target_mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('target_scale',torch.as_tensor(scale,dtype=torch.float32).clone())
        if self.target_mean.shape!=(3,) or self.target_scale.shape!=(3,) or (self.target_scale<=0).any():
            raise ValueError('three training-only motion normalization channels required')

    def normalized(self,current,future):
        if current.shape != future.shape or current.shape[1:] != (192,1024):
            raise ValueError('matching B,192,1024 pooled current/future features required')
        return self.decode(self.project(torch.cat((current,future-current),dim=-1)))

    def forward(self,current,future):
        return self.normalized(current,future)*self.target_scale+self.target_mean

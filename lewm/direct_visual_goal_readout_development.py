"""Observed-image goal displacement without action-conditioned prediction."""
import torch
from torch import nn


class DirectVisualGoalReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(1024,32),nn.GELU())
        self.decode = nn.Sequential(nn.Flatten(1),nn.Linear(192*64,32),nn.GELU(),nn.Linear(32,3,bias=False))
        self.register_buffer('scale',torch.tensor([.03,.03,5*torch.pi/180]))

    def normalized(self,current,goal):
        if current.shape != goal.shape or current.ndim != 3 or current.shape[1:] != (192,1024):
            raise ValueError('matching B,192,1024 observed pooled visual tokens required')
        a,b = self.project(current),self.project(goal)
        # Enforce zero displacement for identical observations by construction.
        # This identity is part of fitting, not a post-hoc calibration.
        return self.decode(torch.cat((a,b-a),dim=-1))-self.decode(torch.cat((a,torch.zeros_like(a)),dim=-1))

    def forward(self,current,goal):
        return self.normalized(current,goal)*self.scale

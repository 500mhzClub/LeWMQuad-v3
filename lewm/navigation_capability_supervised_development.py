"""Preregistered direct motion regressor on the same frozen visual features."""
import torch
from torch import nn


class DirectMotionPredictor(nn.Module):
    def __init__(self, target_mean, target_scale):
        super().__init__()
        self.projection=nn.Linear(1024,32)
        self.network=nn.Sequential(nn.Linear(18432+47,896),nn.GELU(),
            nn.Linear(896,896),nn.GELU(),nn.Linear(896,3))
        self.register_buffer('target_mean',torch.as_tensor(target_mean,dtype=torch.float32).clone())
        self.register_buffer('target_scale',torch.as_tensor(target_scale,dtype=torch.float32).clone())

    def normalized(self, features, past_control, future_action, horizon):
        assert features.shape[1:]==(3,192,1024)
        assert past_control.shape[1:]==(3,5,2) and future_action.shape[1:]==(8,2)
        x=torch.nn.functional.gelu(self.projection(features)).flatten(1)
        steps=torch.arange(8,device=features.device)[None,:]
        future=future_action*(steps<horizon[:,None])[:,:,None]
        condition=torch.cat((past_control.flatten(1),future.flatten(1),horizon[:,None].float()/8),dim=1)
        return self.network(torch.cat((x,condition),dim=1))

    def forward(self, features, past_control, future_action, horizon):
        return self.normalized(features,past_control,future_action,horizon)*self.target_scale+self.target_mean

"""Native 100--800 ms conditioning while preserving the 500-ms parent initially."""
import torch
from torch import nn


class ExtendedActionInput(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not isinstance(original,nn.Linear) or original.in_features!=10:
            raise ValueError('five two-channel applied commands required in the parent')
        self.original=original
        self.extension=nn.Linear(7,original.out_features,bias=False,
                                 device=original.weight.device,dtype=original.weight.dtype)
        nn.init.zeros_(self.extension.weight)

    def forward(self, action):
        if action.ndim!=2 or action.shape[1]!=17:raise ValueError('eight commands plus target time required')
        return self.original(action[:,:10].contiguous())+self.extension(action[:,10:].contiguous())


class HorizonConditionedDensePredictor(nn.Module):
    def __init__(self, parent, *, action_blind=False):
        """Takes ownership of a freshly loaded parent; never mutate a live controller."""
        super().__init__()
        self.backbone=parent
        self.backbone.action[0]=ExtendedActionInput(self.backbone.action[0])
        self.action_blind=action_blind

    @staticmethod
    def condition(future_applied_commands,horizon_ticks,*,action_blind=False):
        if future_applied_commands.ndim!=3 or future_applied_commands.shape[1:]!=(8,2):
            raise ValueError('B,8,2 native forward/yaw commands required')
        h=torch.as_tensor(horizon_ticks,device=future_applied_commands.device)
        if h.dtype not in (torch.int32,torch.int64) or h.shape!=(len(future_applied_commands),):
            raise ValueError('one integer horizon per example required')
        if not bool(((h>=1)&(h<=8)).all()):raise ValueError('trained target domain is 100--800 ms')
        valid=torch.arange(8,device=h.device)[None]<h[:,None]
        safe=torch.where(valid[:,:,None],future_applied_commands,torch.zeros_like(future_applied_commands))
        if not torch.isfinite(safe).all():raise ValueError('finite pre-target applied commands required')
        if action_blind:safe=torch.zeros_like(safe)
        # The time input stays available in both arms. Only future action is ablated.
        time=(h.to(dtype=safe.dtype)-5)/5
        return torch.cat((safe.flatten(1),time[:,None]),dim=1)

    def forward(self,context,future_applied_commands,horizon_ticks,mask,*,control):
        condition=self.condition(future_applied_commands,horizon_ticks,action_blind=self.action_blind)
        return self.backbone(context,condition,mask,control=control)

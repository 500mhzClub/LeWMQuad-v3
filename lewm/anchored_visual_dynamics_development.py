"""Frozen visual state plus learned, action-conditioned visual innovation."""
import torch
from torch import nn


class AnchoredVisualDynamics(nn.Module):
    def __init__(self, statistics, *, use_future_actions=True):
        super().__init__()
        self.use_future_actions = use_future_actions
        for name in ('anchor_mean','anchor_scale','innovation_scale'):
            self.register_buffer(name, statistics[name].clone().float())
        self.history = nn.GRU(32,32,batch_first=True)
        self.history_norm = nn.LayerNorm(32)
        self.transition = nn.Sequential(nn.Linear(101,256),nn.SiLU(),nn.Linear(256,32))
        # Both matched arms start exactly at persistence, not an arbitrary
        # multimodal state that must first learn the visual target's coordinates.
        nn.init.zeros_(self.transition[-1].weight)
        nn.init.zeros_(self.transition[-1].bias)

    def forward(self, past, anchor, blocks, valid):
        _, hidden = self.history(past)
        context = self.history_norm(hidden[-1])
        current = (anchor-self.anchor_mean)/self.anchor_scale
        delta = torch.zeros_like(anchor); values = []
        commands = blocks[:,:,0] if self.use_future_actions else torch.zeros_like(blocks[:,:,0])
        for h in range(8):
            active = valid[:,h]
            seconds = anchor.new_full((len(anchor),1), .1*(h+1))
            token = torch.cat((delta,context,current,commands[:,h],active.to(anchor.dtype),seconds),-1)
            delta = torch.where(active,delta+self.transition(token),delta)
            value = anchor+delta*self.innovation_scale
            values.append(torch.where(active,value,torch.zeros_like(value)))
        return torch.stack(values,1)

"""Unchanged dense planner, using the existing signed goal readout as its cost."""
import torch
from torch import nn

from lewm.dense_visual_arrival_control_development import VisualArrivalControl


class SignedPoseEmbedding(nn.Module):
    def __init__(self, readout, goal):
        super().__init__()
        self.readout=readout
        self.register_buffer('goal',goal)

    def embed(self, pooled):
        return self.readout(pooled,self.goal.expand(len(pooled),-1,-1))/self.readout.scale


class SignedPoseGoalControl(VisualArrivalControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metric=SignedPoseEmbedding(self.arrival_readout,self.pooled_goal)
        self.goal_embedding=torch.zeros((1,3),device=self.pooled_goal.device)

    @torch.inference_mode()
    def observe(self,packet):
        record=super().observe(packet)
        record['current_goal_metric']*=3
        record['goal_cost_model']='squared_signed_goal_pose'
        return record

    @torch.inference_mode()
    def choose(self,packet):
        choice=super().choose(packet)
        if not choice['arrival_latched']:
            # The inherited argmin uses mean over three components. Report
            # sum in the diagnostic's units; a constant factor preserves rank.
            choice['costs']=[3*v for v in choice['costs']]
            choice['cost_kind']='squared_signed_goal_pose'
        choice['goal_cost_model']='squared_signed_goal_pose'
        return choice

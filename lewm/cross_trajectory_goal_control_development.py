"""Existing planner/arrival rule with the mixed-pair goal metric only."""
import torch

from lewm.dense_visual_arrival_control_development import VisualArrivalControl
from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import train_go2_cross_trajectory_goal_metric_development as fitted


class CrossTrajectoryGoalControl(VisualArrivalControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.metric=fitted.load().cuda()
        with torch.inference_mode():
            self.goal_embedding=self.metric.embed(pool_tokens(self.goal[None]))

    def choose(self,packet):
        return super().choose(packet)|dict(goal_cost_model='cross_trajectory_goal_metric')

"""Existing visual planner with observed-image goal recognition and terminal hold."""
import time

import numpy as np
import torch

from lewm.dense_metric_goal_control_development import MetricGoalControl
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import train_go2_direct_visual_goal_readout_development as fitted


class VisualArrivalControl(MetricGoalControl):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.arrival_readout = fitted.load().cuda()
        self.pooled_goal = pool_tokens(self.goal[None])
        self.arrival_latched = False
        self.first_arrival_ns = None
        self.hold_pending = None

    @torch.inference_mode()
    def observe(self,packet):
        record = super().observe(packet)
        now = int(packet['image']['measured_ns'])
        if self.hold_pending is not None:
            assert now == self.hold_pending[0]+500_000_000
            executed = np.asarray(packet['sensor_state']['control']['applied_command']['values'])[-5:]
            np.testing.assert_allclose(executed,self.hold_pending[1],rtol=0,atol=1e-6)
            self.hold_pending = None
        estimate = self.arrival_readout(pool_tokens(self.history[-1][1][None]),self.pooled_goal)[0].cpu().numpy()
        assert np.isfinite(estimate).all()
        within = bool(np.linalg.norm(estimate[:2]) <= .03 and abs(estimate[2]) <= np.deg2rad(5.))
        if within and len(self.history)==3 and not self.arrival_latched:
            self.arrival_latched = True
            self.first_arrival_ns = now
        record.update(estimated_goal_motion=estimate.tolist(),estimated_within_goal=within,
            arrival_latched=self.arrival_latched,first_arrival_ns=self.first_arrival_ns)
        return record

    @torch.inference_mode()
    def choose(self,packet):
        if not self.arrival_latched:
            return super().choose(packet)|dict(arrival_latched=False,controller_mode='world_model_planning')
        start = time.monotonic(); now = int(packet['image']['measured_ns'])
        assert self.history[-1][0] == now and len(self.history)==3
        commands = np.asarray(packet['sensor_state']['control']['applied_command']['values'],np.float32)
        requested = [[0.,0.,0.]]*5
        applied = np.asarray(apply_safety_limits_single(requested,tuple(commands[-1]),self.limits)[0],np.float32)
        self.pending = None; self.hold_pending = (now,applied)
        return dict(observed_ns=now,action='hold',action_index=0,requested_commands=requested,
            expected_applied_commands=applied.tolist(),costs=None,cost_kind='observed_visual_arrival_hold',
            tied_indices=[0],forecast_horizon_ms=None,commit_ticks=5,planning_wall_s=time.monotonic()-start,
            arrival_latched=True,controller_mode='terminal_hold',first_arrival_ns=self.first_arrival_ns)

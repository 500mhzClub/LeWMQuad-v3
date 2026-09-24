"""Fixed discrete visual feedback using signed image-goal estimates."""
import time

import numpy as np
import torch

from lewm.dense_visual_arrival_control_development import VisualArrivalControl
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import apply_safety_limits_single


class DirectVisualFeedbackControl(VisualArrivalControl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Shared initialization preserves the exact encoder/readout path. The
        # predictor is never called by this controller and is released here.
        del self.model

    @torch.inference_mode()
    def observe(self, packet):
        record = super().observe(packet)
        self.estimated_goal = np.asarray(record['estimated_goal_motion'])
        return record

    @torch.inference_mode()
    def choose(self, packet):
        if self.arrival_latched:
            return super().choose(packet)
        start = time.monotonic()
        now = int(packet['image']['measured_ns'])
        assert now == self.history[-1][0] and len(self.history) == 3
        dx, dy, heading = self.estimated_goal
        rho = float(np.hypot(dx, dy))
        bearing = float(np.arctan2(dy, dx))
        wrap = lambda x: float(np.arctan2(np.sin(x), np.cos(x)))
        if rho <= .03:
            # Position reached: align final heading without moving forward.
            desired_yaw = float(np.clip(1.5*heading, -.45, .45))
            candidates = (4, 5)
        elif abs(bearing) > np.pi/2:
            desired_yaw = float(np.clip(1.5*bearing, -.45, .45))
            candidates = (4, 5)
        else:
            desired_yaw = float(np.clip(1.5*bearing-.5*wrap(heading-bearing), -.45, .45))
            candidates = (1, 2, 3)
        # Among the permitted forward or turn primitives, choose the closest
        # steering rate. Hold is reserved for the common arrival recognizer.
        index = min(candidates, key=lambda i: abs(candidate_commands(ACTIONS[i])[0][2]-desired_yaw))
        requested = [candidate_commands(ACTIONS[index])[0]]*5
        commands = np.asarray(packet['sensor_state']['control']['applied_command']['values'], np.float32)
        applied = np.asarray(apply_safety_limits_single(requested, tuple(commands[-1]), self.limits)[0], np.float32)
        self.pending = None
        self.hold_pending = (now, applied)
        return dict(observed_ns=now, action=ACTIONS[index], action_index=index,
            requested_commands=requested, expected_applied_commands=applied.tolist(),
            costs=None, cost_kind='direct_visual_feedback', tied_indices=[index],
            forecast_horizon_ms=None, commit_ticks=5, planning_wall_s=time.monotonic()-start,
            arrival_latched=False, controller_mode='direct_visual_feedback',
            estimated_goal_motion=self.estimated_goal.tolist(), estimated_distance_m=rho,
            estimated_bearing_rad=bearing, desired_yaw_rate=desired_yaw)

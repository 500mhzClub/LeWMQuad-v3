"""Bounded residual memory driven only by sequential measured public poses."""
from collections import deque
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.observation_horizon_predictive_selection_development import require_short_forecast
from lewm.causal_executed_residual_diagnosis_development import WINDOW_TICKS


class OnlineExecutedResidual:
    """The executor must issue each remembered request before the next packet.

    The native collector and raw audit own that execution check. This memory
    neither integrates commands into pose nor certifies that hardware executed.
    Residuals use original forecasts, never the already adjusted scoring pose.
    """
    def __init__(self):
        self.frame = -1
        self.now_ns = None
        self.pose = None
        self.pending = None
        self.history = deque(maxlen=WINDOW_TICKS)

    def observe(self, evidence, *, now_ns):
        p, R, pose = current_joint_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
        frame = pose['frame']
        if frame != self.frame+1 or now_ns != 1_500_000_000+frame*100_000_000:
            raise ValueError('uninterrupted observed residual clock required')
        new = None
        if self.pending is not None:
            old = self.pending
            if old['tick'] != frame-1 or old['measured_ns']+100_000_000 != now_ns:
                raise ValueError('only immediately preceding command may receive an observed label')
            observed = (np.asarray(old['rotation']).T@(p-np.asarray(old['position'])))[:2]
            prediction = np.asarray(old['predicted_body_xy_m'])
            new = dict(tick=old['tick'], available_tick=frame, measured_ns=now_ns,
                action=old['action'], requested_command=old['requested_command'],
                predicted_body_xy_m=prediction.tolist(), observed_body_xy_m=observed.tolist(),
                residual_xy_m=(prediction-observed).tolist(),
                start_rgb_sha256=old['rgb_sha256'], start_depth_sha256=old['depth_sha256'],
                end_rgb_sha256=pose['rgb_sha256'], end_depth_sha256=pose['depth_sha256'])
        if new is not None: self.history.append(new)
        while self.history and frame-self.history[0]['tick'] > WINDOW_TICKS: self.history.popleft()
        self.frame = frame; self.now_ns = now_ns; self.pending = None
        self.pose = dict(position=p.tolist(), rotation=R.tolist(),
            rgb_sha256=pose['rgb_sha256'], depth_sha256=pose['depth_sha256'])

    def remember(self, result):
        if result['tick'] != self.frame or self.pose is None or self.pending is not None:
            raise ValueError('one requested command per current observed residual frame required')
        s = result['new_selection']
        if result['terminal'] is not None or not s or 'prediction' not in s: return
        require_short_forecast(s)
        prediction = np.asarray(s['prediction'], float)
        if (prediction.shape != (6, 8, 5) or not np.isfinite(prediction).all()
                or s.get('actual_commitment_horizon_ns') != 100_000_000
                or [r['action'] for r in s['candidates']] != list(ACTIONS)):
            raise ValueError('original ordered six-action one-step forecast required')
        matches = [i for i, action in enumerate(ACTIONS)
            if result['requested_command'] == candidate_commands(action)[0]]
        if len(matches) != 1: raise ValueError('actual requested command must match one original candidate')
        i = matches[0]
        self.pending = dict(tick=self.frame, measured_ns=self.now_ns, action=ACTIONS[i],
            requested_command=list(result['requested_command']),
            predicted_body_xy_m=prediction[i, 0, :2].tolist(), **deepcopy(self.pose))

    def snapshot(self):
        bias = np.mean([r['residual_xy_m'] for r in self.history], axis=0) if self.history else np.zeros(2)
        return dict(frame=self.frame, measured_ns=self.now_ns, window_ticks=WINDOW_TICKS,
            correction_xy_m=bias.tolist(), observed_residual_samples=len(self.history),
            residual_source_ticks=[r['tick'] for r in self.history],
            residual_available_ticks=[r['available_tick'] for r in self.history],
            residuals=deepcopy(list(self.history)), pending_forecast_tick=None if self.pending is None else self.pending['tick'],
            native_outcomes_used=False, command_integrated_pose_used=False,
            model_weights_changed=False, hardware_execution_certified=False)

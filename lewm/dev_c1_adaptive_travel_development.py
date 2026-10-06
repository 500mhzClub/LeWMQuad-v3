"""Adaptive C1 baseline, C1A (development; Andrew, 5 October 2026, dynamics stage 2).

C1 forecasts motion from commands only (lewm/short_pulse_navigation_runtime_development.py, command_history). C1A
scales C1's XY forecast by the recent ratio of tracked to predicted travel. It separates reacting to slip, which C1A can
do, from anticipating it, which needs a visual cue. It uses the tracker pose only, with no privileged data and no
training.

At each decision (time t, frame f), for an earlier decision of the same mission at t0 = t - h*100 ms, with h from 4 to 8
and the largest available:
- C1's model re-forecasts the displacement over [t0, t). It uses the command history C1 itself saw at t0 and the
  commands the controller requested over [t0, t), rebuilt from its own decision log (committed prefix, then the
  selected command for its commit duration, then stop, until a later decision takes over).
- The tracker gives the displacement between frames f0 and f, in the body frame at t0.
- A sample is |tracked| / |predicted|, taken only when the predicted travel is at least 20 mm.

The ratio is the median of the samples from the last 3 s, clipped to [0.3, 1.5], once there are at least two such
samples; otherwise it is 1.0. It multiplies the XY of every candidate's forecast at every horizon; yaw is unchanged.

Receipt fields (planning.json motion_correction):
- dev_adaptive_forecast_xy_yaw: the applied forecast;
- dev_adaptive_travel: the ratio, its samples and the parameters.
command_history_forecast_xy_yaw stays C1's own forecast.
"""
from collections import deque

import numpy as np

from lewm.short_pulse_navigation_runtime_development import command_predictions, past_commands

STEP_NS = 100_000_000
WINDOW_NS = 3_000_000_000
MIN_PREDICTED_M = .02
CLIP = (.3, 1.5)
MIN_SAMPLES = 2
HORIZONS = range(8, 3, -1)  # prefer the longest window, 800 ms down to 400 ms


def requested_ticks(planning, start_ns, steps):
    """The requested command at start_ns + k*100 ms, k < steps, from the controller's own decision log, or None."""
    records = [r for r in planning if 'selection' in r and 'committed_prefix' in r and 'measured_ns' in r]
    out = []
    for k in range(steps):
        tick = start_ns+k*STEP_NS
        prior = [r for r in records if r['measured_ns'] <= tick]
        if not prior:
            return None
        r = prior[-1]
        j = (tick-r['measured_ns'])//STEP_NS
        prefix = list(r['committed_prefix'])
        duration = int(r['selection'].get('command_duration_ns', 4*STEP_NS))//STEP_NS
        command = r['selection'].get('requested_command')
        if command is None:
            return None
        if j < len(prefix):
            out.append(prefix[j])
        elif j < len(prefix)+duration:
            out.append(command)
        else:
            out.append([0., 0., 0.])
    return np.asarray(out, float)


def tracked_displacement(poses, frame0, frame1):
    """Displacement from frame0 to frame1 in the body frame at frame0 (tracker poses), XY metres."""
    p0 = np.asarray(poses[frame0]['position_initial_body_m'], float)
    R0 = np.asarray(poses[frame0]['rotation_initial_body_from_current_body'], float)
    p1 = np.asarray(poses[frame1]['position_initial_body_m'], float)
    return (R0.T@(p1-p0))[:2]


def travel_sample(model, decisions, planning, poses, now_ns, frame):
    """One tracked/predicted travel sample against the longest usable earlier decision, or None."""
    by_ns = {d['ns']: d for d in decisions}
    for h in HORIZONS:
        d = by_ns.get(now_ns-h*STEP_NS)
        if d is None or d['frame'] not in poses or frame not in poses:
            continue
        ticks = requested_ticks(planning, d['ns'], h)
        if ticks is None:
            continue
        sequence = np.zeros((6, 8, 3))
        sequence[:, :h] = ticks
        sequence[:, h:] = ticks[-1]
        predicted = command_predictions(model, d['history'], sequence)[0, h-1, :2]
        if np.linalg.norm(predicted) < MIN_PREDICTED_M:
            return None
        tracked = tracked_displacement(poses, d['frame'], frame)
        return dict(ns=int(now_ns), horizon_steps=h, predicted_m=predicted.tolist(), tracked_m=tracked.tolist(),
                    ratio=float(np.linalg.norm(tracked)/np.linalg.norm(predicted)))
    return None


def current_ratio(samples, now_ns):
    recent = [s['ratio'] for s in samples if now_ns-s['ns'] <= WINDOW_NS]
    if len(recent) < MIN_SAMPLES:
        return 1., len(recent)
    return float(np.clip(np.median(recent), *CLIP)), len(recent)


class AdaptiveTravelMixin:
    """Outermost mixin for C1 (prediction_source command_history) only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, 'pulse_prediction_source', None) != 'command_history':
            raise ValueError('C1A wraps the command-history (C1) controller only')
        self.adaptive_decisions = deque(maxlen=64)
        self.adaptive_samples = deque(maxlen=64)

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        selected, receipt = super()._correct_prediction(prediction, packet, evidence, prefix)
        now, frame = int(packet.measured_ns), int(packet.frame)
        with self.correction_pose_lock:
            poses = dict(self.correction_poses)
        sample = travel_sample(self.command_model, list(self.adaptive_decisions), list(self.planning), poses, now, frame)
        if sample is not None:
            self.adaptive_samples.append(sample)
        self.adaptive_decisions.append(dict(ns=now, frame=frame, history=past_commands(packet.history, now)))
        ratio, used = current_ratio(self.adaptive_samples, now)
        adapted = selected.copy()
        adapted[:, :, :2] *= ratio
        if not np.isfinite(adapted).all():
            raise ValueError('finite adaptive forecast required')
        yaw = np.arctan2(adapted[:, :, 2], adapted[:, :, 3])
        receipt = dict(receipt, prediction_source='command_history', controller_variant='C1A',
                       corrected_forecast_xy_m=adapted[:, :, :2].tolist(),
                       dev_adaptive_forecast_xy_yaw=np.concatenate((adapted[:, :, :2], yaw[..., None]), axis=-1).tolist(),
                       dev_adaptive_travel=dict(ratio=ratio, samples_used=used, new_sample=sample,
                                                window_s=WINDOW_NS/1e9, clip=list(CLIP), min_samples=MIN_SAMPLES,
                                                min_predicted_m=MIN_PREDICTED_M, horizons_steps=list(HORIZONS),
                                                inputs='tracker pose and own requested commands only'))
        return adapted, receipt

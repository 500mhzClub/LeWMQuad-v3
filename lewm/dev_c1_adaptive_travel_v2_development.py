"""Adaptive C1 baseline, C1A v2: separate translation and rotation ratios (development; parameters fixed by Andrew,
5 October 2026, before any stage-2 run).

It extends lewm/dev_c1_adaptive_travel_development.py (v1, unchanged), with two ratios:
- translation: tracked / predicted XY travel;
- rotation: tracked / predicted heading change.
Both compare the tracker pose with C1's own forecast. At each decision (time t, frame f), against an earlier decision
of the mission at t0 = t - h*100 ms (h from 8 down to 4, the longest available):
- C1's model re-forecasts [t0, t), with the command history C1 saw at t0 and the commands the controller requested over
  [t0, t), rebuilt from its own decision log;
- the tracker gives the XY displacement (body frame at t0) and the heading change between frames f0 and f.
Only windows with meaningful commanded motion give samples:
- a window whose requested commands are all hold gives none;
- a translation sample needs |predicted XY| >= 2 cm;
- a rotation sample needs |predicted heading change| >= 2 degrees.
Each ratio is the median of its own samples from the last 3 s, clipped to [0.3, 1.5], once it has at least two such
samples; otherwise it is 1.0.

Applied to every candidate at every horizon: C1's XY is multiplied by the translation ratio and its yaw by the
rotation ratio. Receipt fields:
- dev_adaptive_forecast_xy_yaw: the applied forecast;
- dev_adaptive_travel: both ratios, the samples and the parameters.
"""
from collections import deque

import numpy as np

from lewm.dev_c1_adaptive_travel_development import HORIZONS, STEP_NS, requested_ticks, tracked_displacement
from lewm.short_pulse_navigation_runtime_development import command_predictions, past_commands

WINDOW_NS = 3_000_000_000
MIN_TRANSLATION_M = .02
MIN_ROTATION_RAD = float(np.radians(2.))
CLIP = (.3, 1.5)
MIN_SAMPLES = 2
VERSION = 'C1A-v2'


def tracked_heading_change(poses, frame0, frame1):
    R0 = np.asarray(poses[frame0]['rotation_initial_body_from_current_body'], float)
    R1 = np.asarray(poses[frame1]['rotation_initial_body_from_current_body'], float)
    relative = R0.T@R1
    return float(np.arctan2(relative[1, 0], relative[0, 0]))


def motion_samples(model, decisions, planning, poses, now_ns, frame):
    """Translation and rotation samples against the longest usable earlier decision; either may be None."""
    by_ns = {d['ns']: d for d in decisions}
    for h in HORIZONS:
        d = by_ns.get(now_ns-h*STEP_NS)
        if d is None or d['frame'] not in poses or frame not in poses:
            continue
        ticks = requested_ticks(planning, d['ns'], h)
        if ticks is None:
            continue
        if not np.any(np.abs(ticks) > 1e-9):
            return None, None  # all hold: no meaningful commanded motion
        sequence = np.zeros((6, 8, 3))
        sequence[:, :h] = ticks
        sequence[:, h:] = ticks[-1]
        forecast = command_predictions(model, d['history'], sequence)[0, h-1]
        base = dict(ns=int(now_ns), horizon_steps=h)
        translation = rotation = None
        if np.linalg.norm(forecast[:2]) >= MIN_TRANSLATION_M:
            tracked = tracked_displacement(poses, d['frame'], frame)
            translation = dict(base, predicted_m=forecast[:2].tolist(), tracked_m=tracked.tolist(),
                               ratio=float(np.linalg.norm(tracked)/np.linalg.norm(forecast[:2])))
        if abs(forecast[2]) >= MIN_ROTATION_RAD:
            turned = tracked_heading_change(poses, d['frame'], frame)
            rotation = dict(base, predicted_rad=float(forecast[2]), tracked_rad=turned,
                            ratio=float(abs(turned)/abs(forecast[2])))
        return translation, rotation
    return None, None


def current_ratio(samples, now_ns):
    recent = [s['ratio'] for s in samples if now_ns-s['ns'] <= WINDOW_NS]
    if len(recent) < MIN_SAMPLES:
        return 1., len(recent)
    return float(np.clip(np.median(recent), *CLIP)), len(recent)


def adapt(selected, translation_ratio, rotation_ratio):
    """C1's (6, 8, 5) forecast [x, y, sin yaw, cos yaw, contact] with XY and yaw scaled."""
    adapted = selected.copy()
    adapted[:, :, :2] *= translation_ratio
    yaw = np.arctan2(selected[:, :, 2], selected[:, :, 3])*rotation_ratio
    adapted[:, :, 2], adapted[:, :, 3] = np.sin(yaw), np.cos(yaw)
    return adapted, yaw


class AdaptiveTravelMixin:
    """Outermost mixin for C1 (prediction_source command_history) only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, 'pulse_prediction_source', None) != 'command_history':
            raise ValueError('C1A wraps the command-history (C1) controller only')
        self.adaptive_decisions = deque(maxlen=64)
        self.adaptive_translation = deque(maxlen=64)
        self.adaptive_rotation = deque(maxlen=64)

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        selected, receipt = super()._correct_prediction(prediction, packet, evidence, prefix)
        now, frame = int(packet.measured_ns), int(packet.frame)
        with self.correction_pose_lock:
            poses = dict(self.correction_poses)
        translation, rotation = motion_samples(self.command_model, list(self.adaptive_decisions), list(self.planning),
                                               poses, now, frame)
        if translation is not None:
            self.adaptive_translation.append(translation)
        if rotation is not None:
            self.adaptive_rotation.append(rotation)
        self.adaptive_decisions.append(dict(ns=now, frame=frame, history=past_commands(packet.history, now)))
        t_ratio, t_used = current_ratio(self.adaptive_translation, now)
        r_ratio, r_used = current_ratio(self.adaptive_rotation, now)
        adapted, yaw = adapt(selected, t_ratio, r_ratio)
        if not np.isfinite(adapted).all():
            raise ValueError('finite adaptive forecast required')
        receipt = dict(receipt, prediction_source='command_history', controller_variant=VERSION,
                       corrected_forecast_xy_m=adapted[:, :, :2].tolist(),
                       dev_adaptive_forecast_xy_yaw=np.concatenate((adapted[:, :, :2], yaw[..., None]), axis=-1).tolist(),
                       dev_adaptive_travel=dict(
                           version=VERSION, translation_ratio=t_ratio, rotation_ratio=r_ratio,
                           translation_samples_used=t_used, rotation_samples_used=r_used,
                           new_translation_sample=translation, new_rotation_sample=rotation,
                           window_s=WINDOW_NS/1e9, statistic='median', start=1., clip=list(CLIP), min_samples=MIN_SAMPLES,
                           min_translation_m=MIN_TRANSLATION_M, min_rotation_deg=2., holds_excluded=True,
                           horizons_steps=list(HORIZONS), inputs='tracker pose and own requested commands only'))
        return adapted, receipt

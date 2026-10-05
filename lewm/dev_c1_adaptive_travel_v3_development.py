"""Adaptive C1 baseline, C1A v3: ratios per movement type (development; Andrew, 5 October 2026, designed on stage-1 data
only, frozen before any stage-2 evaluation).

C1A v2 (lewm/dev_c1_adaptive_travel_v2_development.py, unchanged) keeps one translation ratio and one rotation ratio.
At uniform μ = 0.3 its translation ratio stayed near 1.0, because C1's error there depends on the movement:
- cruise is over-predicted (x1.17);
- turning translation is under-predicted (x0.55);
- arcs are x0.91 and switches x1.01.
v3 therefore keeps a separate translation ratio and rotation ratio for each movement type: cruise, arc_steady, turn and
switch.

Movement types use the feature-cache rule (scripts/build_go2_dev_c3_feature_cache_development.category), as the
closed-loop scorer does: an 8-step command tape against the 1.5 s of commands before it.
- **A sample** is classified by the commands the controller requested over its window [t0, t0 + h*100 ms), padded
  with the last of them, against the requested commands of the preceding 1.5 s. Both come from the controller's own
  decision log.
- **A candidate** at decision t is classified by its committed prefix followed by its command (held), against the
  requested commands of the 1.5 s before t.
- **Hold and rest_start** are never adapted (ratio 1.0).

Sampling and the rules per ratio are v2's:
- samples need meaningful commanded motion (all-hold windows excluded; at least 2 cm or 2 degrees predicted);
- each ratio is the median over a 3-s window, 1.0 until there are two samples, clipped to [0.3, 1.5].
Each candidate's XY is scaled by its type's translation ratio and its yaw by its type's rotation ratio.
Inputs are the tracker pose and the controller's own requested commands only.
"""
from collections import defaultdict, deque

import numpy as np

from lewm.dev_c1_adaptive_travel_development import HORIZONS, STEP_NS, requested_ticks, tracked_displacement
from lewm.dev_c1_adaptive_travel_v2_development import (CLIP, MIN_ROTATION_RAD, MIN_SAMPLES, MIN_TRANSLATION_M, WINDOW_NS,
                                                        current_ratio, tracked_heading_change)
from lewm.short_pulse_navigation_runtime_development import command_predictions, past_commands
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.build_go2_dev_c3_feature_cache_development import category

ADAPTED = ('cruise', 'arc_steady', 'turn', 'switch')
VERSION = 'C1A-v3'
PAST_STEPS = 15


def past_requested(planning, end_ns):
    """The 15 requested commands before end_ns (zeros where the log has none yet)."""
    out = np.zeros((PAST_STEPS, 3))
    start = end_ns-PAST_STEPS*STEP_NS
    for k in range(PAST_STEPS):
        tick = requested_ticks(planning, start+k*STEP_NS, 1)
        if tick is not None:
            out[k] = tick[0]
    return out


def window_type(ticks, past):
    tape = np.zeros((8, 3))
    tape[:len(ticks)] = ticks
    tape[len(ticks):] = ticks[-1]
    return category(tape, past)


def candidate_types(prefix, pulse, past):
    sequences = command_sequences(prefix, pulse=pulse)
    types = []
    for i in range(len(sequences)):
        tape = np.concatenate((sequences[i, :3], np.repeat(sequences[i, 3:4], 5, axis=0)))
        types.append(category(tape, past))
    return types


def motion_samples(model, decisions, planning, poses, now_ns, frame):
    """(movement type, translation sample, rotation sample) against the longest usable earlier decision."""
    by_ns = {d['ns']: d for d in decisions}
    for h in HORIZONS:
        d = by_ns.get(now_ns-h*STEP_NS)
        if d is None or d['frame'] not in poses or frame not in poses:
            continue
        ticks = requested_ticks(planning, d['ns'], h)
        if ticks is None:
            continue
        if not np.any(np.abs(ticks) > 1e-9):
            return None, None, None
        kind = window_type(ticks, past_requested(planning, d['ns']))
        sequence = np.zeros((6, 8, 3))
        sequence[:, :h] = ticks
        sequence[:, h:] = ticks[-1]
        forecast = command_predictions(model, d['history'], sequence)[0, h-1]
        base = dict(ns=int(now_ns), horizon_steps=h, movement=kind)
        translation = rotation = None
        if np.linalg.norm(forecast[:2]) >= MIN_TRANSLATION_M:
            tracked = tracked_displacement(poses, d['frame'], frame)
            translation = dict(base, ratio=float(np.linalg.norm(tracked)/np.linalg.norm(forecast[:2])))
        if abs(forecast[2]) >= MIN_ROTATION_RAD:
            turned = tracked_heading_change(poses, d['frame'], frame)
            rotation = dict(base, ratio=float(abs(turned)/abs(forecast[2])))
        return kind, translation, rotation
    return None, None, None


def adapt(selected, types, translation, rotation):
    """Scale each candidate's XY and yaw by its movement type's ratios (1.0 for hold and rest_start)."""
    adapted = selected.copy()
    yaw = np.arctan2(selected[:, :, 2], selected[:, :, 3])
    for i, kind in enumerate(types):
        t, r = translation.get(kind, 1.), rotation.get(kind, 1.)
        adapted[i, :, :2] *= t
        yaw[i] *= r
    adapted[:, :, 2], adapted[:, :, 3] = np.sin(yaw), np.cos(yaw)
    return adapted, yaw


class AdaptiveTravelMixin:
    """Outermost mixin for C1 (prediction_source command_history) only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, 'pulse_prediction_source', None) != 'command_history':
            raise ValueError('C1A wraps the command-history (C1) controller only')
        self.adaptive_decisions = deque(maxlen=64)
        self.adaptive_translation = defaultdict(lambda: deque(maxlen=64))
        self.adaptive_rotation = defaultdict(lambda: deque(maxlen=64))

    def _correct_prediction(self, prediction, packet, evidence, prefix):
        selected, receipt = super()._correct_prediction(prediction, packet, evidence, prefix)
        now, frame = int(packet.measured_ns), int(packet.frame)
        with self.correction_pose_lock:
            poses = dict(self.correction_poses)
        planning = list(self.planning)
        kind, translation, rotation = motion_samples(self.command_model, list(self.adaptive_decisions), planning, poses, now, frame)
        if kind in ADAPTED:
            if translation is not None:
                self.adaptive_translation[kind].append(translation)
            if rotation is not None:
                self.adaptive_rotation[kind].append(rotation)
        self.adaptive_decisions.append(dict(ns=now, frame=frame, history=past_commands(packet.history, now)))
        t_ratios, r_ratios, used = {}, {}, {}
        for k in ADAPTED:
            t_ratios[k], tu = current_ratio(self.adaptive_translation[k], now)
            r_ratios[k], ru = current_ratio(self.adaptive_rotation[k], now)
            used[k] = [tu, ru]
        types = candidate_types(prefix, bool(self.planning_translation_pulse), past_requested(planning, now))
        adapted, yaw = adapt(selected, types, t_ratios, r_ratios)
        if not np.isfinite(adapted).all():
            raise ValueError('finite adaptive forecast required')
        receipt = dict(receipt, prediction_source='command_history', controller_variant=VERSION,
                       corrected_forecast_xy_m=adapted[:, :, :2].tolist(),
                       dev_adaptive_forecast_xy_yaw=np.concatenate((adapted[:, :, :2], yaw[..., None]), axis=-1).tolist(),
                       dev_adaptive_travel=dict(
                           version=VERSION, candidate_movement=types, translation_ratio=t_ratios, rotation_ratio=r_ratios,
                           samples_used=used, new_sample_movement=kind, new_translation_sample=translation,
                           new_rotation_sample=rotation, adapted_types=list(ADAPTED), window_s=WINDOW_NS/1e9,
                           statistic='median', start=1., clip=list(CLIP), min_samples=MIN_SAMPLES,
                           min_translation_m=MIN_TRANSLATION_M, min_rotation_deg=2., holds_excluded=True,
                           horizons_steps=list(HORIZONS), inputs='tracker pose and own requested commands only'))
        return adapted, receipt

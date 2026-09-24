"""Unchanged two-leg endpoints plus evaluation-only initial-heading diagnostics."""
import math

import numpy as np

from lewm.observed_continuation_metrics_development import reduce_continuation
from lewm.physical_execution_development import rotation_xyzw


def initial_alignment_geometry(raw, start, decisions):
    chosen = [r for r in decisions if r['controller'].get('selected_initial_bearing') is not None]
    first = [r for r in decisions if r['controller']['stage'] == 'FIRST']
    if len(chosen) > 1: raise ValueError('one initial selected bearing maximum')
    if not chosen or not first: return None
    selected = chosen[0]['controller']['selected_initial_bearing']
    direction = selected['direction_initial_body']
    target = math.atan2(direction[1], direction[0])
    index = first[0]['pre_sample_index']
    relative = rotation_xyzw(raw['base_pose_world'][start, 3:]).T@rotation_xyzw(raw['base_pose_world'][index, 3:])
    heading = math.atan2(relative[1, 0], relative[0, 0])
    error = abs(math.atan2(math.sin(target-heading), math.cos(target-heading)))
    return {'observed_target_heading_initial_rad': target,
            'first_start_true_heading_initial_rad': heading,
            'first_start_abs_heading_error_rad': error,
            'first_start_sample_index': index,
            'first_start_xy_world_m': raw['base_pose_world'][index, :2].tolist(),
            'initial_phase_actual_translation_m': float(np.linalg.norm(raw['base_pose_world'][index, :2]-raw['base_pose_world'][start, :2])),
            'evaluation_only': True, 'corridor_centering_qualified': False}


def reduce_aligned_continuation(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, model):
    result = reduce_continuation(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, model)
    return {**result, 'initial_alignment_geometry': initial_alignment_geometry(raw, start, decisions)}

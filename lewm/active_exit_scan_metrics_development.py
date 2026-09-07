"""Evaluation-only scan accounting. Geometry and true pose never guide scans."""
import math

import numpy as np

from lewm.gyro_turn_assay_development import reduce_turn
from lewm.physical_execution_development import rotation_xyzw


def candidate_side(pose, bearing, pitch):
    """Which central-cell boundary a nominal proposal ray intersects first.

    This tests an opening-directed bearing, NOT finite-body traversability.
    Outside-cell origins, vertical rays and exact corners are unscorable.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or not math.isfinite(bearing) or not math.isfinite(pitch) or pitch <= 0:
        raise ValueError('finite physical pose, bearing and positive pitch required')
    position = pose[:2]
    if np.any(np.abs(position) >= pitch / 2):
        return None
    direction = (rotation_xyzw(pose[3:]) @ [math.cos(bearing), math.sin(bearing), 0.])[:2]
    distances = []
    for axis in range(2):
        if abs(direction[axis]) > 1e-12:
            sign = 1 if direction[axis] > 0 else -1
            distances.append(((sign * pitch / 2 - position[axis]) / direction[axis], axis, sign))
    distances.sort()
    if not distances or (len(distances) > 1 and abs(distances[0][0] - distances[1][0]) <= 1e-9):
        return None
    _, axis, sign = distances[0]
    return [sign, 0] if axis == 0 else [0, sign]


def opening_accounting(spec, raw, decisions, *, selected_views):
    expected = {tuple(d) for d in spec['evaluation_open_directions']}
    hits, misses, unknown = [], [], []
    frames = [d for d in decisions if not selected_views or d['selected_view']]
    for row in frames:
        pose = raw['base_pose_world'][row['pre_sample_index']]
        for candidate in row['observation']['candidate_rows']:
            side = candidate_side(pose, candidate['bearing_body_rad'], spec['width_m'] + .08)
            item = {'observation_id': candidate['observation_id'], 'side': side}
            (unknown if side is None else hits if tuple(side) in expected else misses).append(item)
    covered = {tuple(item['side']) for item in hits}
    return {'frames': len(frames), 'proposals': len(hits) + len(misses) + len(unknown),
            'opening_directed': len(hits), 'closed_side_directed': len(misses), 'unscorable': len(unknown),
            'open_sides_expected': len(expected), 'open_sides_covered': len(covered),
            'covered_directions': [list(d) for d in sorted(covered)],
            'missed_directions': [list(d) for d in sorted(expected - covered)],
            'all_open_sides_covered_without_false_proposals': covered == expected and not misses and not unknown,
            'proposal_evidence': hits + misses + unknown,
            'qualified_traversals': 0}


def reduce_scan(spec, raw, start, decisions, terminal, stop_reason, sensor_fault):
    response = reduce_turn(raw, start, decisions, 2 * math.pi, terminal, stop_reason)
    response['sensor_fault'] = sensor_fault
    response['checks']['no_sensor_fault'] = sensor_fault is None
    response['checks']['all_four_target_views'] = bool(decisions and decisions[-1]['controller']['completed_target_views'] == 4)
    response['physical_task_success'] = all(response['checks'].values())
    response['completed_target_views'] = decisions[-1]['controller']['completed_target_views'] if decisions else 0
    response['initial_view'] = opening_accounting(spec, raw, decisions[:1], selected_views=True)
    response['selected_views'] = opening_accounting(spec, raw, decisions, selected_views=True)
    response['all_control_frames'] = opening_accounting(spec, raw, decisions, selected_views=False)
    response['scope'] = 'finite local scan and opening-directed proposal evidence; no traversal, place identity, maze or hardware qualification'
    return response

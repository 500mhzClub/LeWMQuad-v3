"""Evaluation-only physical outcomes; never imported by the runtime controller."""
import math

import numpy as np

from lewm.physical_execution_development import rotation_xyzw


def pose_checks(spec, raw, index, model):
    geometry = spec['geometry']
    edge = geometry['selected_directed_edge']
    midpoint = np.mean(edge['opening_segment_world'], axis=0)
    normal = np.asarray(edge['opening_normal_world'], dtype=float)
    direction = np.array([*normal, 0.])
    pose = raw['base_pose_world'][index]
    rotation = rotation_xyzw(pose[3:])
    support = model.supports(raw['joint_position'][index], (rotation.T@direction)[None])
    distance = float((pose[:2]-midpoint)@normal)
    body_past = distance+support['lower'][0]
    polygon = np.asarray(geometry['target_node']['boundary_polygon_world'])
    inside = bool(np.all(pose[:2] >= polygon.min(0)) and np.all(pose[:2] <= polygon.max(0)))
    return {'base_past_opening_m': distance, 'whole_body_past_opening_m': body_past,
            'base_inside_destination': inside,
            'whole_body_past_opening': bool(body_past >= .02)}


def reduce_traversal(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, model):
    pose = raw['base_pose_world'][-1]
    rotation = rotation_xyzw(pose[3:])
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[2, 1], rotation[2, 2]))
    edge = spec['geometry']['selected_directed_edge']
    midpoint = np.mean(edge['opening_segment_world'], axis=0)
    normal = np.asarray(edge['opening_normal_world'])
    distances = (raw['base_pose_world'][start:, :2]-midpoint)@normal
    sustained = bool(len(distances) >= 150 and np.all(distances[-150:] >= .15))
    release = np.flatnonzero(raw['phase'] == 2)
    window = release[-100:]
    settled = bool(len(release) == 250 and len(window) == 100
        and np.all(np.linalg.norm(raw['base_twist_world'][window, :2], axis=1) <= .1)
        and np.all(np.abs(raw['base_twist_world'][window, 5]) <= .25))
    final = pose_checks(spec, raw, len(raw['timestamp_s'])-1, model)
    physical_checks = {
        'no_physical_stop': stop_reason is None,
        'no_contact': not bool(raw['physics_contact'].any()),
        'sustained_center_crossing': sustained,
        'whole_body_past_opening': final['whole_body_past_opening'],
        'base_inside_destination': final['base_inside_destination'],
        'release_motion': settled,
        'terminal_body_stable': bool(pose[2] >= .2 and abs(roll) <= .5 and abs(pitch) <= .5)}
    candidates = [r for r in decisions if r['controller']['status'] == 'ARRIVAL_CANDIDATE']
    if len(candidates) > 1: raise ValueError('one provisional arrival maximum')
    candidate_pose = pose_checks(spec, raw, candidates[0]['pre_sample_index'], model) if candidates else None
    nominal = terminal == 'ARRIVAL_CANDIDATE'
    if nominal != bool(candidates): raise ValueError('terminal and candidate evidence disagree')
    actual_arrival = all(physical_checks.values())
    candidate_crossed = bool(candidate_pose and candidate_pose['whole_body_past_opening']
                             and candidate_pose['base_inside_destination'])
    progress = float((pose[:2]-raw['base_pose_world'][start, :2])@normal)
    return {'controller_terminal': terminal, 'stop_reason': stop_reason, 'sensor_fault': sensor_fault,
        'physical_checks': physical_checks, 'physical_arrival': actual_arrival,
        'nominal_arrival_candidate': nominal, 'candidate_pose': candidate_pose,
        'candidate_geometry_agrees': candidate_crossed,
        'false_arrival_candidate': nominal and not candidate_crossed,
        'candidate_without_viable_release': nominal and not actual_arrival,
        'physical_arrival_without_candidate': actual_arrival and not nominal,
        'integration_success': nominal and candidate_crossed and actual_arrival and sensor_fault is None,
        'final_pose_checks': final, 'actual_progress_m': progress,
        'maximum_center_past_opening_m': float(distances.max()),
        'total_post_settle_seconds': float(raw['timestamp_s'][-1]-raw['timestamp_s'][start]),
        'trusted_graph_edges': 0,
        'scope': 'development one-transition geometry/release evaluation; not place recognition, maze success or hardware evidence'}

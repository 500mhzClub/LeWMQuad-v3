"""Evaluation-only physical return and marker geometry; never controller input."""
import math

import numpy as np


def ray_box_entry(origin, target, box):
    yaw = box['yaw_rad']
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                         [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
    p = rotation.T@(np.asarray(origin)-box['centre_xyz'])
    d = rotation.T@(np.asarray(target)-origin)
    half = np.asarray(box['size_xyz'])/2
    low, high = 0., 1.
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if abs(p[axis]) > half[axis]: return None
        else:
            a, b = (-half[axis]-p[axis])/d[axis], (half[axis]-p[axis])/d[axis]
            low, high = max(low, min(a, b)), min(high, max(a, b))
        if low > high: return None
    return float(low)


def marker_centres_occluded(spec, camera_origin):
    boxes = spec['geometry']['wall_boxes']
    panels = [b for b in boxes if b['wall_id'].startswith('marker_')]
    walls = [b for b in boxes if not b['wall_id'].startswith('marker_')]
    return bool(len(panels) == 2 and all(any(
        (entry := ray_box_entry(camera_origin, panel['centre_xyz'], wall)) is not None and entry < 1.-1e-6
        for wall in walls) for panel in panels))


def all_window(mask, length):
    mask = np.asarray(mask, dtype=bool)
    sums = np.r_[0, np.cumsum(mask)]
    result = np.zeros(len(mask), dtype=bool)
    if len(mask) >= length: result[length-1:] = sums[length:]-sums[:-length] == length
    return result


def reduce_whole_task(raw, start_index, decisions, *, terminal, stop_reason, sensor_fault, initial_marker_occluded):
    """A home claim and true physical home arrival are distinct outcome variables.

    Primary physical task success does not require a controller home declaration.
    It does require hidden-marker discovery after departure, contact/sensor-free
    execution, actual return and an actual zero-command release. A controller may
    physically return but fail to recognize it; that failure is reported separately.
    """
    times = np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    count = len(times)
    if not 0 <= start_index < count or not decisions:
        raise ValueError('actual settled start and recorded policy decisions required')
    distance = np.linalg.norm(raw['base_pose_world'][:, :2]-raw['base_pose_world'][start_index, :2], axis=1)
    discoveries = [d for d in decisions if d['controller']['marker']['newly_discovered']]
    found_ns = discoveries[0]['decision_ns'] if discoveries else None
    initial_hidden_rgb = len(decisions) >= 4 and not any(d['controller']['marker']['detections'] for d in decisions[:4])
    left_home = bool(found_ns is not None and np.any((times >= times[start_index]) & (times < found_ns) & (distance >= .70)))
    at_home = distance <= .35
    pose = raw['base_pose_world']
    # Unit xyzw quaternion, using direct component relations independently of
    # the controller's integrated attitude. Values are evaluation-only.
    q = pose[:, 3:]
    roll = np.arctan2(2*(q[:, 3]*q[:, 0]+q[:, 1]*q[:, 2]), 1-2*(q[:, 0]**2+q[:, 1]**2))
    pitch = np.arcsin(np.clip(2*(q[:, 3]*q[:, 1]-q[:, 2]*q[:, 0]), -1., 1.))
    stable = (pose[:, 2] >= .20) & (np.abs(roll) <= .5) & (np.abs(pitch) <= .5)
    quiet = (np.linalg.norm(raw['base_twist_world'][:, :2], axis=1) <= .10) & (np.abs(raw['base_twist_world'][:, 5]) <= .25)
    zero = np.all(raw['requested_command'] == 0., axis=1)
    after_detection = times >= found_ns if found_ns is not None else np.zeros(count, dtype=bool)
    returned = all_window(at_home & stable & after_detection, 250) & all_window(zero, 250) & all_window(quiet, 100)
    valid = np.flatnonzero(returned)
    terminal_index = sensor_fault['pre_sample_index'] if sensor_fault is not None else decisions[-1]['pre_sample_index']
    release = bool(count-terminal_index-1 == 250 and np.all(zero[terminal_index+1:]))
    checks = {'initially_hidden_rgb': initial_hidden_rgb, 'initial_marker_centres_occluded': bool(initial_marker_occluded),
              'actual_marker_discovery': found_ns is not None, 'departed_home_before_discovery': left_home,
              'actual_terminal_home_return': bool(returned[-1]), 'actual_zero_release': release,
              'no_contact': not bool(raw['physics_contact'].any()), 'no_native_stop': stop_reason is None,
              'no_sensor_fault': sensor_fault is None}
    home_claim = terminal.startswith('HOME_CANDIDATE')
    return {'physical_task_success': all(checks.values()), 'checks': checks, 'controller_terminal': terminal,
            'controller_home_claim': home_claim, 'false_home_claim': home_claim and not bool(returned[-1]),
            'physical_return_without_home_claim': bool(returned[-1]) and not home_claim,
            'first_marker_discovery_ns': found_ns, 'first_stable_return_ns': int(times[valid[0]]) if len(valid) else None,
            'final_distance_from_start_m': float(distance[-1]), 'maximum_distance_from_start_m': float(distance[start_index:].max()),
            'base_path_length_m': float(np.linalg.norm(np.diff(pose[start_index:, :2], axis=0), axis=1).sum()),
            'elapsed_seconds': float(raw['timestamp_s'][-1]-raw['timestamp_s'][start_index]),
            'stop_reason': stop_reason, 'sensor_fault': sensor_fault,
            'trusted_graph_edges': 0, 'scope': 'fresh whole-task development endpoints only; no independent-maze or hardware qualification'}

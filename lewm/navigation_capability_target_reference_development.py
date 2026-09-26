"""Fixed world mission targets, independent of the controller's pose estimate."""
import numpy as np


def nominal_targets(episode):
    x, y, yaw = episode['home_se2_world']
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    origin = np.array([x, y])
    return {phase: origin + rotation @ np.asarray(episode['mission'][key])
            for phase, key in [('OUTBOUND', 'goal_initial_body_xy_m'),
                               ('RETURN', 'return_initial_body_xy_m')]}


def world_target(episode, phase):
    if phase == 'OUTBOUND':
        return np.asarray(episode['beacon_xy_world'], dtype=float)
    if phase == 'RETURN':
        return np.asarray(episode['home_se2_world'][:2], dtype=float)
    raise ValueError('Unknown mission phase')


def fixed_world_arrivals(episode, mission, frames, trace, requests):
    """Same 40-mm/1-s/50-mm-s/zero-command criteria, one world reference.

    The initial-body readout is not an additional geometry criterion. The
    controller's recorded arrival events only select the dwell interval.
    """
    lookup = {r['frame']: r for r in frames}
    commands = {r['simulator_ns']: r['requested_command'] for r in requests}
    physics = trace['base_pose_world']
    rows = []
    for arrival in mission[-1]['arrivals']:
        end = arrival['frame']; start = end - 10
        assert start >= 0 and all(i in lookup for i in range(start, end + 1))
        a, b = lookup[start], lookup[end]
        assert b['measured_ns'] - a['measured_ns'] == 1_000_000_000
        distance = np.linalg.norm(physics[a['physical_sample_index']:b['physical_sample_index'] + 1, :2]
                                  - world_target(episode, arrival['phase']), axis=1)
        boundaries = physics[[lookup[i]['physical_sample_index'] for i in range(start, end + 1)], :3]
        speed = np.linalg.norm(np.diff(boundaries, axis=0), axis=1) / .1
        zero = all(t in commands and np.array_equal(commands[t], [0., 0., 0.])
                   for t in range(a['measured_ns'], b['measured_ns'], 20_000_000))
        passed = bool(np.all(distance <= .04) and np.all(speed <= .05) and zero)
        rows.append(dict(phase=arrival['phase'], frame=end,
            target_world_xy_m=world_target(episode, arrival['phase']).tolist(),
            physical_radius_m=.04, dwell_seconds=1.,
            native_minimum_distance_m=float(distance.min()), native_maximum_distance_m=float(distance.max()),
            native_final_distance_m=float(distance[-1]), native_maximum_100ms_speed_m_s=float(speed.max()),
            all_requested_intervals_zero=zero, physical_distance_passed=bool(np.all(distance <= .04)),
            measured_motion_quiet=bool(np.all(speed <= .05)), arrival_checks_passed=passed, passed=passed,
            reference='registered fixed world beacon/home'))
    return rows

"""Fixed world mission targets, independent of the controller's pose estimate."""
import numpy as np


def settled_task_cues(episode, position_world, rotation_world_from_body):
    """Task-layer-only, one-time cues in the tracker's initial-body XY plane.

    The tracker estimates full initial-body-relative translation, then navigation
    uses its first two coordinates. Lift the XY-only task targets to the initial
    body-origin height and use that same projection, not a yaw-only rotation.
    Neither the pose nor the transform is included in the returned public cues.
    """
    p=np.asarray(position_world,dtype=float)
    R=np.asarray(rotation_world_from_body,dtype=float)
    if p.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(R).all():
        raise ValueError('Finite settled rigid pose required by task layer')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0) or np.linalg.det(R)<.999999:
        raise ValueError('Proper settled rotation required')
    A=R[:2,:2].T
    if np.linalg.cond(A)>2:
        raise ValueError('Task XY projection ill-conditioned at start')
    return dict(goal_initial_body_xy_m=(A@(world_target(episode,'OUTBOUND')-p[:2])).tolist(),
        return_initial_body_xy_m=(A@(world_target(episode,'RETURN')-p[:2])).tolist(),
        require_return_after_goal=True)


def cue_world_xy(cue, position_world, rotation_world_from_body):
    """Evaluator inverse of the declared planar projection."""
    return np.asarray(position_world)[:2]+np.linalg.solve(
        np.asarray(rotation_world_from_body)[:2,:2].T,np.asarray(cue))


def install_task_cues(controller,cues):
    """Supply both cues through the unchanged mission class before observations.

    The legacy runtime convenience constructor hard-codes a zero home. Replace
    its not-yet-started mission with the same class and limits, using both public
    task cues. No controller/tracker method or selection rule is replaced.
    """
    old=controller.mission
    if old.frame!=-1 or controller.mission_rows:
        raise ValueError('Task cues may only be installed once before mission start')
    if getattr(controller,'_task_cues_installed',False):
        raise ValueError('Task transform is one-time only')
    controller.mission=type(old)(cues,navigation_ticks=old.navigation_ticks,
        arrival_radius_m=old.arrival_radius_m)
    np.testing.assert_array_equal(controller.goal,cues['goal_initial_body_xy_m'])
    controller._task_cues_installed=True


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

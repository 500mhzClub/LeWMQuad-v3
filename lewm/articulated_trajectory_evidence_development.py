"""Action-conditioned nominal baseline and explicit time-indexed evidence.

The command baseline is NOT a learned gait, measured velocity, physical motion
bound or stopping model. It is an intentionally falsifiable baseline for fresh
execution validation. Alternative predictors can supply the same pose/q arrays.
"""
import hashlib

import numpy as np

from lewm.causal_sensor_state import SensorContractError, _identity
from lewm.factored_configuration_evidence_development import query_factored_configuration
from lewm.observed_turn_region_development import gravity_basis
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import validate_policy_packet


def command_baseline_trajectory(owner, policy, requested_commands, *, now_ns):
    """Slew-limited ideal-command SE(2) + measured joint-velocity persistence.

    Preserve measured tilt; translate/yaw in the observed gravity tangent.
    Joint velocities are extrapolated, never called a learned gait prediction.
    A zero command does not imply the physical robot has stopped. A stopping
    sequence must be explicitly supplied and independently validated later.
    """
    owner.navigation_snapshot(now_ns=now_ns); validate_policy_packet(policy)
    state = policy['sensor_state']; commands = np.asarray(requested_commands, float)
    if (type(now_ns) is not int or state['decision_ns'] != now_ns or policy['image']['measured_ns'] != now_ns
            or _identity(state['identity']) != owner._memory._rays.identity
            or commands.ndim != 2 or commands.shape[1:] != (3,) or not 1 <= len(commands) <= 50
            or not np.isfinite(commands).all() or np.any(np.abs(commands) > [.3, 0., .5])):
        raise SensorContractError('fresh same-episode packet and bounded explicit 100-ms command sequence required')
    joints, control = state['sensed']['joints'], state['control']['applied_command']
    for source in (joints, control):
        if (source['measured_ns'][-1] != now_ns or not np.asarray(source['valid']).all()):
            raise SensorContractError('complete current joint and applied-command histories required')
    q = np.asarray(joints['values'][-1, :12], float)
    dq = np.asarray(joints['values'][-1, 12:], float)
    if not np.array_equal(q, owner._memory._joints): raise SensorContractError('trajectory anchor differs from owner joints')
    prior = np.asarray(control['values'][-1], float).copy()
    if np.any(np.abs(prior) > np.array([.3, 0., .5])+1e-7): raise SensorContractError('acknowledged command outside baseline domain')
    up = owner._memory._rays.latest_frame['evidence']['up']; basis = gravity_basis(up)
    positions = [np.zeros(3)]; rotations = [np.eye(3)]; postures = [q.copy()]; applied = []
    heading = 0.
    for index, command in enumerate(commands):
        prior = prior+np.clip(command-prior, [-.25, 0., -.35], [.25, 0., .35])
        applied.append(prior.copy())
        turn = .1*prior[2]; midpoint = heading+turn/2
        distance = .1*prior[0]*np.sinc(turn/(2*np.pi))
        positions.append(positions[-1]+basis@np.array([distance*np.cos(midpoint), distance*np.sin(midpoint), 0.]))
        heading += turn
        rotations.append(rotation_increment(up*heading))
        postures.append(q+dq*((index+1)*.1))
    bound_inputs = b''.join(np.asarray(source[field]).tobytes() for source in (joints, control)
        for field in ('values', 'valid', 'measured_ns', 'available_ns'))
    return dict(model_id='ideal_command_se2_joint_velocity_persistence.v1', identity=owner._memory._rays.identity,
        anchor_ns=now_ns, offsets_ns=(np.arange(len(commands)+1, dtype=np.int64)*100_000_000).tolist(),
        positions_current_body_m=np.asarray(positions).tolist(), rotations_current_body=np.asarray(rotations).tolist(),
        joints_rad=np.asarray(postures).tolist(), requested_commands=commands.tolist(), expected_applied_commands=np.asarray(applied).tolist(),
        joint_command_history_sha256=hashlib.sha256(bound_inputs).hexdigest(), up_current_body=np.asarray(up).tolist(),
        measured_motion_prediction=False, learned_prediction=False, stopping_model_validated=False,
        execution_error_validated=False, navigation_action_permitted=False)


def evaluate_trajectory(owner, trajectory, *, point_errors_m, physical_point_speed_bounds_m_s,
                        now_ns, backend='compiled'):
    """Check endpoint balls covering each interval under EXPLICIT assumptions.

    If every actual physical point has speed <= V in the fixed current-body
    reference, it lies within V*dt/2 of its nearest endpoint. That endpoint's
    prediction error e gives radius e+V*dt/2 about its predicted position.
    This is a conditional enclosure argument, not calibration of e or V.
    Per-node radius is the maximum allowance of its adjoining intervals.
    The whole trajectory, including the caller's stopping tail, must fit the
    evidence horizon. No successful endpoint query grants action permission.
    """
    owner.navigation_snapshot(now_ns=now_ns)
    offsets = np.asarray(trajectory['offsets_ns'])
    p = np.asarray(trajectory['positions_current_body_m'], float)
    R = np.asarray(trajectory['rotations_current_body'], float)
    q = np.asarray(trajectory['joints_rad'], float)
    e, speed = np.asarray(point_errors_m, float), np.asarray(physical_point_speed_bounds_m_s, float)
    n = len(offsets) if offsets.ndim == 1 else 0
    if (type(now_ns) is not int or type(trajectory['anchor_ns']) is not int or trajectory['anchor_ns'] != now_ns
            or _identity(trajectory['identity']) != owner._memory._rays.identity
            or not isinstance(trajectory['model_id'], str) or not trajectory['model_id']
            or not 2 <= n <= 51 or offsets.dtype.kind not in 'iu' or offsets[0] != 0
            or any(int(b) <= int(a) for a,b in zip(offsets[:-1], offsets[1:]))
            or int(offsets[-1]) > 5_000_000_000
            or p.shape != (n,3) or R.shape != (n,3,3) or q.shape != (n,12)
            or e.shape != (n,) or speed.shape != (n-1,) or np.any(e < 0) or np.any(speed < 0)
            or not all(np.isfinite(v).all() for v in (p,R,q,e,speed))
            or not np.array_equal(p[0], np.zeros(3)) or not np.array_equal(R[0], np.eye(3))
            or not np.array_equal(q[0], owner._memory._joints)):
        raise SensorContractError('finite anchored trajectory and explicit endpoint-error/physical-speed assumptions required')
    if (not np.allclose(R.transpose(0,2,1)@R, np.eye(3), atol=1e-12, rtol=0)
            or not np.allclose(np.linalg.det(R), 1., atol=1e-12, rtol=0)):
        raise SensorContractError('proper predicted rotations required')
    half_interval = speed*np.diff(offsets).astype(float)*.5e-9
    radius = e.copy()
    radius[:-1] += half_interval
    radius[1:] = np.maximum(radius[1:], e[1:]+half_interval)
    if not np.isfinite(radius).all(): raise SensorContractError('representable intersample enclosure required')
    through = now_ns+int(offsets[-1])
    queries = [query_factored_configuration(owner,p[i],R[i],q[i],float(radius[i]),now_ns=now_ns,
        through_ns=through,backend=backend) for i in range(n)]
    return dict(model_id=trajectory['model_id'], anchor_ns=now_ns, through_ns=through,
        offsets_ns=offsets.tolist(), endpoint_prediction_error_m=e.tolist(),
        supplied_physical_point_speed_bounds_m_s=speed.tolist(),
        intersample_expanded_point_errors_m=radius.tolist(), configuration_queries=queries,
        all_nodes_conditionally_nonfloor_clear=all(r['all_primitives_conditionally_nonfloor_clear'] for r in queries),
        enclosure_method='nearest_endpoint_physical_point_speed_ball',
        motion_and_error_assumptions_validated=False, ground_support_permission=False,
        stopping_model_validated=False, continuous_swept_volume_established=False,
        navigation_action_permitted=False)

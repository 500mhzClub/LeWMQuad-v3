"""Independent evaluator-only motion/depth metrics; never a controller input."""
import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth


def moving_depth_check(native, boxes, transform):
    reference = expected_optical_depth(boxes, transform, floor_z_m=-.005)
    expected = reference['expected_depth_m']
    # Keep the same pixel-centre/interior/range/tolerance semantics as the
    # verified interface, without marker-specific visibility population tests.
    eligible = reference['surface_interior'] & (expected > .22) & (expected < 4.98)
    actual = native[np.ix_(reference['rows'], reference['columns'])]
    error = np.abs(actual[eligible]-expected[eligible])
    finite = np.isfinite(error).all()
    return {'eligible_rays': int(eligible.sum()),
        'maximum_error_m': float(error.max()) if len(error) and finite else None,
        'passes': bool(len(error) >= 1000 and finite and error.max() <= .005),
        'reference': 'actual_verified_visual_surfaces_not_collision_floor'}


def reduce_motion(raw, cameras, observations):
    poses = raw['base_pose_world']
    first = poses[cameras[0]['physical_sample_index']]
    initial_rotation = rotation_xyzw(first[3:])
    previous = None
    rows = []
    for camera, record in zip(cameras, observations, strict=True):
        observed = record['observer']
        if observed.get('status') == 'NON_DECISION_TERMINAL_CAPTURE': continue
        pose = poses[camera['physical_sample_index']]
        rotation = rotation_xyzw(pose[3:])
        item = {'observation_index': record['observation_index'], 'measured_ns': observed['measured_ns'],
                'position_error_m': None, 'step_error_m': None, 'observable_projection_error_m': None}
        true_position = initial_rotation.T@(pose[:3]-first[:3])
        if observed['position_initial_body_m'] is not None:
            item['position_error_m'] = float(np.linalg.norm(np.asarray(observed['position_initial_body_m'])-true_position))
        if previous is not None:
            prev_pose, prev_observed = previous
            delta = rotation_xyzw(prev_pose[3:]).T@(pose[:3]-prev_pose[:3])
            motion = observed['motion']
            if motion['translation_previous_body_m'] is not None:
                item['step_error_m'] = float(np.linalg.norm(np.asarray(motion['translation_previous_body_m'])-delta))
            if motion['observable_projection_previous_body_m'] is not None:
                weak = np.asarray(motion['weak_directions_previous_body']).reshape(-1, 3)
                projector = np.eye(3)-weak.T@weak
                item['observable_projection_error_m'] = float(np.linalg.norm(
                    np.asarray(motion['observable_projection_previous_body_m'])-projector@delta))
            item.update(registration_status=motion['status'], rank=motion['rank'])
        previous = (pose.copy(), observed)
        rows.append(item)
    errors = [r['step_error_m'] for r in rows[1:] if r['step_error_m'] is not None]
    ratio = len(errors)/max(1, len(rows)-1)
    final = rows[-1]['position_error_m'] if rows else None
    return {'frames': len(rows), 'intervals': max(0, len(rows)-1), 'fully_observed_intervals': len(errors),
        'fully_observed_fraction': ratio, 'maximum_step_error_m': max(errors) if errors else None,
        'final_position_error_m': final,
        'passes_declared_moving_state_check': bool(ratio >= .9 and errors and max(errors) <= .01
                                                   and final is not None and final <= .05),
        'rows': rows, 'scope': 'development moving-state evidence, not navigation or hardware qualification'}

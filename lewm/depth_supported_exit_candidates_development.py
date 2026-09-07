"""Appearance-independent measured floor extensions, not traversability.

RGB still supplies tracking, marker and appearance memory. These proposals use
declared optical depth and the current observed floor plane, never a color label,
world geometry or extrapolated ray/plane intersection beyond a depth return.
"""
from dataclasses import asdict
import hashlib

import numpy as np

from lewm.causal_depth_observation_development import body_points
from lewm.causal_ground_plane_development import URDF_SHA256
from lewm.causal_sensor_state import SensorContractError
from lewm.rgb_exit_candidates_development import candidates_from_floor_points
from lewm.uncertain_ray_memory_development import depth_evidence


def observe_depth_exit_candidates(policy, depth, ground_state, *, now_ns, observation_id):
    cloud = body_points(depth, policy, now_ns=now_ns, stride=8)
    if (ground_state['decision_ns'] != now_ns or ground_state['robot_geometry_sha256'] != URDF_SHA256
            or ground_state['ground_plane_qualified'] is not False
            or ground_state.get('estimator_mode') != 'current_observed_depth_floor'
            or observation_id != hashlib.sha256(policy['image']['rgb'].tobytes()).hexdigest()):
        raise SensorContractError('current bound image and observed depth-floor state required')
    normal = np.asarray(ground_state['up_current_body'], dtype=float)
    height = ground_state['body_origin_height_m']
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(np.linalg.norm(normal)-1.) > 1e-8 or isinstance(height, bool)
            or not np.isfinite(height) or not .1 <= height <= .6):
        raise SensorContractError('finite unit floor normal and observed body height required')
    evidence = depth_evidence(depth['depth_m'], depth['valid'], normal)
    points = cloud['points_body_m']
    support = evidence['ground'][np.ix_(cloud['rows'], cloud['columns'])]
    # Local planarity alone would accept an elevated tabletop. Require the
    # same measured floor plane, retaining every invalid ray as unknown.
    observed = cloud['valid'] & support & (np.abs(points@normal+height) <= .01)
    result = candidates_from_floor_points(points, observed, timestamp_ns=now_ns,
                                           observation_id=observation_id)
    return result | dict(decision_ns=now_ns, candidate_rows=[asdict(c) for c in result['candidates']],
        proposal_source='MEASURED_DEPTH_FLOOR_EXTENSION', color_segmentation_used=False,
        measured_floor_points=int(observed.sum()),
        depth_sha256=hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
        body_ground_estimator=ground_state['estimator_mode'],
        scope='sampled measured floor extensions only; holes/occlusions not traversability or closed-branch evidence')

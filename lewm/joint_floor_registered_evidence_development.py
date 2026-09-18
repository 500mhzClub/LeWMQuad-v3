"""Causal common-plane registration; original visual witnesses remain independently validated."""
from copy import deepcopy
import hashlib
import numpy as np

from lewm.causal_sensor_state import _identity, _ns
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, validate_depth
from lewm.auxiliary_downward45_depth_observation_development import validate_depth as validate_auxiliary
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.joint_sensor_anchored_goal_development import current_supported_rgbd_pose as current_joint_pose
from lewm.floor_pose_registration_development import (
    measured_candidates, register_pose, unit)
from lewm.joint_measured_floor_plane_development import fit_joint_plane, validate_joint_plane

SCHEMA = 'joint_floor_registered_pose_evidence_development.v1'


def depth_hash(depth):
    return hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()


def original_pose_validation_fields(pose):
    if pose['mode'] == 'gyro_rgbd_refit':
        return dict(original_joint_witness_validated_separately=False,
            original_gyro_rgbd_witness_validated_separately=True)
    return dict(original_joint_witness_validated_separately=True)


def current_joint_floor_registered_pose(evidence, *, identity, now_ns):
    """Check witness composition; fresh raw replay still owns pixel reconstruction."""
    _ns(now_ns, 'registered pose decision')
    if (evidence['schema'] != SCHEMA or _identity(evidence['identity']) != _identity(identity)
            or evidence['decision_ns'] != now_ns or evidence['status'] != 'CURRENT_FLOOR_REGISTERED_POSE'):
        raise ValueError('current separately typed joint-floor-registered evidence required')
    p, R, raw_pose = current_joint_pose(evidence['original_visual_evidence'], identity=identity, now_ns=now_ns)
    receipt = evidence['floor_registration']; pose = evidence['current_pose']
    if (receipt['frame'] != raw_pose['frame'] or receipt['measured_ns'] != now_ns
            or receipt['rgb_sha256'] != raw_pose['rgb_sha256']
            or receipt['primary_depth_sha256'] != raw_pose['depth_sha256']
            or receipt['reference']['frame'] != 0
            or receipt['reference']['measured_ns'] != 1_500_000_000
            or receipt['reference']['initial_specific_force_only'] is not True):
        raise ValueError('same-image current registration and frozen initial reference required')
    initial_up = unit(receipt['reference']['initial_up_body'])
    for witness in (receipt, receipt['reference']):
        for key in ('rgb_sha256', 'primary_depth_sha256', 'auxiliary_depth_sha256'):
            value = witness[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('explicit paired-camera registration binding required')
        validate_joint_plane(witness['joint_plane'], R.T@initial_up if witness is receipt else initial_up)
    reference = receipt['reference']
    unit(reference['initial_up_body'])
    if raw_pose['frame'] == 0:
        for key in ('rgb_sha256', 'primary_depth_sha256', 'auxiliary_depth_sha256',
                    'joint_plane'):
            if receipt[key] != reference[key]: raise ValueError('initial paired floor reference changed')
    correction = register_pose(p, R, reference['joint_plane'], receipt['joint_plane'])
    if correction != receipt['correction']:
        raise ValueError('registered pose must exactly reconstruct from separate visual and floor witnesses')
    expected = deepcopy(raw_pose) | dict(mode='joint_floor_registered_'+raw_pose['mode'],
        position_initial_body_m=correction['position_initial_body_m'],
        rotation_initial_body_from_current_body=correction['rotation_initial_body_from_current_body'],
        floor_registration_used=True, **original_pose_validation_fields(raw_pose))
    if pose != expected: raise ValueError('current pose differs from registered witness composition')
    return (np.asarray(pose['position_initial_body_m'], float),
        np.asarray(pose['rotation_initial_body_from_current_body'], float), pose)


class JointFloorRegistration:
    def __init__(self, *, identity=(0, 0, 0)):
        self.identity = tuple(identity)
        self.reference = None
        self.frame = -1
        self.failed = False

    def observe(self, policy, primary, auxiliary, raw, *, now_ns):
        if self.failed: raise ValueError('floor registration failure latched')
        try:
            validate_depth(primary, policy, now_ns=now_ns)
            validate_auxiliary(auxiliary, policy, now_ns=now_ns)
            p, R, pose = current_joint_pose(raw, identity=self.identity, now_ns=now_ns)
            if (pose['frame'] != self.frame+1 or now_ns != 1_500_000_000+pose['frame']*100_000_000
                    or primary['measured_ns'] != now_ns or auxiliary['measured_ns'] != now_ns
                    or pose['depth_sha256'] != depth_hash(primary)
                    or pose['rgb_sha256'] != primary['rgb_sha256']
                    or _identity(primary['identity']) != _identity(self.identity)):
                raise ValueError('uninterrupted current paired depth and visual pose required')
            if self.reference is None:
                force = policy['sensor_state']['sensed']['specific_force']
                command = policy['sensor_state']['control']['applied_command']
                if (not force['valid'].all() or not command['valid'].all()
                        or np.any(np.abs(command['values']) > 1e-8)):
                    raise ValueError('quiet initial public force history required')
                up = force['values'].mean(0); magnitude = np.linalg.norm(up)
                if not 8 <= magnitude <= 12: raise ValueError('initial gravity magnitude inconsistent')
                up = up/magnitude
            else:
                up = np.asarray(self.reference['initial_up_body'])
            clouds = []
            for depth, E in ((primary, np.asarray(BODY_FROM_OPTICAL)), (auxiliary, body_from_optical())):
                points, _ = measured_candidates(depth['depth_m'], depth['valid'], E, R.T@up)
                clouds.append(points)
            joint = fit_joint_plane(*clouds, R.T@up)
            validate_joint_plane(joint, R.T@up)
            witness = dict(frame=pose['frame'], measured_ns=now_ns, rgb_sha256=pose['rgb_sha256'],
                primary_depth_sha256=depth_hash(primary), auxiliary_depth_sha256=depth_hash(auxiliary),
                joint_plane=joint)
            reference = self.reference
            if reference is None:
                reference = deepcopy(witness) | dict(initial_up_body=up.tolist(), initial_specific_force_only=True)
            correction = register_pose(p, R, reference['joint_plane'], joint)
            corrected_pose = deepcopy(pose) | dict(mode='joint_floor_registered_'+pose['mode'],
                position_initial_body_m=correction['position_initial_body_m'],
                rotation_initial_body_from_current_body=correction['rotation_initial_body_from_current_body'],
                floor_registration_used=True, **original_pose_validation_fields(pose))
            result = dict(schema=SCHEMA, identity=self.identity, decision_ns=now_ns,
                status='CURRENT_FLOOR_REGISTERED_POSE', original_visual_evidence=deepcopy(raw),
                current_pose=corrected_pose, floor_registration=witness | dict(reference=deepcopy(reference),
                    correction=correction), native_pose_used=False, command_integration_used=False,
                historical_map_rewritten=False, navigation_qualified=False)
            current_joint_floor_registered_pose(result, identity=self.identity, now_ns=now_ns)
            self.reference = deepcopy(reference); self.frame = pose['frame']
            return result
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
            self.failed = True
            raise

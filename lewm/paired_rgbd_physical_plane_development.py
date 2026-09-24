"""Joint physical gap/pose/plane perturbations, never a clearance consumer.

One named raw error affects registration, inertial state and measured plane
together. The pose-only/floor-only sum deliberately drops those cross terms and
is reported solely as an incorrect-independence comparator. No error coverage,
contact permission or continuous swept volume follows from finite differences.
"""
from copy import deepcopy

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.raw_complementary_rgbd_sensitivity_development import RawComplementaryRGBDSensitivity


def minimum_gaps(geometry, joints, anchor, normal, observation_R, observation_p):
    a, n = (np.asarray(anchor)-observation_p)@observation_R, np.asarray(normal)@observation_R
    return np.asarray([s['lower'][0]-n@a for s in geometry.supports(joints, n[None])['shapes']])


class PairedRGBDPhysicalPlane:
    def __init__(self, observer):
        if not isinstance(observer, RawComplementaryRGBDSensitivity):
            raise SensorContractError('actual complementary RGBD paired estimator required')
        self.observer = observer; self.retained = {}

    def retain(self, label):
        observer = self.observer
        if (observer.failed or observer.current is None or type(label) is not str or not label
                or label in self.retained or len(self.retained) >= 64):
            raise SensorContractError('valid current paired states and a new bounded label required')
        frames = []
        for packet in observer.current_packets:
            prepared = PreparedFloorFrame(packet['depth'], packet['valid'], packet['up'])
            frames.append(dict(packet=deepcopy(packet), prepared=prepared,
                               hypothesis=MeasuredPlaneHypothesis.from_frame(prepared)))
        self.retained[label] = dict(measured_ns=observer.current['measured_ns'], frames=frames)

    def query(self, label, geometry, joints, position, rotation, *, now_ns,
              normal_error, up_error, plane_offset_error):
        observer = self.observer
        if (observer.failed or observer.current is None or now_ns != observer.current['measured_ns']
                or label not in self.retained or not isinstance(geometry, ArticulatedCollisionGeometry)):
            raise SensorContractError('current paired observer and retained observed planes required')
        q, p, R = [np.asarray(v, float) for v in (joints, position, rotation)]
        if (q.shape != (12,) or p.shape != (3,) or R.shape != (3, 3) or not all(np.isfinite(x).all() for x in (q, p, R))
                or not np.allclose(R.T@R, np.eye(3), atol=1e-12, rtol=0) or abs(np.linalg.det(R)-1) > 1e-12):
            raise SensorContractError('finite proper explicit configuration required')
        ids = tuple(s['shape_id'] for s in geometry.supports(q, np.eye(3))['shapes'])
        stored = self.retained[label]; planes = []; transforms = []; covered = []; cells = []
        for current, frame in zip(observer.current_packets, stored['frames'], strict=True):
            old, prepared = frame['packet'], frame['prepared']
            obs_R = old['rotation'].T@current['rotation']@R
            obs_p = old['rotation'].T@(current['position']+current['rotation']@p-old['position'])
            transforms.append((obs_R, obs_p))
            cell = frame['hypothesis'].cell_for(prepared); cells.append(cell)
            if cell is None:
                planes.append(None); covered.append(np.zeros(len(ids), bool)); continue
            floor = prepared.query(geometry, q, cell, rotation_observation_from_body=obs_R,
                translation_observation_from_body=obs_p, point_error_by_shape={sid: 0. for sid in ids},
                normal_error=normal_error, up_error=up_error, plane_offset_error=plane_offset_error, floor_backend='cached')
            planes.append((floor['plane_anchor_observation_m'], floor['plane_normal_observation']))
            covered.append(np.asarray([floor['floor_coverage'][sid] for sid in ids], bool))
        values = [[], [], []]
        for index, plane in enumerate(planes):
            for rows, pl, transform in zip(values, (plane, planes[0], plane),
                    (transforms[index], transforms[index], transforms[0]), strict=True):
                rows.append(np.full(len(ids), np.nan) if pl is None else minimum_gaps(geometry, q, *pl, *transform))
        factors = [(np.asarray(v)[1::2]-np.asarray(v)[2::2]).T/(2*observer.step) for v in values]
        observed = np.asarray(covered).all(axis=0)
        nominal = np.asarray(values[0][0])
        seed_changed = [i for i, cell in enumerate(cells) if cell != cells[0]]
        return dict(shape_ids=ids, nominal_minimum_gap_m=nominal,
            all_perturbed_footprints_observed=observed, per_member_floor_coverage=np.asarray(covered),
            joint_gap_factor_m=factors[0], pose_only_gap_factor_m=factors[1], floor_only_gap_factor_m=factors[2],
            joint_gap_variance_m2=np.sum(factors[0]**2, axis=1),
            incorrect_independent_gap_variance_m2=np.sum(factors[1]**2+factors[2]**2, axis=1),
            gap_midpoint_remainder_m=np.abs((np.asarray(values[0])[1::2]+np.asarray(values[0])[2::2]).T/2-nominal[:, None]),
            hypothesis_cells_rc=cells, changed_plane_seed_members=seed_changed,
            categorical_estimator_change_seen=bool(observer.branch_changes),
            current_measured_ns=now_ns, stored_measured_ns=stored['measured_ns'],
            source_ids=observer.source_ids, difference_step=observer.step,
            finite_pairs_are_continuous_error_bounds=False, uncertainty_model_calibrated=False,
            linearization_validated=False, correspondence_identity_fully_observed=False,
            queried_joint_posture_is_held_fixed=True, nonfloor_clearance_queried=False,
            navigation_action_permitted=False, foot_contact_permitted=False, future_gait_qualified=False)

"""Factored physical configuration evidence from one existing RGB-D owner.

No new integration, setup region, action prediction, contact permission or
calibrated uncertainty is introduced. Existing global transport scales remain
unchanged. An explicit configuration is a query, not an executable trajectory.
"""
import hashlib
from itertools import product

import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.factored_configuration_evidence_development import ground_summary
from lewm.measured_plane_obstacle_memory_development import MeasuredPlaneHypothesis
from lewm.primitive_beam_kernel_development import warm_primitive_beam_kernel
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_obstacle_memory_development import non_floor_box_evidence
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory
from lewm.uncertain_ray_memory_development import transport_radius

_CORNERS = np.asarray(list(product((0, 1), repeat=3)), bool)


def frame_identity(frame):
    """Bind pose/envelope and raw observation; not just a reusable timestamp."""
    evidence = frame['evidence']
    return (frame['measured_ns'], hashlib.sha256(evidence['depth'].tobytes()+evidence['valid'].tobytes()).hexdigest(),
            tuple(evidence['up']), tuple(frame['position']), tuple(np.asarray(frame['rotation']).ravel()),
            frame['position_scale_m'], frame['orientation_scale_rad'])


class RGBDPhysicalConfigurationEvidence:
    def __init__(self, owner, geometry, *, normal_error, up_error, plane_offset_error, range_error_m):
        errors = np.asarray([normal_error, up_error, plane_offset_error, range_error_m], float)
        if (not isinstance(owner, RGBDInertialRayMemory) or not isinstance(geometry, ArticulatedCollisionGeometry)
                or errors.shape != (4,) or not np.isfinite(errors).all() or np.any(errors < 0)
                or normal_error >= 1 or up_error >= 1):
            raise SensorContractError('one RGBD sensor owner, verified geometry and explicit bounded error hypotheses required')
        self.owner, self.geometry = owner, geometry
        self.errors = dict(normal_error=float(normal_error), up_error=float(up_error), plane_offset_error=float(plane_offset_error))
        self.range_error = float(range_error_m)
        self.prepared = {}; self.hypotheses = {}; self.bindings = {}
        self.last_ns = None; self.failed = False
        warm_primitive_beam_kernel()

    def _frames(self, now_ns):
        owner = self.owner; rays = owner.rays
        if (self.failed or owner.failed or rays.failed or type(now_ns) is not int or owner.last_ns != now_ns
                or rays.last_ns != now_ns or not rays.frames or rays.latest_frame is None
                or owner._reader.pending is not None or rays.fusion['measured_ns'] != now_ns
                or not rays.fusion['usable_under_declared_proxy_budget']):
            raise SensorContractError('current uninterrupted admitted RGBD owner required')
        frames = list(rays.frames)
        if frames[-1]['measured_ns'] != now_ns:
            frames.append(rays.latest_frame)
        stamps = [f['measured_ns'] for f in frames]
        if stamps != sorted(set(stamps)) or stamps[-1] != now_ns:
            raise SensorContractError('ordered unique same-owner retained observation population required')
        if (frame_identity(frames[-1]) != frame_identity(rays.latest_frame)
                or not np.array_equal(rays.position, rays.latest_frame['position'])
                or not np.array_equal(rays.rotation, rays.latest_frame['rotation'])):
            raise SensorContractError('current owner and current frame must share the exact pose')
        return frames

    def refresh(self, *, now_ns):
        """Prepare only the owner's current retained raw frames; never observe again."""
        try:
            frames = self._frames(now_ns)
            prepared, hypotheses, bindings = {}, {}, {}
            for stored in frames:
                ns = stored['measured_ns']; identity = frame_identity(stored)
                if ns in self.bindings and self.bindings[ns] != identity:
                    raise SensorContractError('retained observation/pose identity changed')
                if ns in self.prepared:
                    frame, hypothesis = self.prepared[ns], self.hypotheses[ns]
                else:
                    evidence = stored['evidence']
                    frame = PreparedFloorFrame(evidence['depth'], evidence['valid'], evidence['up'])
                    hypothesis = MeasuredPlaneHypothesis.from_frame(frame)
                if frame.depth_sha256 != identity[1]:
                    raise SensorContractError('prepared floor does not bind the original RGBD observation')
                hypothesis.cell_for(frame)
                prepared[ns], hypotheses[ns], bindings[ns] = frame, hypothesis, identity
            self.prepared, self.hypotheses, self.bindings = prepared, hypotheses, bindings
            self.last_ns = now_ns
            return dict(measured_ns=now_ns, prepared_views=len(prepared), sensor_owner_reintegrated=False,
                        initial_setup_region_used=False, navigation_action_permitted=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('RGBD physical evidence preparation unavailable') from error

    def query(self, position, rotation, joints, point_error_m, *, now_ns, backend='compiled'):
        """Whole-primitive evidence at one supplied CURRENT-BODY configuration."""
        try:
            return self._query(position, rotation, joints, point_error_m, now_ns=now_ns, backend=backend)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('RGBD physical configuration query unavailable') from error

    def _query(self, position, rotation, joints, point_error_m, *, now_ns, backend):
        frames = self._frames(now_ns)
        if (now_ns != self.last_ns or set(self.prepared) != {f['measured_ns'] for f in frames}
                or any(self.bindings[f['measured_ns']] != frame_identity(f) for f in frames)):
            raise SensorContractError('prepare exact current retained population before querying')
        offset, local_R, q = [np.asarray(v, float) for v in (position, rotation, joints)]
        if (offset.shape != (3,) or local_R.shape != (3, 3) or q.shape != (12,)
                or not all(np.isfinite(v).all() for v in (offset, local_R, q))
                or not np.allclose(local_R.T@local_R, np.eye(3), atol=1e-12, rtol=0)
                or abs(np.linalg.det(local_R)-1) > 1e-12
                or type(point_error_m) not in (int, float) or not np.isfinite(point_error_m) or point_error_m < 0
                or backend not in ('compiled', 'reference')):
            raise SensorContractError('proper supplied configuration and nonnegative explicit endpoint error required')
        rays = self.owner.rays; geometry = self.geometry
        p, R = rays.position+rays.rotation@offset, rays.rotation@local_R
        local_shapes = geometry.supports(q, local_R)['shapes']
        body_shapes = geometry.supports(q, np.eye(3))['shapes']
        ids = tuple(s['shape_id'] for s in body_shapes)
        clear_sources = [[] for _ in ids]; conflict_sources = [[] for _ in ids]
        witnesses = [[] for _ in ids]; observations = []
        for stored in frames:
            ns = stored['measured_ns']; frame = self.prepared[ns]
            cell = self.hypotheses[ns].cell_for(frame)
            obs_R = stored['rotation'].T@R
            obs_p = (p-stored['position'])@stored['rotation']
            shapes = geometry.supports(q, obs_R)['shapes']
            point_errors = {}
            for i, sid in enumerate(ids):
                corners = np.where(_CORNERS, local_shapes[i]['upper'], local_shapes[i]['lower'])+offset
                point_errors[sid] = float(point_error_m+transport_radius(corners, rays.latest_frame, stored).max())
            floor = None if cell is None else frame.query(geometry, q, cell,
                rotation_observation_from_body=obs_R, translation_observation_from_body=obs_p,
                point_error_by_shape=point_errors, floor_backend='cached' if backend == 'compiled' else 'reference', **self.errors)
            errors = np.asarray([point_errors[sid] for sid in ids])[:, None]
            low = np.asarray([s['lower'] for s in shapes])+obs_p-errors-.04
            high = np.asarray([s['upper'] for s in shapes])+obs_p+errors+.04
            plane = None if floor is None else (floor['plane_anchor_observation_m'], floor['plane_normal_observation'])
            nonfloor = non_floor_box_evidence(frame, low, high, plane=plane,
                normal_error=self.errors['normal_error'], plane_offset_error=self.errors['plane_offset_error'],
                range_error_m=self.range_error, backend=backend)
            for i, sid in enumerate(ids):
                if nonfloor['non_floor_clearance'][i]: clear_sources[i].append(ns)
                if nonfloor['non_floor_conflict'][i]: conflict_sources[i].append(ns)
                gap = None if floor is None else floor['gap_bounds']['primitives'][i]
                if gap is not None and gap['shape_id'] != sid:
                    raise SensorContractError('exact physical primitive identity required')
                witness = dict(measured_ns=ns, depth_sha256=frame.depth_sha256, hypothesis_cell_rc=cell, gap=gap,
                    floor_coverage=False if floor is None else floor['floor_coverage'][sid],
                    plane_vertex_residual_lower_m=None, plane_vertex_residual_upper_m=None)
                if floor is not None:
                    a, n = np.asarray(floor['gap_bounds']['plane_anchor_body_m']), np.asarray(floor['gap_bounds']['plane_normal_body'])
                    vertices = np.where(_CORNERS, body_shapes[i]['upper'], body_shapes[i]['lower'])
                    nominal = (vertices-a)@n
                    # Virtual AABB vertices need coordinate-support inflation,
                    # unlike physical endpoints used by the exact gap bounds.
                    ve = np.sqrt(3.)*point_error_m+transport_radius(vertices@local_R.T+offset, rays.latest_frame, stored)
                    error = (self.errors['normal_error']*np.linalg.norm(vertices-a, axis=1)
                        +self.errors['plane_offset_error']+(1+self.errors['normal_error'])*ve)
                    rounding = 1e-12+128*np.finfo(float).eps*(np.abs(nominal)+np.linalg.norm(vertices-a, axis=1)+error)
                    witness['plane_vertex_residual_lower_m'] = (nominal-error-rounding).tolist()
                    witness['plane_vertex_residual_upper_m'] = (nominal+error+rounding).tolist()
                witnesses[i].append(witness)
            observations.append(dict(measured_ns=ns, depth_sha256=frame.depth_sha256, hypothesis_cell_rc=cell,
                hypothesis_policy=self.hypotheses[ns].policy, nonfloor_conflict_primitives=int(nonfloor['non_floor_conflict'].sum()),
                nonfloor_clear_primitives=int(nonfloor['non_floor_clearance'].sum()),
                scanned_pixels=int(nonfloor['scanned_pixels'].sum()), maximum_point_error_m=float(errors.max())))
        rows = []
        for i, sid in enumerate(ids):
            ground = ground_summary(sid, witnesses[i])
            nonfloor = bool(clear_sources[i] and not conflict_sources[i])
            rows.append(dict(shape_id=sid, ground=ground, ground_witnesses=witnesses[i],
                nonfloor_clearance_sources=clear_sources[i], nonfloor_conflict_sources=conflict_sources[i],
                conditional_nonfloor_clearance=nonfloor,
                conditional_observed_separation=bool(nonfloor and ground['physical_floor_separation_observed']),
                contact_candidate_with_nonfloor_clearance=bool(nonfloor and ground['observed_foot_contact_candidate']),
                contact_permitted=False))
        return dict(measured_ns=now_ns, identity=rays.identity, primitives=rows, observation_bindings=observations,
            supplied_configuration=dict(reference='current_body', position_m=offset.tolist(), rotation=local_R.tolist(),
                joints_rad=q.tolist(), point_error_m=float(point_error_m), padding_m=.04),
            all_primitives_conditionally_nonfloor_clear=all(r['conditional_nonfloor_clearance'] for r in rows),
            all_primitives_observed_separated=all(r['conditional_observed_separation'] for r in rows),
            static_scene_assumed=True, supplied_error_scales_validated=False,
            common_ground_surface_identity_established=False, initial_setup_region_used=False,
            sensor_owner_reintegrated=False, configuration_is_execution_prediction=False,
            continuous_swept_volume_established=False, ground_support_permission=False,
            future_gait_qualified=False, navigation_action_permitted=False)

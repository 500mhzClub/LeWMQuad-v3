"""Keep original geometry and resolve only old foot ambiguity with later views."""
from copy import deepcopy
import hashlib
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.measured_floor_partition_development import FOOT_IDS
from lewm.joint_visual_floor_map_development import GRID
from lewm.frame_cached_floor_geometry_development import FloorFrameGeometry
from lewm.frame_cached_floor_map_development import (
    FrameCachedFloorMemory, FrameCachedFloorMap, FrameCachedJointFloorRoundTripController)
from lewm.later_floor_evidence_development import LaterFloorEvidence, resolve_sphere_query


class RecordingFloorGeometry(FloorFrameGeometry):
    def __init__(self, primary, auxiliary, frame, now_ns):
        super().__init__()
        self.packets = (primary, auxiliary)
        self.frame = frame; self.now_ns = now_ns; self.observations = []

    def floor_coverage(self, depth, valid, map_from_body, translation_map, floor_height, cells=GRID):
        index = len(self.observations)
        if index >= 2: raise SensorContractError('one primary and one auxiliary floor coverage per observation required')
        camera = ('primary', 'auxiliary')[index]; packet = self.packets[index]
        if (depth is not packet['depth_m'] or valid is not packet['valid']
                or packet['measured_ns'] != self.now_ns or not np.array_equal(cells, GRID)):
            raise SensorContractError('ordered original camera arrays and complete fixed floor grid required')
        result = super().floor_coverage(depth, valid, map_from_body, translation_map, floor_height, cells)
        witness = dict(camera=camera, frame=self.frame, measured_ns=self.now_ns,
            calibration_id=packet['calibration_id'],
            rgb_sha256=packet['rgb_sha256'] if camera == 'primary' else packet['primary_rgb_sha256'],
            depth_sha256=hashlib.sha256(depth.tobytes()+valid.tobytes()).hexdigest(),
            rotation_map_from_reference=np.asarray(map_from_body).tolist(),
            position_map_m=np.asarray(translation_map).tolist())
        self.observations.append(dict(witness=witness, cells=GRID[result['covered']].copy()))
        return result


class LaterResolvedFloorMemory(FrameCachedFloorMemory):
    def __init__(self, *, identity):
        super().__init__(identity=identity)
        self.later_floor_evidence = LaterFloorEvidence()

    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent=True):
        original = super().footprint(geometry, displacement_body_xy, yaw_rad, now_ns=now_ns, persistent=persistent)
        ledger = self.later_floor_evidence
        if ledger.now_ns != now_ns or ledger.frame != len(self.route)-1:
            raise SensorContractError('current complete paired floor-evidence history required')
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        R = self.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        p = self.position+self.rotation@np.r_[displacement_body_xy,0.]
        primitives = {v['shape_id']:v for v in geometry._shapes}
        supports = {v['shape_id']:v for v in geometry.supports(self.joints, R)['shapes']}
        primary = []; auxiliary = []; resolutions = []; found = set()
        for old_primary, old_auxiliary in zip(original['shapes'], original['auxiliary_shapes'], strict=True):
            key = old_primary['shape_id']
            if old_auxiliary['shape_id'] != key: raise SensorContractError('ordered paired original shapes required')
            if key not in FOOT_IDS:
                primary.append(old_primary); auxiliary.append(old_auxiliary); continue
            found.add(key)
            primitive = primitives[key]
            if primitive['kind'] != 'sphere' or float(primitive['dimensions'][0]) != .022:
                raise SensorContractError('unchanged four nominal foot spheres required')
            center = p+R@np.asarray(supports[key]['center_body_m'])
            for camera, index, old, output in (
                    ('primary', self.partition.other, old_primary, primary),
                    ('auxiliary', self.confirmed_auxiliary_partition.other, old_auxiliary, auxiliary)):
                query = {k:v for k,v in old.items() if k != 'shape_id'}
                revised, receipt = resolve_sphere_query(index, query, center, .022, ledger, now_ns=now_ns)
                output.append(dict(shape_id=key, **revised))
                resolutions.append(dict(shape_id=key, source_camera=camera, **receipt))
        if found != set(FOOT_IDS): raise SensorContractError('all four nominal foot queries required')
        return original | dict(shapes=primary, auxiliary_shapes=auxiliary,
            primary_possible_intersection=any(q['intersecting_voxels'] for q in primary),
            auxiliary_possible_intersection=any(q['intersecting_voxels'] for q in auxiliary),
            possible_intersection=any(q['intersecting_voxels'] for q in primary+auxiliary),
            original_contact_check_before_later_floor_resolution=deepcopy(original),
            later_floor_contact_resolution=resolutions,
            collision_contact_policy='later_complete_floor_evidence_nominal_feet_v1',
            ground_contact_scope='four_nominal_feet_original_or_later_observed_floor_support_unknown',
            original_partitions_and_all_return_indices_retained=True,
            historically_ambiguous_returns_resolved_only_with_later_measurements=True,
            unresolved_contacts_exempted=False, non_foot_contacts_exempted=False,
            ground_support_approved=False, unobserved_space_certified=False)


class LaterResolvedFloorMap(FrameCachedFloorMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.surface = LaterResolvedFloorMemory(identity=identity)

    def observe(self, policy, depth, evidence, *, auxiliary_depth, now_ns):
        if self.frame_geometry is not None:
            raise SensorContractError('nested floor-map observation is unavailable')
        frame = len(self.surface.route)
        context = RecordingFloorGeometry(depth, auxiliary_depth, frame, now_ns)
        self.frame_geometry = self.surface.frame_geometry = context
        try:
            receipt = self._observe_both(policy, depth, evidence, auxiliary_depth=auxiliary_depth, now_ns=now_ns)
            self.surface.later_floor_evidence.record_pair(frame, now_ns, self.map_from_initial,
                self.floor_height, context.observations)
            return receipt
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
            self.failed = self.surface.failed = True
            raise
        finally:
            self.last_cache_counts = context.counts()
            context.close()
            self.frame_geometry = self.surface.frame_geometry = None


class LaterFloorResolutionRoundTripController(FrameCachedJointFloorRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = LaterResolvedFloorMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='later_floor_resolution_round_trip_controller_v1',
            later_measured_floor_contact_resolution_enabled=True)

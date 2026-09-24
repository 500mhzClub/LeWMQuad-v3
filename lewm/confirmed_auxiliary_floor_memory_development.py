"""Retain original returns/classifications alongside current-primary confirmation."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.observed_floor_contact_development import ObservedFloorContactMemory, ObservedFloorContactMap
from lewm.measured_floor_partition_development import MeasuredFloorPartition, FOOT_IDS
from lewm.auxiliary_downward45_depth_observation_development import body_points
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.current_primary_floor_plane_development import primary_floor_plane, confirm_auxiliary_floor


def confirmed_contact_check(original, geometry, joints, R, p, partition, receipt):
    """Only four nominal auxiliary foot queries use the additional partition."""
    primitives = {s['shape_id']: s for s in geometry._shapes}
    supports = {s['shape_id']: s for s in geometry.supports(joints, R)['shapes']}
    auxiliary = []; ground = {}; other = {}
    for hit in original['auxiliary_shapes']:
        key = hit['shape_id']
        if key not in FOOT_IDS:
            auxiliary.append(deepcopy(hit)); continue
        primitive = primitives[key]
        if primitive['kind'] != 'sphere' or float(primitive['dimensions'][0]) != .022:
            raise SensorContractError('unchanged four nominal foot spheres required')
        center = p+R@np.asarray(supports[key]['center_body_m'])
        ground[key] = partition.floor.intersect_sphere(center, .022)
        other[key] = partition.other.intersect_sphere(center, .022)
        auxiliary.append(dict(shape_id=key, **deepcopy(other[key])))
    contacts = deepcopy(original['observed_ground_contacts'])
    rules = deepcopy(original['auxiliary_foot_floor_contacts'])
    if set(ground) != {r['shape_id'] for r in contacts} or set(ground) != {r['shape_id'] for r in rules}:
        raise SensorContractError('complete ordered nominal-foot evidence required')
    for contact in contacts:
        contact['auxiliary_measured_floor'] = deepcopy(ground[contact['shape_id']])
    for rule in rules:
        key = rule['shape_id']
        rule['auxiliary_floor'] = deepcopy(ground[key])
        rule['auxiliary_other_or_unknown'] = deepcopy(other[key])
        if max(ground[key]['intersecting_voxels'], other[key]['intersecting_voxels']) > rule['auxiliary_all_returns']['intersecting_voxels']:
            raise SensorContractError('confirmed classification escaped original all-return enclosure')
    auxiliary_conflict = any(r['intersecting_voxels'] for r in auxiliary)
    return original | dict(auxiliary_shapes=auxiliary, auxiliary_possible_intersection=auxiliary_conflict,
        possible_intersection=bool(original['primary_possible_intersection'] or auxiliary_conflict),
        observed_ground_contacts=contacts, auxiliary_foot_floor_contacts=rules,
        original_auxiliary_floor_contact_check=deepcopy(original),
        current_primary_floor_confirmation=deepcopy(receipt),
        collision_contact_policy='current_primary_confirmed_auxiliary_floor_v1',
        ground_contact_scope='four_nominal_feet_with_original_or_primary_confirmed_measured_floor_only',
        non_foot_contacts_exempted=False, non_floor_or_unknown_contacts_exempted=False,
        ground_support_approved=False, unobserved_space_certified=False)


class ConfirmedAuxiliaryFloorMemory(ObservedFloorContactMemory):
    def __init__(self, *, identity):
        super().__init__(identity=identity)
        self.confirmed_auxiliary_partition = MeasuredFloorPartition()
        self.primary_plane = None; self.primary_plane_ns = None
        self.confirmation_receipt = None; self.confirmation_ns = None

    def classify_current(self, policy, depth, B, floor_height, floor_cells, *, now_ns):
        super().classify_current(policy, depth, B, floor_height, floor_cells, now_ns=now_ns)
        plane = primary_floor_plane(depth['depth_m'], depth['valid'], B@self.rotation,
            B@self.position, floor_height)
        self.primary_plane = plane | {k: self.classification_receipt[k] for k in
            ('frame', 'measured_ns', 'rgb_sha256', 'depth_sha256')}
        self.primary_plane_ns = now_ns

    def observe_auxiliary(self, policy, depth, B, floor_height, floor_cells, occupied, *, now_ns):
        if self.primary_plane_ns != now_ns:
            raise SensorContractError('current primary plane evidence required before auxiliary confirmation')
        original = super().observe_auxiliary(policy, depth, B, floor_height, floor_cells, occupied, now_ns=now_ns)
        cloud = body_points(depth, policy, now_ns=now_ns, stride=4)
        Q, q = reference_pose(B@self.rotation, B@self.position)
        mask, classification = confirm_auxiliary_floor(depth['depth_m'], depth['valid'], Q, q,
            floor_height, cloud['rows'], cloud['columns'], self.primary_plane)
        if classification['original_floor_count'] != original['current_floor_returns']:
            raise SensorContractError('original auxiliary classification must reconstruct exactly')
        points = cloud['points_body_m'][cloud['valid']]@self.rotation.T+self.position
        witness = {k: original[k] for k in ('frame', 'measured_ns', 'calibration_id', 'rgb_sha256', 'depth_sha256')}
        partition = self.confirmed_auxiliary_partition
        partition.insert(points, mask[cloud['valid']], witness)
        if partition.total_returns != self.auxiliary_partition.total_returns:
            raise SensorContractError('every original auxiliary return requires one confirmed classification')
        self.confirmation_ns = now_ns
        self.confirmation_receipt = dict(**witness, primary_plane=deepcopy(self.primary_plane),
            classification=classification, total_returns=partition.total_returns,
            floor_returns=partition.floor_returns, other_returns=partition.other_returns,
            original_auxiliary_partition_retained=True, original_all_return_index_retained=True,
            pose_or_floor_grid_changed=False, classifications_use_only_contemporary_public_packets=True)
        # Keep self.auxiliary_receipt exactly original so predecessor footprint
        # receipts remain reproducible independently of the new partition.
        return original | dict(current_primary_floor_confirmation=deepcopy(self.confirmation_receipt))

    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent=True):
        if self.confirmation_ns != now_ns:
            raise SensorContractError('current auxiliary confirmation required for footprint query')
        if self.confirmed_auxiliary_partition.total_returns != self.auxiliary_partition.total_returns:
            raise SensorContractError('complete original and confirmed auxiliary populations required')
        original = super().footprint(geometry, displacement_body_xy, yaw_rad, now_ns=now_ns, persistent=persistent)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        R = self.rotation@np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        p = self.position+self.rotation@np.r_[displacement_body_xy, 0.]
        return confirmed_contact_check(original, geometry, self.joints, R, p,
            self.confirmed_auxiliary_partition, self.confirmation_receipt)


class ConfirmedAuxiliaryFloorMap(ObservedFloorContactMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.surface = ConfirmedAuxiliaryFloorMemory(identity=identity)

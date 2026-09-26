"""Initialise the shared map's scalar floor from its qualified paired plane.

The measured tracker, map orientation, subsequent mapping, clearance rules and
sensor packets are unchanged. Mesh-normal candidates alone no longer establish
the initial scalar floor height. No simulator state enters this module.
"""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import validate_depth
from lewm.auxiliary_downward45_depth_observation_development import validate_depth as validate_auxiliary
from lewm.current_plane_floor_coverage_development import current_paired_plane
from lewm.navigation_capability_startup_recovery_development import (
    RecoverableStartupMap, INITIAL_FLOOR_MISSING, StartupRecoveryRuntimeMixin,
    mapping_update as previous_mapping_update, initialize_mapping as previous_initialize_mapping)
from lewm.eligible_floor_registration_development import bind
from lewm import process_mapped_runtime_development as process


def measured_map_floor_height(plane, rotation_map_from_body, position_map):
    """Intersect the admitted measured plane with the map origin's vertical."""
    if not plane['available']:
        raise SensorContractError(INITIAL_FLOOR_MISSING)
    normal = np.asarray(rotation_map_from_body) @ np.asarray(plane['normal_body'])
    offset = float(plane['offset_body_m'] - normal @ np.asarray(position_map))
    if not np.isfinite([*normal, offset]).all() or normal[2] <= 0:
        raise SensorContractError('finite upward qualified initial floor plane required')
    return -offset/float(normal[2])


class PairedFloorStartupMap(RecoverableStartupMap):
    def update(self, policy, depth, evidence, *, auxiliary_depth, measured_ns):
        if self.failed:
            raise SensorContractError('routing-map failure latched')
        if self.floor_height is None:
            try:
                validate_depth(depth, policy, now_ns=measured_ns)
                validate_auxiliary(auxiliary_depth, policy, now_ns=measured_ns)
                p, R, pose = self._read_pose(evidence, identity=self.identity, now_ns=measured_ns)
                if self.B is None:
                    # Preserve the original quiet-gravity map convention even
                    # when later camera packets are needed to measure a plane.
                    if pose['frame'] != 0:
                        raise SensorContractError('initial measured map frame required')
                    force = policy['sensor_state']['sensed']['specific_force']
                    command = policy['sensor_state']['control']['applied_command']
                    if not force['valid'].all() or not command['valid'].all() or np.any(np.abs(command['values']) > 1e-8):
                        raise SensorContractError('quiet measured gravity initialization required')
                    up = force['values'].mean(0); magnitude = np.linalg.norm(up)
                    if not 8 <= magnitude <= 12:
                        raise SensorContractError('initial gravity magnitude inconsistent')
                    up /= magnitude; forward = np.array([1.,0.,0.])-up*up[0]
                    if np.linalg.norm(forward) < .8:
                        raise SensorContractError('initial forward/gravity frame degenerate')
                    forward /= np.linalg.norm(forward)
                    self.B = np.stack((forward, np.cross(up,forward), up))
                up_body = R.T @ self.B[2]
                up_body /= np.linalg.norm(up_body)
                plane = current_paired_plane(depth, auxiliary_depth, up_body)
                self.floor_height = measured_map_floor_height(plane, self.B@R, self.B@p)
                self.initial_floor_source = 'qualified_current_paired_plane'
                self.initial_floor_receipt = dict(frame=pose['frame'],measured_ns=measured_ns,
                    floor_height_m=self.floor_height,plane=plane,tracker_changed=False,
                    map_orientation_unchanged=True,native_state_used=False)
            except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
                self.failed = True
                raise
        return super().update(policy, depth, evidence, auxiliary_depth=auxiliary_depth, measured_ns=measured_ns)


def initialize_mapping():
    previous_initialize_mapping()
    process._mapper = PairedFloorStartupMap()


def mapping_update(packet, evidence):
    snapshot, receipt = previous_mapping_update(packet, evidence)
    initial = getattr(process._mapper, 'initial_floor_receipt', None)
    if initial is not None and initial['frame'] == packet.frame:
        receipt = receipt | dict(initial_floor_measurement=initial)
    return snapshot, receipt


class PairedFloorRuntimeMixin(StartupRecoveryRuntimeMixin):
    _map = bind(StartupRecoveryRuntimeMixin._map, mapping_update=mapping_update)

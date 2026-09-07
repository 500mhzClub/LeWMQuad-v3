"""Simulator-side sensor adapter. Never import native state into a controller."""
import numpy as np

from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.physical_execution_development import rotation_xyzw


class IdealFastGyro:
    def __init__(self):
        self.last_ns = None

    def sample(self, *, measured_ns, quaternion_xyzw, angular_velocity_world):
        now = _ns(measured_ns, 'virtual gyro sample')
        angular = np.asarray(angular_velocity_world, dtype=float)
        if (now % 2_000_000 or (self.last_ns is not None and now - self.last_ns != 2_000_000)
                or angular.shape != (3,) or not np.isfinite(angular).all()):
            raise SensorContractError('finite ordered actual2-ms virtual gyro measurement required')
        values = rotation_xyzw(quaternion_xyzw).T @ angular
        self.last_ns = now
        return values, np.ones(3, dtype=bool)

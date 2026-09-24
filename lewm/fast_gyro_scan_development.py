"""Unchanged scan decision rule supplied by separate causal high-rate gyro."""
import numpy as np

from lewm.active_gyro_scan_development import ActiveGyroScan
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.causal_sensor_state import SensorContractError


class FastGyroScan(ActiveGyroScan):
    def __init__(self):
        super().__init__()
        self.orientation = FastRelativeOrientation()

    def begin(self, packet, fast_packet, *, now_ns):
        if self.status != 'NEW':
            raise SensorContractError('fresh scan required')
        try:
            attitude = self.orientation.begin(packet, fast_packet, now_ns=now_ns)
            self.start_ns = now_ns
            self.views = [{'view_index': 0, 'decision_ns': now_ns, 'relative_heading_rad': 0.,
                           'rotation_initial_body_from_current_body': np.eye(3).tolist()}]
            self.status = 'SCANNING'
            return self._decision(packet, attitude)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('fast scan initialization failed') from error

    def step(self, packet, fast_packet, *, now_ns):
        if self.status not in ('SCANNING', 'DWELLING'):
            raise SensorContractError('active scan required; terminal action is explicit zero')
        try:
            attitude = self.orientation.step(packet, fast_packet, now_ns=now_ns)
            return self._decision(packet, attitude)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.status = 'FAILED_SENSOR'
            raise SensorContractError('fast scan sensor update failed') from error

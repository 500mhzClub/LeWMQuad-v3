"""Consume every gyro interval even when visual processing skips camera frames.

This retains measured rotation for later image consistency checks. It does not
supply visual translation, replace visual pose or permit camera-frame gaps in
the existing tracker. A single acquisition owner must call observe in order.
"""
from collections import OrderedDict
from copy import deepcopy

from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.fast_gyro_development import FastRelativeOrientation

RETAINED_SNAPSHOTS = 16


class CameraIndependentGyro:
    def __init__(self):
        self.integrator = FastRelativeOrientation()
        self.snapshots = OrderedDict()

    def observe(self, policy, fast, *, now_ns):
        """Process one complete 100 ms packet, including all 50 gyro intervals."""
        integrator = self.integrator
        method = integrator.begin if integrator.status == 'NEW' else integrator.step
        result = method(policy, fast, now_ns=now_ns)
        self.snapshots[result['decision_ns']] = deepcopy(result)
        while len(self.snapshots) > RETAINED_SNAPSHOTS:
            self.snapshots.popitem(last=False)
        return deepcopy(result)

    def for_camera(self, *, measured_ns):
        """Return only a retained exact-time result; never interpolate a gap."""
        stamp = _ns(measured_ns, 'camera gyro timestamp')
        if self.integrator.status != 'ACTIVE':
            raise SensorContractError('active uninterrupted gyro stream required')
        if stamp not in self.snapshots:
            raise SensorContractError('exact camera-time gyro snapshot unavailable')
        return deepcopy(self.snapshots[stamp])

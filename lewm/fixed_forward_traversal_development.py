"""Non-learned forward control under the unchanged traversal/arrival wrapper."""
import hashlib

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.observed_traversal_controller_development import ObservedTraversalController


class FixedForwardChoice:
    def __init__(self):
        self.history = OnlineRGBHistory()
        self.history.begin_episode((0, 0, 0))
        self.clock = self.start_ns = self.last_select = None
        self.image_hash = None

    def observe(self, packet, *, now_ns):
        self.history.push(packet, now_ns=now_ns)
        self.clock = now_ns
        self.image_hash = hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()

    def begin_control(self, direction, *, now_ns):
        self.history.tensors(now_ns=now_ns)
        if (self.start_ns is not None or self.clock != now_ns or np.shape(direction) != (2,)
                or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-.8) > 1e-8):
            raise SensorContractError('fresh four-frame primitive initialization required')
        self.start_ns = now_ns

    def select(self, *, now_ns):
        expected = self.start_ns if self.last_select is None else self.last_select+500_000_000
        if expected is None or self.clock != now_ns or now_ns != expected:
            raise SensorContractError('fresh half-second primitive decision required')
        self.history.tensors(now_ns=now_ns)
        self.last_select = now_ns
        return {'method': 'fixed_forward', 'decision_ns': now_ns, 'selected_action_name': 'forward',
                'requested_command_tape': [[.3, 0., 0.]]*5,
                'image_sha256': self.image_hash, 'learned_prediction_used': False,
                'scope': 'fixed primitive; same hold cadence and traversal wrapper, no learned ranker'}


class FixedForwardTraversal(ObservedTraversalController):
    def __init__(self, geometry):
        # A separately scoped successor composition; predecessor source and
        # experiment remain unchanged. The inherited stop/proposal logic is exact.
        super().__init__('always_stop', geometry)
        self.method = 'fixed_forward'
        self.adapter = FixedForwardChoice()

"""Declared red-left/blue-right marker baseline using current camera pixels only.

Not appearance-general semantics or place recognition. An identical visual copy
is intentionally indistinguishable; the task must specify which physical pattern
counts as a beacon. Absence of this pattern does not establish empty/free space.
"""
from copy import deepcopy
import hashlib

import numpy as np
from scipy.ndimage import find_objects, label

from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.simulated_body_observation_development import validate_policy_packet

MARKER_ID = 'red_left_blue_right_panel_v1'


def _components(mask):
    labels, _ = label(mask)  # Four-connected pixels, no hole filling.
    counts = np.bincount(labels.ravel())
    rows = []
    for index, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        y, x = slices
        width, height = x.stop-x.start, y.stop-y.start
        area = int(counts[index])
        if width >= 4 and height >= 8 and area >= 64 and area/(width*height) >= .75:
            rows.append({'bbox_xyxy': [x.start, y.start, x.stop, y.stop], 'positive_pixels': area})
    return rows


def observe_marker(packet, *, now_ns):
    validate_policy_packet(packet)
    now_ns = _ns(now_ns, 'marker clock')
    if packet['image']['measured_ns'] != now_ns or packet['sensor_state']['decision_ns'] != now_ns:
        raise SensorContractError('current marker observation required')
    rgb = packet['image']['rgb']
    red, green, blue = np.moveaxis(rgb.astype(np.int16), -1, 0)
    reds = _components((red >= 16) & (red >= 2*green) & (red >= 2*blue))
    blues = _components((blue >= 16) & (blue >= 2*red) & (blue >= 2*green))
    matches = []
    for r in reds:
        rx0, ry0, rx1, ry1 = r['bbox_xyxy']
        rw, rh = rx1-rx0, ry1-ry0
        for b in blues:
            bx0, by0, bx1, by1 = b['bbox_xyxy']
            bw, bh = bx1-bx0, by1-by0
            overlap = max(0, min(ry1, by1)-max(ry0, by0))
            if (0 <= bx0-rx1 <= .5*max(rw, bw) and .5 <= rw/bw <= 2.
                    and .7 <= rh/bh <= 1/.7 and overlap/max(rh, bh) >= .7):
                matches.append({'marker_id': MARKER_ID, 'red_component': r, 'blue_component': b,
                                'bbox_xyxy': [rx0, min(ry0, by0), bx1, max(ry1, by1)]})
    return {'decision_ns': now_ns, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'detections': matches, 'marker_identity_definition': 'ordered adjacent red/blue rectangular panels',
            'place_identity': None, 'metric_distance_available': False,
            'appearance_generalization_qualified': False}


class MarkerDiscovery:
    """Three consecutive 100-ms RGB observations register one pattern identity.

    Duplicate timestamps are idempotent only for identical pixels. A time gap or
    absent detection resets the current streak, not past discoveries. Faults latch
    and must stop the enclosing controller; a new episode requires a new instance.
    """
    def __init__(self):
        self._identity = None
        self._last = None
        self._streak = []
        self._discovered = {}
        self._failed = False
        self._last_result = None

    def observe(self, packet, *, now_ns):
        if self._failed:
            raise SensorContractError('marker observer fault is latched')
        try:
            row = observe_marker(packet, now_ns=now_ns)
            identity = _identity(packet['sensor_state']['identity'])
            if self._identity is not None and self._identity != identity:
                raise SensorContractError('marker observer episode/reset changed')
            if self._last is not None and now_ns <= self._last['decision_ns']:
                if row == self._last and now_ns == self._last['decision_ns']:
                    return deepcopy(self._last_result)
                raise SensorContractError('stale or rewritten marker observation')
            if self._last is not None and now_ns-self._last['decision_ns'] != 100_000_000:
                self._streak = []
            evidence = {'decision_ns': row['decision_ns'], 'rgb_sha256': row['rgb_sha256']}
            self._streak = [*self._streak[-2:], evidence] if row['detections'] else []
            new = len(self._streak) == 3 and MARKER_ID not in self._discovered
            if new:
                self._discovered[MARKER_ID] = deepcopy(self._streak)
            self._identity, self._last = identity, row
            self._last_result = {**row, 'consecutive_observations': len(self._streak),
                                 'newly_discovered': [MARKER_ID] if new else [],
                                 'discovered': deepcopy(self._discovered),
                                 'distinct_marker_count': len(self._discovered)}
            return deepcopy(self._last_result)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self._failed = True
            raise SensorContractError('marker observation failed; enclosing controller must stop') from error

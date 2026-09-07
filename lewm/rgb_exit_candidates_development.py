"""Current RGB/body floor-extension proposals, not certified topological exits.

The static flat-walled-maze prior motivates a fixed radial support annulus.
No map, world pose, hidden exit, arrival, or beacon identity is an input.
Absence of positive evidence is unknown, not a closed branch.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.ground_projection_envelope_development import observe_ground_envelope, ORIGIN_BODY

ANGULAR_EDGES = np.deg2rad(np.arange(-50., 52., 2.))
RADIAL_EDGES = np.arange(1., 1.600001, .1)


@dataclass(frozen=True)
class ExitCandidate:
    observation_id: str
    timestamp_ns: int
    bearing_body_rad: float
    angular_lower_rad: float
    angular_upper_rad: float
    supported_angle_bins: int
    support_points: int
    radial_bands_per_angle_minimum: int
    qualified_exit: bool = False
    qualified_traversal: bool = False

    def __post_init__(self):
        if (not isinstance(self.observation_id, str) or not self.observation_id
                or type(self.timestamp_ns) is not int or self.timestamp_ns < 0
                or self.qualified_exit is not False or self.qualified_traversal is not False):
            raise SensorContractError('explicit unqualified proposal identity required')
        values = (self.bearing_body_rad, self.angular_lower_rad, self.angular_upper_rad)
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                or not -math.pi <= self.angular_lower_rad <= self.bearing_body_rad <= self.angular_upper_rad <= math.pi
                or type(self.supported_angle_bins) is not int or self.supported_angle_bins < 4
                or type(self.radial_bands_per_angle_minimum) is not int or not 3 <= self.radial_bands_per_angle_minimum <= 6
                or type(self.support_points) is not int or self.support_points < 3 * self.supported_angle_bins):
            raise SensorContractError('finite angular interval and explicit multi-band support required')


def candidates_from_floor_points(points, observed, *, timestamp_ns, observation_id):
    points, observed = np.asarray(points, dtype=float), np.asarray(observed)
    if (points.shape[:-1] != observed.shape or points.shape[-1:] != (3,) or observed.dtype != bool
            or not np.isfinite(points[observed]).all()
            or type(timestamp_ns) is not int or timestamp_ns < 0
            or not isinstance(observation_id, str) or not observation_id):
        raise SensorContractError('finite observed points and explicit current observation identity required')
    xy = points[observed, :2]
    distance = np.linalg.norm(xy, axis=-1)
    bearing = np.arctan2(xy[:, 1], xy[:, 0])
    angular = np.searchsorted(ANGULAR_EDGES, bearing, side='right') - 1
    radial = np.searchsorted(RADIAL_EDGES, distance, side='right') - 1
    valid = (angular >= 0) & (angular < len(ANGULAR_EDGES) - 1) & (radial >= 0) & (radial < len(RADIAL_EDGES) - 1)
    support = np.zeros((len(ANGULAR_EDGES) - 1, len(RADIAL_EDGES) - 1), dtype=np.int64)
    np.add.at(support, (angular[valid], radial[valid]), 1)
    bands = (support > 0).sum(1)
    positive = bands >= 3
    result = []
    start = None
    for index, value in enumerate([*positive, False]):
        if value and start is None:
            start = index
        if not value and start is not None:
            if index - start >= 4:  # At least8 degrees of unbroken angular support.
                lower, upper = float(ANGULAR_EDGES[start]), float(ANGULAR_EDGES[index])
                result.append(ExitCandidate(observation_id + f':proposal-{len(result)}', timestamp_ns,
                                            (lower + upper) / 2, lower, upper, index - start,
                                            int(support[start:index].sum()), int(bands[start:index].min())))
            start = None
    return {'candidates': result, 'angle_radial_support_counts': support,
            'supported_angle_bins': positive, 'observed_point_count': int(observed.sum()),
            'annulus_supported_point_count': int(support.sum()), 'absence_means_closed': False,
            'radial_annulus_m': [1., 1.6], 'metric_clearance_qualified': False}


def observe_exit_candidates(packet, ground_state, *, now_ns, observation_id):
    # The plane's validity remains conditional: zero radius here computes the
    # nominal location only, not a calibrated zero-uncertainty claim.
    floor = observe_ground_envelope(packet, ground_state, now_ns=now_ns,
                                    height_radius=0., angle_radius=0.)
    depth = floor['nominal_optical_depth_m']
    points = ORIGIN_BODY + depth[..., None] * floor['rays_body']
    observed = floor['nominal_valid'] & floor['bottom_connected_floor_pixels']
    result = candidates_from_floor_points(points, observed, timestamp_ns=int(now_ns), observation_id=observation_id)
    return {**result, 'decision_ns': int(now_ns), 'positive_floor_pixels': int(floor['positive_floor_pixels'].sum()),
            'body_ground_estimator': ground_state.get('estimator_mode', 'gyro_only'),
            'candidate_rows': [asdict(candidate) for candidate in result['candidates']],
            'scope': 'nominal visible-floor extensions only; no place/exit association, body-volume or traversability qualification'}

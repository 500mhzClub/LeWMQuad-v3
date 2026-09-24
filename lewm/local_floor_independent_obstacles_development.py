"""Inactive local floor-depth treatment for independent current obstacle sensing.

Only floor candidate geometry is estimated locally. Obstacle points, unknown
rays, cell geometry, gyro integration and plane acceptance rules are unchanged.
"""
from lewm.eligible_floor_registration_development import bind
from lewm.fine_obstacle_round_trip_development import FineDepthObstacles
from lewm.partial_height_round_trip_development import PartialHeightObstacles, _partial_observe
from lewm.local_inverse_depth_floor_development import measured_candidates

_initial = bind(FineDepthObstacles._observe, measured_candidates=measured_candidates)
_subsequent = bind(_partial_observe, measured_candidates=measured_candidates)


class LocalFloorIndependentObstacles(PartialHeightObstacles):
    def _observe(self, policy, depth, fast, auxiliary, now):
        extract = _initial if self.frames == 0 else _subsequent
        current = extract(self, policy, depth, fast, auxiliary, now)
        self.receipts[-1].update(
            floor_candidate_depth_source='local_inverse_depth_5x5',
            floor_candidates_are_raw_pixel_depth=False,
            obstacle_points_use_original_depth=True,
            invalid_depth_filled=False,
            floor_acceptance_thresholds_changed=False)
        return current

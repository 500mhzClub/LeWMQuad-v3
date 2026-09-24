"""Prospective turn-only obstacle evidence when primary depth has no returns.

The original current floor fit, timing, nominal disk and mission gates remain.
This is sampled auxiliary obstacle evidence, not a visibility or body-safety
certificate. No missing primary ray becomes an observed free ray.
"""
import numpy as np
from lewm.causal_depth_observation_development import body_points
from lewm.auxiliary_downward45_depth_observation_development import body_points as auxiliary_points
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles
from lewm.robust_height_floor_tracking_development import RobustHeightIndependentObstacles
from lewm import fine_obstacle_round_trip_development as fine
from lewm.eligible_floor_registration_development import bind
from lewm.paced_multirate_controller_development import PacedMultirateController


def auxiliary_only_obstacles(policy, depth, auxiliary, receipt, *, now_ns):
    """Extract current raw obstacle returns using the observer's accepted plane."""
    if receipt['measured_ns'] != now_ns:
        raise ValueError('same current floor acquisition required')
    plane = receipt['joint_plane']
    if not plane['available']:
        return None
    clouds = []
    for packet, project in ((depth, body_points), (auxiliary, auxiliary_points)):
        cloud = project(packet, policy, now_ns=now_ns, stride=4)
        clouds.append(cloud['points_body_m'][cloud['valid']])
    if len(clouds[0]) or not len(clouds[1]):
        return None
    points = clouds[1]
    height = points @ np.asarray(plane['normal_body']) + plane['offset_body_m']
    above = points[(height > .03) & (height < .65)]
    cells = frozenset(tuple(map(int, k)) for k in
        np.unique(np.floor(above[:, :2] / fine.CELL_M).astype(int), axis=0))
    return CurrentObstacles(receipt['frame'], now_ns, (0., 0., 0.),
        tuple(map(tuple, np.eye(3))), cells, (0, len(points)), 'current_body')


class AuxiliaryTurnObstacles(RobustHeightIndependentObstacles):
    def _observe(self, policy, depth, fast, auxiliary, now):
        current = super()._observe(policy, depth, fast, auxiliary, now)
        if current is not None:
            return current
        receipt = self.receipts[-1]
        current = auxiliary_only_obstacles(policy, depth, auxiliary, receipt, now_ns=now)
        if current is not None:
            receipt.update(primary_depth_unavailable=True,
                auxiliary_only_current_obstacles=True, translation_requires_both_cameras=True,
                missing_primary_rays_inferred_free=False)
        return current


def dispatch_request(plan, current, *, now_ns):
    result = fine.dispatch_request(plan, current, now_ns=now_ns)
    if current is None or current.valid_return_counts[0] != 0:
        return result
    result = result | dict(auxiliary_only_turn_recovery=True,
        current_valid_return_counts=list(current.valid_return_counts),
        translation_requires_both_cameras=True, missing_primary_rays_inferred_free=False)
    if plan is not None and any(plan.command[:2]):
        return result | dict(requested_command=[0., 0., 0.],
            reason='PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO')
    return result


class AuxiliaryTurnDispatch(fine._FineDispatch):
    request = bind(PacedMultirateController.request, dispatch_request=dispatch_request)


def initialize_obstacles():
    from lewm import independent_depth_process_development as process
    from scripts.run_go2_live_gyro_height_floor_noise_development import initialize_obstacles as previous
    previous()
    process._observer = AuxiliaryTurnObstacles()

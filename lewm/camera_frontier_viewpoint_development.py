"""Propose an observed-route viewing position from calibrated camera geometry.

Projection is a visibility hypothesis, not a floor observation or motion grant.
The current measured roll, pitch and height are retained for a candidate yaw.
"""
import math
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.observed_floor_waypoint_development import centre, segment_cells


def floor_cell_projection(cell, position_map, rotation_map_from_body, floor_height):
    cell = np.asarray(cell, int); p = np.asarray(position_map, float)
    R = np.asarray(rotation_map_from_body, float)
    if (cell.shape != (2,) or p.shape != (3,) or R.shape != (3, 3)
            or not np.isfinite(p).all() or not np.isfinite(R).all()
            or not math.isfinite(floor_height)
            or not np.allclose(R.T@R, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-8, rtol=0)):
        raise ValueError('finite measured pose, floor height and proper rotation required')
    xy = (cell + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * .05
    world = np.c_[xy, np.full(4, floor_height)]
    rows = []
    for name, mount in (('primary', np.asarray(BODY_FROM_OPTICAL)),
                        ('auxiliary', body_from_optical())):
        camera = ((world-p)@R-mount[:3, 3])@mount[:3, :3]
        z = camera[:, 2]
        uv = camera[:, :2]/np.maximum(z[:, None], 1e-12)*FOCAL+[319.5, 239.5]
        lo, hi = uv.min(0)-1e-9, uv.max(0)+1e-9
        visible = bool(np.all((z >= .2)&(z <= 5.))
            and np.all(lo >= 0) and np.all(hi < [639, 479]))
        rows.append(dict(camera=name, fully_projected=visible,
            lower_pixel=lo.tolist(), upper_pixel=hi.tolist(),
            corner_depth_min_m=float(z.min()), corner_depth_max_m=float(z.max()),
            camera_origin_map_xy_m=(p+R@mount[:3, 3])[:2].tolist()))
    return rows


def directed_rotation(rotation, position_xy, cell):
    R = np.asarray(rotation, float)
    direction = centre(cell)-position_xy
    heading = math.atan2(direction[1], direction[0])
    delta = heading-math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(delta), math.sin(delta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])@R, heading


def choose_route_viewpoint(snapshot, route, position_map, rotation_map_from_body, unknown_cell,
                           *, excluded_viewpoint_cells=()):
    """Use the nearest-to-frontier feasible point on the existing floor route."""
    if route['status'] != 'OBSERVED_FLOOR_ROUTE_TO_FRONTIER': return None
    if tuple(unknown_cell) in snapshot.floor | snapshot.occupied: return None
    p = np.asarray(position_map, float)
    for index in range(len(route['route_cells'])-1, -1, -1):
        cell = tuple(route['route_cells'][index])
        if cell in excluded_viewpoint_cells: continue
        if cell not in snapshot.floor: raise ValueError('viewpoint must be on observed route floor')
        candidate = np.r_[centre(cell), p[2]]
        R, heading = directed_rotation(rotation_map_from_body, candidate[:2], unknown_cell)
        projected = floor_cell_projection(unknown_cell, candidate, R, snapshot.floor_height)
        cameras = [row for row in projected if row['fully_projected'] and not
            (segment_cells(row['camera_origin_map_xy_m'], centre(unknown_cell)) & snapshot.occupied)]
        if cameras:
            return dict(viewpoint_cell=list(cell), viewpoint_map_xy_m=candidate[:2].tolist(),
                route_cells=route['route_cells'][:index+1], unknown_cell=list(unknown_cell),
                frontier_target_map_xy_m=route['target_map_xy_m'],
                view_heading_rad=heading, projected_cameras=cameras,
                current_roll_pitch_height_used=True, future_visibility_requires_new_observation=True,
                unknown_floor_admitted=False, motion_authorized=False)
    return None

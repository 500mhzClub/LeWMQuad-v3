"""Evaluator-only visible depth ranges over an explicitly supplied pixel region.

Project front-facing opaque box faces and the floor. Subtract regions occluded
by nearer faces, then bound each surviving face's optical depths. The supplied
pixel radius is an assumption, not an established native or hardware error bound.
No policy data, old score, sensor pixel or public mask is changed.
"""
import numpy as np
from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth

AREA_EPS = 1e-14
POSITIVE_Z = 1e-9
MAX_PIECES = 4096


def area(polygon):
    if len(polygon) < 3: return 0.
    # Translate before products to reduce cancellation in screen coordinates.
    p = np.asarray(polygon)-polygon[0]
    return float(.5*np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))


def clip(polygon, coefficient):
    """Keep a convex polygon's closed affine >=0 half-space."""
    if len(polygon) < 3: return np.empty((0, polygon.shape[1]))
    values = polygon@coefficient[:-1]+coefficient[-1]; result = []
    for i in range(len(polygon)):
        a, b = polygon[i-1], polygon[i]; va, vb = values[i-1], values[i]
        if (va >= 0.) != (vb >= 0.):
            cross = a+va/(va-vb)*(b-a)
            # Solve one coordinate on the clipping plane explicitly. This
            # preserves exact axis-aligned rectangle boundaries instead of
            # leaving cancellation slivers outside the requested pixel region.
            axis = int(np.argmax(np.abs(coefficient[:-1])))
            other = [k for k in range(len(cross)) if k != axis]
            cross[axis] = (-coefficient[-1]-coefficient[other]@cross[other])/coefficient[axis]
            result.append(cross)
        if vb >= 0.: result.append(b)
    return np.asarray(result).reshape(-1, polygon.shape[1])


def halfspaces(polygon):
    if area(polygon) < 0: polygon = polygon[::-1]
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        d = b-a
        result.append(np.array([-d[1], d[0], d[1]*a[0]-d[0]*a[1]]))
    return result


def intersect(polygon, other):
    for edge in halfspaces(other):
        polygon = clip(polygon, edge)
        if len(polygon) < 3: break
    return polygon


def subtract(polygon, occluder):
    """Disjoint convex pieces of polygon outside a convex occluder."""
    inside = polygon; pieces = []
    for edge in halfspaces(occluder):
        outside = clip(inside, -edge)
        if len(outside) >= 3 and abs(area(outside)) > AREA_EPS: pieces.append(outside)
        inside = clip(inside, edge)
        if len(inside) < 3 or abs(area(inside)) <= AREA_EPS: break
    return pieces


def inverse_depth_plane(normal, offset):
    # Plane n.x=offset, x=z*((u-cx)/f,(v-cy)/f,1).
    return np.array([normal[0]/FOCAL, normal[1]/FOCAL,
        normal[2]-(normal[0]*320+normal[1]*240)/FOCAL])/offset


def face_candidates(boxes, transform, rectangle, floor_z_m):
    result = []; T = np.asarray(transform, float)
    normal = T[2, :3]; offset = floor_z_m-T[2, 3]
    floor_q = inverse_depth_plane(normal, offset)
    floor_polygon = clip(rectangle, floor_q)
    if len(floor_polygon) >= 3 and abs(area(floor_polygon)) > AREA_EPS:
        result.append(dict(object='ground_plane', face='floor', polygon=floor_polygon, q=floor_q))
    for box in boxes:
        c, s = np.cos(box['yaw_rad']), np.sin(box['yaw_rad'])
        R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        half = np.asarray(box['size_xyz'])/2; center = np.asarray(box['centre_xyz'])
        for axis in range(3):
            others = [k for k in range(3) if k != axis]
            for sign in (-1, 1):
                n_world = sign*R[:, axis]; face_center = center+sign*half[axis]*R[:, axis]
                n = n_world@T[:3, :3]; offset = float(n_world@(face_center-T[:3, 3]))
                # The camera must be outside this face's supporting half-space.
                if offset >= 0.: continue
                corners = []
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    point = np.zeros(3); point[axis] = sign*half[axis]
                    point[others] = [a*half[others[0]], b*half[others[1]]]
                    corners.append((center+R@point-T[:3, 3])@T[:3, :3])
                optical = clip(np.asarray(corners), np.array([0., 0., 1., -POSITIVE_Z]))
                if len(optical) < 3: continue
                uv = optical[:, :2]/optical[:, 2, None]*FOCAL+[320, 240]
                polygon = intersect(uv, rectangle)
                if len(polygon) < 3 or abs(area(polygon)) <= AREA_EPS: continue
                result.append(dict(object=box['wall_id'], face=f'{axis}:{sign}', polygon=polygon,
                    q=inverse_depth_plane(n, offset)))
    return result


def visible_intervals(boxes, transform, pixel_row_column, *, radius_pixels, floor_z_m=0.):
    expected_optical_depth(boxes, transform, floor_z_m=floor_z_m)
    pixel = np.asarray(pixel_row_column)
    if (pixel.shape != (2,) or pixel.dtype.kind not in 'iu' or not 0 <= pixel[0] < 480
            or not 0 <= pixel[1] < 640 or not np.isfinite(radius_pixels) or not 0 < radius_pixels <= .5
            or not 1 <= len(boxes) <= 128 or len({b['wall_id'] for b in boxes}) != len(boxes)):
        raise ValueError('bounded unique geometry and explicit pixel region required')
    row, column = pixel; u, v = column+.5, row+.5; r = radius_pixels
    rectangle = np.array([[u-r, v-r], [u+r, v-r], [u+r, v+r], [u-r, v+r]])
    candidates = face_candidates(boxes, transform, rectangle, floor_z_m)
    intervals = []
    for face in candidates:
        pieces = [face['polygon']]
        for other in candidates:
            if other is face: continue
            difference = other['q']-face['q']
            # Identical depth planes do not hide one another's common surface.
            if not np.any(difference): continue
            nearer = clip(other['polygon'], difference)
            if len(nearer) < 3 or abs(area(nearer)) <= AREA_EPS: continue
            revised = []
            for piece in pieces:
                overlap = intersect(piece, nearer)
                if len(overlap) < 3 or abs(area(overlap)) <= AREA_EPS: revised.append(piece)
                else: revised.extend(subtract(piece, nearer))
            pieces = revised
            if len(pieces) > MAX_PIECES: raise ValueError('bounded visible-fragment population exceeded')
            if not pieces: break
        for piece in pieces:
            q = piece@face['q'][:2]+face['q'][2]
            if q.max() <= 0.: continue
            low = float(1./q.max()); high = float(1./q.min()) if q.min() > 0 else None
            intervals.append(dict(object=face['object'], face=face['face'],
                optical_depth_lower_m=low, optical_depth_upper_m=high,
                visible_projected_area_pixels2=abs(area(piece)), polygon_xy=piece.tolist()))
    intervals.sort(key=lambda x: (x['optical_depth_lower_m'], x['object'], x['face']))
    return dict(pixel_row_column=pixel.tolist(), assumed_pixel_radius=float(radius_pixels), intervals=intervals,
        coincident_object_faces_may_duplicate_projected_area=True,
        supplied_angular_bound_validated=False, floating_point_error_bound_proven=False,
        zero_area_tie_cases_certified=False, camera_plane_clip_m=POSITIVE_Z,
        native_near_plane_used_to_remove_occluders=False,
        native_depth_or_public_mask_changed=False, policy_filter=False, navigation_qualified=False)


def supported_depth(report, depth_m, *, metric_tolerance_m):
    if not np.isfinite(metric_tolerance_m) or not 0 <= metric_tolerance_m <= .001:
        raise ValueError('explicit metric tolerance no larger than original 1 mm required')
    return bool(np.isfinite(depth_m) and depth_m > 0 and any(
        item['optical_depth_lower_m']-metric_tolerance_m <= depth_m
        and (item['optical_depth_upper_m'] is None or depth_m <= item['optical_depth_upper_m']+metric_tolerance_m)
        for item in report['intervals']))

"""Evaluator-only projected-edge sensitivity; not a native raster error bound."""
import itertools
import numpy as np
from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth


def edge_distance(point, endpoints):
    point = np.asarray(point, float); uv = np.asarray(endpoints, float)
    if point.shape != (2,) or uv.shape != (2, 2) or not np.isfinite(point).all() or not np.isfinite(uv).all():
        raise ValueError('finite projected pixel and edge endpoints required')
    delta = uv[1]-uv[0]; length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return dict(distance_pixels=float(np.linalg.norm(point-uv[0])), signed_line_distance_pixels=None,
            closest_segment_fraction=0.)
    fraction = float(np.clip((point-uv[0])@delta/(length*length), 0., 1.))
    signed = float((delta[0]*(point[1]-uv[0,1])-delta[1]*(point[0]-uv[0,0]))/length)
    return dict(distance_pixels=float(np.linalg.norm(point-(uv[0]+fraction*delta))),
        signed_line_distance_pixels=signed, closest_segment_fraction=fraction)


def projected_edge_witnesses(boxes, transform, pixel_row_column, subpixel_bits):
    if (type(subpixel_bits) is not int or not 1 <= subpixel_bits <= 16 or not 1 <= len(boxes) <= 128
            or len(pixel_row_column) != 2 or any(type(v) is not int for v in pixel_row_column)
            or not 0 <= pixel_row_column[0] < 480 or not 0 <= pixel_row_column[1] < 640):
        raise ValueError('bounded measured raster precision, pixel and boxes required')
    expected_optical_depth(boxes, transform, stride=480)
    T = np.asarray(transform, float); point = np.array(pixel_row_column[::-1], float)+.5
    signs = np.array(list(itertools.product((-1, 1), repeat=3)))
    indices = [(i, j) for i, j in itertools.combinations(range(8), 2) if np.count_nonzero(signs[i]!=signs[j]) == 1]
    records = []; scale = 2**subpixel_bits; positive = 1e-9
    for box in boxes:
        c, s = np.cos(box['yaw_rad']), np.sin(box['yaw_rad'])
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        world = (signs*np.asarray(box['size_xyz'])/2)@rotation.T+box['centre_xyz']
        optical = (world-T[:3,3])@T[:3,:3]
        for i, j in indices:
            a, b = optical[i].copy(), optical[j].copy()
            if max(a[2], b[2]) <= positive: continue
            if a[2] < positive: a += (positive-a[2])/(b[2]-a[2])*(b-a)
            if b[2] < positive: b += (positive-b[2])/(a[2]-b[2])*(a-b)
            uv = np.stack((a, b)); uv = uv[:,:2]/uv[:,2,None]*FOCAL+[320, 240]
            rounded = np.rint(uv*scale)/scale
            original = edge_distance(point, uv); quantized = edge_distance(point, rounded)
            x, y = original['signed_line_distance_pixels'], quantized['signed_line_distance_pixels']
            records.append(dict(wall_id=box['wall_id'], corner_indices=[i, j],
                exact_projected_endpoints_xy=uv.tolist(), hypothetical_nearest_grid_endpoints_xy=rounded.tolist(),
                original=original, hypothetical_nearest_grid=quantized,
                hypothetical_line_side_changes=bool(x is not None and y is not None and x*y < 0),
                native_rounding_rule_proven=False, native_depth_explained=False))
    return sorted(records, key=lambda r: (r['original']['distance_pixels'], r['wall_id'], r['corner_indices']))

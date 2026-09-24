"""Hypothetical snapping of actual float32 mesh inputs, not native raster replay."""
import numpy as np
from lewm.causal_depth_observation_development import FOCAL


def barycentric(triangles, point):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    d0, d1, q = b-a, c-a, point-a
    def cross(x, y): return x[:, 0]*y[:, 1]-x[:, 1]*y[:, 0]
    determinant = cross(d0, d1)
    with np.errstate(divide='ignore', invalid='ignore'):
        w1, w2 = cross(q, d1)/determinant, cross(d0, q)/determinant
    return np.stack((1-w1-w2, w1, w2), axis=1)


def diagnose_mesh(vertices, faces, transform, pixel_row_column, *, subpixel_bits, near_m):
    v, f, T, pixel = np.asarray(vertices), np.asarray(faces), np.asarray(transform, float), np.asarray(pixel_row_column)
    if (v.dtype != np.float32 or v.ndim != 2 or v.shape[1:] != (3,) or not 3 <= len(v) <= 1_000_000
            or not np.isfinite(v).all() or f.ndim != 2 or f.shape[1:] != (3,) or f.dtype.kind not in 'iu'
            or not 1 <= len(f) <= 200_000 or f.min() < 0 or f.max() >= len(v)
            or T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0., 0., 0., 1.])
            or not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-10, rtol=0)
            or pixel.shape != (2,) or pixel.dtype.kind not in 'iu' or not 0 <= pixel[0] < 480 or not 0 <= pixel[1] < 640
            or type(subpixel_bits) is not int or not 1 <= subpixel_bits <= 16
            or not np.isfinite(near_m) or near_m <= 0.):
        raise ValueError('bounded native float32 inputs, proper pose and pixel required')
    triangles = ((v.astype(float)-T[:3, 3])@T[:3, :3])[f]
    ids = np.flatnonzero((triangles[:, :, 2] > near_m).all(axis=1))
    triangles = triangles[ids]
    uv = triangles[:, :, :2]/triangles[:, :, 2, None]*FOCAL+[320, 240]
    scale = 2**subpixel_bits; snapped = np.rint(uv*scale)/scale
    point = pixel[::-1]+.5
    weights, rounded = barycentric(uv, point), barycentric(snapped, point)
    original_hit = np.isfinite(weights).all(axis=1)&(weights >= 0.).all(axis=1)
    rounded_hit = np.isfinite(rounded).all(axis=1)&(rounded >= 0.).all(axis=1)
    reports = {}
    for label, hits in (('exact_mesh_projection', original_hit), ('hypothetical_snapped_projection', rounded_hit)):
        records = []
        for i in np.flatnonzero(hits):
            inverse_depth = float(np.sum(weights[i]/triangles[i, :, 2]))
            records.append(dict(triangle_index=int(ids[i]), exact_projection_covers_sample=bool(original_hit[i]),
                hypothetical_snapped_projection_covers_sample=bool(rounded_hit[i]),
                unsnapped_plane_depth_m=float(1/inverse_depth) if inverse_depth > 0 else None,
                world_vertices=v[f[ids[i]]].tolist(), projected_vertices_xy=uv[i].tolist(),
                hypothetical_snapped_vertices_xy=snapped[i].tolist(),
                unsnapped_plane_value_may_extrapolate_outside_triangle=not bool(original_hit[i])))
        records.sort(key=lambda x: (x['unsnapped_plane_depth_m'] or float('inf'), x['triangle_index']))
        reports[label] = dict(covering_triangles=len(records), nearest_candidates=records[:5])
    return dict(pixel_row_column=pixel.tolist(), subpixel_bits=subpixel_bits, vertices=len(v), triangles=len(f),
        tested_unclipped_triangles=len(ids), excluded_triangles=len(f)-len(ids), **reports,
        native_clipping_shader_arithmetic_and_culling_reproduced=False,
        framebuffer_depth_interpolation_reproduced=False, native_rounding_mode_proven=False,
        complete_visibility_evaluation=False, native_pixels_changed=False, navigation_qualified=False)

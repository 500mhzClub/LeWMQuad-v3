"""Actual visual-surface reference; physical-ground offset is separately reported."""
import math

import numpy as np

from lewm.causal_depth_observation_development import FOCAL


def check_floor_identity(record):
    expected = {'visual_local_vertices_m', 'visual_faces', 'visual_position_world_m', 'visual_quaternion_wxyz',
                'collision_position_world_m', 'collision_quaternion_wxyz', 'collision_plane_data',
                'collision_enabled', 'scope'}
    if set(record) != expected or record['collision_enabled'] is not True:
        raise ValueError('exact native floor identity required')
    for key in ('visual_position_world_m', 'collision_position_world_m'):
        if not np.array_equal(record[key], [0., 0., 0.]): raise ValueError('fixed plane origin differs')
    for key in ('visual_quaternion_wxyz', 'collision_quaternion_wxyz'):
        if not np.array_equal(record[key], [1., 0., 0., 0.]): raise ValueError('fixed plane orientation differs')
    if not np.array_equal(record['collision_plane_data'], [0., 0., 1., 0., 0., 0., 0.]):
        raise ValueError('native z-up collision plane differs')
    vertices, faces = np.asarray(record['visual_local_vertices_m']), np.asarray(record['visual_faces'])
    if (vertices.shape not in ((4, 3), (6, 3)) or faces.shape != (2, 3) or not np.isfinite(vertices).all()
            or not np.allclose(vertices[:, 2], -.005, atol=1e-9, rtol=0)
            or not np.all(np.abs(vertices[:, :2]) == 500.) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('actual visual floor mesh differs from reviewed source')
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    if (len(np.unique(triangles.reshape(-1, 3), axis=0)) != 4
            or not np.array_equal(cross, [[0., 0., 1e6], [0., 0., 1e6]])):
        raise ValueError('two complete nonoverlapping upward square triangles required')
    return {'visual_floor_z_m': -.005, 'physical_floor_z_m': 0.,
            'rendered_floor_below_collision_m': .005, 'physical_clearance_qualified': False}


def expected_optical_depth(boxes, world_from_optical, *, stride=8, floor_z_m=-.005):
    """Independent nearest-surface ray intersection at native pixel centres."""
    rows, columns = np.arange(stride//2, 480, stride), np.arange(stride//2, 640, stride)
    u, v = np.meshgrid(columns+.5, rows+.5)
    rays = np.stack(((u-320)/FOCAL, (v-240)/FOCAL, np.ones_like(u)), axis=-1)
    transform = np.asarray(world_from_optical)
    origin = transform[:3, 3]
    direction = rays@transform[:3, :3].T
    distances, margins, names = [], [], []
    ground = np.divide(floor_z_m-origin[2], direction[..., 2], out=np.full(u.shape, np.inf), where=direction[..., 2] < -1e-12)
    distances.append(np.where(ground > .05, ground, np.inf))
    margins.append(np.full(u.shape, np.inf)); names.append('ground_plane')
    for box in boxes:
        angle = box['yaw_rad']; c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        p = (origin-np.asarray(box['centre_xyz']))@rotation
        d = direction@rotation
        half = np.asarray(box['size_xyz'])/2
        moving = np.abs(d) > 1e-12
        a = np.divide(-half-p, d, out=np.full_like(d, -np.inf), where=moving)
        b = np.divide(half-p, d, out=np.full_like(d, np.inf), where=moving)
        entry = np.max(np.minimum(a, b), axis=-1)
        leave = np.min(np.maximum(a, b), axis=-1)
        outside_parallel = np.any((~moving) & (np.abs(p) > half), axis=-1)
        visible = ~outside_parallel & (entry > .05) & (entry <= leave)
        hit = p+np.where(visible, entry, 0.)[..., None]*d
        # Exclude points within2cm of a box edge, fixed before actual rendering.
        face_distances = np.sort(np.abs(half-np.abs(hit)), axis=-1)
        margins.append(np.where(visible, face_distances[..., 1], 0.))
        distances.append(np.where(visible, entry, np.inf)); names.append(box['wall_id'])
    distances = np.stack(distances)
    winner = np.argmin(distances, axis=0)
    expected = np.min(distances, axis=0)
    margin = np.take_along_axis(np.stack(margins), winner[None], axis=0)[0]
    return {'rows': rows, 'columns': columns, 'expected_depth_m': expected, 'object_index': winner,
            'object_names': names, 'surface_interior': margin > .02}


def evaluate_depth(native_depth, boxes, transform, *, marker_case, floor_z_m=-.005):
    reference = expected_optical_depth(boxes, transform, floor_z_m=floor_z_m)
    actual = native_depth[np.ix_(reference['rows'], reference['columns'])]
    expected = reference['expected_depth_m']
    eligible = reference['surface_interior'] & (expected > .22) & (expected < 4.98)
    error = np.abs(actual[eligible]-expected[eligible])
    names = reference['object_names']
    populations = {name: int(np.count_nonzero(eligible & (reference['object_index'] == i))) for i, name in enumerate(names)}
    panels = sum(n for name, n in populations.items() if name.endswith('_panel'))
    marker_visibility = panels >= 20 if marker_case == 'positive' else panels == 0 and populations.get('marker_occluder', 0) >= 100
    # Truly missing/background readings are not treated as free at sensor max.
    background = ~np.isfinite(expected)
    # The close occluder can cover the whole camera frustum. Require background
    # coverage in the visible case, and invalidity wherever it exists in either.
    background_invalid = bool((marker_case != 'positive' or np.any(background))
        and np.all((~np.isfinite(actual[background])) | (actual[background] > 5.)))
    checks = {'sufficient_interior_surface_rays': len(error) >= 1000,
              'metric_optical_depth_within_5mm': bool(len(error) and np.isfinite(error).all() and error.max() <= .005),
              'expected_visible_occluded_geometry': bool(marker_visibility),
              'background_is_invalid_not_free_space': background_invalid}
    return {'checks': checks, 'passes_declared_depth_check': all(checks.values()),
            'interior_rays': int(len(error)), 'maximum_absolute_error_m': float(error.max()) if len(error) else None,
            'mean_absolute_error_m': float(error.mean()) if len(error) else None,
            'background_rays': int(np.count_nonzero(background)), 'visible_surface_rays': populations,
            'visual_floor_z_m': floor_z_m, 'physical_floor_z_m': 0.,
            'scope': 'visual-surface depth check; collision-floor offset explicit, no physical-clearance qualification'}

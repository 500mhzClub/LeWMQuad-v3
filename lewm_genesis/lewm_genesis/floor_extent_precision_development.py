"""Fresh development comparison of tangential plane extent and depth decoding.

No correction of old data or policy inputs. Both native surfaces remain aligned
at zero; only visual extent differs. Finite visible support must cover every
reference ray being assessed, otherwise no geometry-equivalence claim is made.
"""
from copy import deepcopy
import math

import numpy as np
import numba as nb

from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_semantics import world_from_optical
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.aligned_floor_development import check_aligned_floor_identity, _array

EXTENTS_M = (1000., 32.)
# x,y,z,yaw,pitch; new development viewpoints, not sealed/held-out navigation.
VIEWS = ((0., 0., .30, .17, -.08), (1., -1., .35, .53, -.12),
         (-1., 1., .42, -.67, -.21), (1.2, .8, .30, 1.19, -.06),
         (-1.1, -.7, .40, -1.47, -.18), (.6, 1.3, .33, 2.1, -.10),
         (-.8, -1.2, .37, -2.6, -.14), (0., .4, .38, 3., .04))


def camera_pose(view):
    x, y, z, yaw, pitch = view
    forward = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    return world_from_optical([x, y, z], forward, [0., 0., 1.])


def add_extent_floor(scene, gs, extent_m):
    if extent_m not in EXTENTS_M: raise ValueError('fixed preregistered extents only')
    collision = scene.add_entity(gs.morphs.Plane(visualization=False))
    visual = scene.add_entity(gs.morphs.Plane(pos=(0., 0., .005), collision=False,
                                            plane_size=(extent_m, extent_m)))
    return collision, visual


def check_extent_identity(record, extent_m):
    if extent_m not in EXTENTS_M: raise ValueError('fixed extent required')
    normalized = deepcopy(record)
    v = np.asarray(record['visual_local_vertices_m'], float)
    if (v.shape not in ((4, 3), (6, 3)) or not np.isfinite(v).all()
            or not np.all(np.abs(v[:, :2]) == extent_m / 2)):
        raise ValueError('actual visual extent differs')
    # Reuse the frozen role/plane/winding test after an EXACT tangential scale.
    # Actual z, entity positions, collision data and all flags remain unchanged.
    v = v.copy(); v[:, :2] *= 1000. / extent_m
    normalized['visual_local_vertices_m'] = v.tolist()
    return check_aligned_floor_identity(normalized) | {'actual_visual_extent_m': extent_m}


def read_extent_identity(collision, visual, extent_m):
    if (collision is visual or type(collision.morph).__name__ != 'Plane' or type(visual.morph).__name__ != 'Plane'
            or len(collision.geoms) != 1 or collision.vgeoms or visual.geoms or len(visual.vgeoms) != 1):
        raise ValueError('one collision-only and one visual-only native plane required')
    g, v = collision.geoms[0], visual.vgeoms[0]; mesh = v.get_trimesh()
    record = {'schema': 'aligned_native_floor_development_v1',
        'collision_enabled': bool(collision.morph.collision),
        'collision_visualization': bool(collision.morph.visualization),
        'appearance_collision': bool(visual.morph.collision),
        'appearance_visualization': bool(visual.morph.visualization),
        'visual_local_vertices_m': np.asarray(mesh.vertices).tolist(), 'visual_faces': np.asarray(mesh.faces).tolist(),
        'visual_position_world_m': _array(v.get_pos()).reshape(3).tolist(),
        'visual_quaternion_wxyz': _array(v.get_quat()).reshape(4).tolist(),
        'collision_position_world_m': _array(g.get_pos()).reshape(3).tolist(),
        'collision_quaternion_wxyz': _array(g.get_quat()).reshape(4).tolist(),
        'collision_plane_data': np.asarray(g.data).tolist(),
        'scope': 'evaluation-only scene identity; never policy input'}
    check_extent_identity(record, extent_m)
    return record


def decode_depth64(buffer, near=.05, far=200.):
    """Independent double-precision decode, not replacement native acquisition."""
    z = np.asarray(buffer)
    if (z.dtype != np.float32 or z.shape != (480, 640) or not np.isfinite(z).all()
            or np.any((z < 0) | (z > 1)) or near != .05 or far != 200.):
        raise ValueError('native finite normalized depth buffer and fixed clips required')
    ndc = z.astype(np.float64) * 2 - 1
    return 2 * near * far / (far + near - ndc * (far - near))


@nb.jit(nb.float32[:, :](nb.float32[:, :], nb.float32, nb.float32), cache=False)
def decode_native_reference(buffer, near=np.float32(.05), far=np.float32(200.)):
    """Same arithmetic as installed read_depth_buf, without GL acquisition."""
    depth = buffer * 2 - 1
    return (near * far) / (far + near - depth * (far - near)) * 2


def reference(view, extent_m):
    pose = camera_pose(view)
    ref = expected_optical_depth([], pose, stride=1, floor_z_m=0.)
    expected = ref['expected_depth_m']
    use = (expected >= .20) & (expected <= 5.)
    yy, xx = np.mgrid[:480, :640]
    rays = np.stack(((xx + .5 - 320) / FOCAL, (yy + .5 - 240) / FOCAL, np.ones_like(xx)), axis=-1)
    distances = np.where(use, expected, 0.)
    hit = pose[:3, 3] + distances[..., None] * (rays @ pose[:3, :3].T)
    if (not use.any() or not np.all(np.abs(hit[use, :2]) < extent_m / 2 - .01)):
        raise ValueError('all assessed physical-plane rays must lie inside the visual support')
    return expected, use


def evaluate(view, extent_m, native, raw_buffer):
    d = np.asarray(native)
    if d.shape != (480, 640) or d.dtype != np.float32: raise ValueError('native float32 optical depth required')
    expected, use = reference(view, extent_m)
    decoded = decode_depth64(raw_buffer)
    if not np.isfinite(d[use]).all(): raise ValueError('finite rendered positive rays required')
    native_error = np.abs(d[use] - expected[use])
    decoded_error = np.abs(decoded[use] - expected[use])
    return {'rays': int(use.sum()), 'native_max_error_m': float(native_error.max()),
            'native_buffer_reconstruction_exact': bool(np.array_equal(d, decode_native_reference(raw_buffer, np.float32(.05), np.float32(200.)))),
            'native_mean_error_m': float(native_error.mean()),
            'decode64_max_error_m': float(decoded_error.max()),
            'native_minus_decode64_max_m': float(np.abs(d[use] - decoded[use]).max()),
            'native_within1mm': bool(native_error.max() <= .001),
            'decode64_within1mm': bool(decoded_error.max() <= .001),
            'all_assessed_rays_inside_visual_support': True,
            'sensor_calibrated': False, 'navigation_qualified': False}

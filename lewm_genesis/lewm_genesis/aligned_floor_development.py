"""Explicit scene-construction alternative, never a correction to policy depth.

Genesis 0.4.6 renders its native plane 5 mm below its collision plane. A
collision-only plane at zero and visual-only plane at +5 mm align the actual
surfaces. The independent post-build readback must pass before acquisition.
This module does not modify the existing scene builder or any old experiment.
"""
import numpy as np


def add_aligned_floor(scene, gs, *, material=None, surface=None):
    """Add the pair before scene.build(); retain both references for audit."""
    physical = {} if material is None else {'material': material}
    visual = {} if surface is None else {'surface': surface}
    collision = scene.add_entity(gs.morphs.Plane(visualization=False), **physical)
    appearance = scene.add_entity(gs.morphs.Plane(pos=(0., 0., .005), collision=False), **visual)
    return collision, appearance


def _array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def aligned_floor_identity(collision, appearance):
    """Actual geometry readback, evaluation-only; no policy observation output."""
    if (type(collision.morph).__name__ != 'Plane' or type(appearance.morph).__name__ != 'Plane'
            or collision is appearance or len(collision.geoms) != 1 or len(collision.vgeoms) != 0
            or len(appearance.geoms) != 0 or len(appearance.vgeoms) != 1):
        raise ValueError('distinct collision-only and visual-only native planes required')
    geom, vgeom = collision.geoms[0], appearance.vgeoms[0]
    mesh = vgeom.get_trimesh()
    record = {'schema': 'aligned_native_floor_development_v1',
              'collision_enabled': bool(collision.morph.collision),
              'collision_visualization': bool(collision.morph.visualization),
              'appearance_collision': bool(appearance.morph.collision),
              'appearance_visualization': bool(appearance.morph.visualization),
              'visual_local_vertices_m': np.asarray(mesh.vertices).tolist(),
              'visual_faces': np.asarray(mesh.faces).tolist(),
              'visual_position_world_m': _array(vgeom.get_pos()).reshape(-1, 3)[0].tolist(),
              'visual_quaternion_wxyz': _array(vgeom.get_quat()).reshape(-1, 4)[0].tolist(),
              'collision_position_world_m': _array(geom.get_pos()).reshape(-1, 3)[0].tolist(),
              'collision_quaternion_wxyz': _array(geom.get_quat()).reshape(-1, 4)[0].tolist(),
              'collision_plane_data': np.asarray(geom.data).tolist(),
              'scope': 'evaluation-only scene identity; never policy input'}
    check_aligned_floor_identity(record)
    return record


def check_aligned_floor_identity(record):
    """Check native world-space coincidence, not merely the requested morph."""
    flags = {'collision_enabled': True, 'collision_visualization': False,
             'appearance_collision': False, 'appearance_visualization': True}
    keys = set(flags) | {'schema', 'visual_local_vertices_m', 'visual_faces',
                        'visual_position_world_m', 'visual_quaternion_wxyz',
                        'collision_position_world_m', 'collision_quaternion_wxyz',
                        'collision_plane_data', 'scope'}
    if (set(record) != keys or record['schema'] != 'aligned_native_floor_development_v1'
            or record['scope'] != 'evaluation-only scene identity; never policy input'
            or any(record[k] is not v for k, v in flags.items())):
        raise ValueError('exact aligned-floor identity schema and roles required')
    for key in ('visual_quaternion_wxyz', 'collision_quaternion_wxyz'):
        if not np.array_equal(record[key], [1., 0., 0., 0.]):
            raise ValueError('reviewed fixed z-up orientation required')
    if (not np.array_equal(record['collision_position_world_m'], [0., 0., 0.])
            or not np.array_equal(record['collision_plane_data'], [0., 0., 1., 0., 0., 0., 0.])):
        raise ValueError('native collision plane at zero required')
    p = np.asarray(record['visual_position_world_m'], float)
    v = np.asarray(record['visual_local_vertices_m'], float)
    f = np.asarray(record['visual_faces'])
    if (p.shape != (3,) or not np.isfinite(p).all() or not np.array_equal(p[:2], [0., 0.])
            or v.shape not in ((4, 3), (6, 3)) or not np.isfinite(v).all()
            or f.shape != (2, 3) or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v)
            or not np.all(np.abs(v[:, :2]) == 500.)):
        raise ValueError('finite complete native visual square required')
    world = v + p
    delta = float(np.max(np.abs(world[:, 2])))
    if delta > 1e-9:
        raise ValueError('actual visual and collision surfaces are not aligned')
    triangles = world[f]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if (len(np.unique(triangles.reshape(-1, 3), axis=0)) != 4
            or not np.array_equal(cross, [[0., 0., 1e6], [0., 0., 1e6]])):
        raise ValueError('two complete nonoverlapping upward triangles required')
    return {'maximum_visual_collision_offset_m': delta, 'scene_surface_alignment_verified': True,
            'contact_model_validated': False, 'physical_clearance_qualified': False,
            'hardware_calibrated': False, 'navigation_qualified': False}

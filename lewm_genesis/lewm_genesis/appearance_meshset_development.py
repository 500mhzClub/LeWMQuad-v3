"""Persist exact color/geometry witnesses, then use Genesis's supported MeshSet."""
import numpy as np
import trimesh


def visual_morph(gs, path, mesh):
    if path.exists() or path.is_symlink():
        raise ValueError('fresh visual artifact required')
    # Exclusive creation: no predecessor or partial-attempt overwrite.
    with path.open('xb') as stream:
        mesh.export(stream, file_type='ply')
    loaded = trimesh.load_mesh(path, file_type='ply', process=False)
    np.testing.assert_array_equal(loaded.vertices, mesh.vertices.astype(np.float32))
    np.testing.assert_array_equal(loaded.faces, mesh.faces)
    np.testing.assert_array_equal(loaded.visual.vertex_colors, mesh.visual.vertex_colors)
    return gs.morphs.MeshSet(files=[loaded], fixed=True, collision=False,
        visualization=True, decimate=False, convexify=False, align=False,
        file_meshes_are_zup=True)

from types import SimpleNamespace

import numpy as np
import pytest

from lewm_genesis.appearance_surface_development import patch, triangle_identity
from lewm_genesis.appearance_meshset_development import visual_morph


@pytest.mark.parametrize('arm', ['neutral', 'repeated', 'distinctive'])
def test_persisted_geometry_colors_and_visual_only_contract(tmp_path, arm):
    mesh = patch([.13, -.27, .3], [1, 0, 0], [0, 1, 0], [.5, .5], arm=arm, seed=271828)
    gs = SimpleNamespace(morphs=SimpleNamespace(MeshSet=lambda **kw: kw))
    path = tmp_path / 'surface.ply'
    morph = visual_morph(gs, path, mesh)
    assert path.is_file()
    assert morph.keys() == {'files','fixed','collision','visualization','decimate','convexify','align','file_meshes_are_zup'}
    assert morph['fixed'] and morph['visualization'] and not morph['collision']
    assert not morph['decimate'] and not morph['convexify'] and not morph['align']
    actual = morph['files'][0]
    assert triangle_identity(actual.vertices, actual.faces) == triangle_identity(mesh.vertices.astype(np.float32), mesh.faces)
    np.testing.assert_array_equal(actual.visual.vertex_colors, mesh.visual.vertex_colors)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='fresh'):
        visual_morph(gs, path, mesh)
    assert path.read_bytes() == before


def test_broken_symlink_rejected(tmp_path):
    path = tmp_path / 'surface.ply'
    path.symlink_to(tmp_path / 'missing.ply')
    with pytest.raises(ValueError, match='fresh'):
        visual_morph(None, path, None)


def test_correction_preserves_scientific_functions():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    old = ast.parse((root/'scripts/run_go2_appearance_information_development_v1.py').read_text())
    new = ast.parse((root/'scripts/run_go2_appearance_information_meshset_development_v1.py').read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    for name in ('boxes_from_recording', 'native_identity', 'collect', 'score', 'main'):
        assert a[name] == b[name], name

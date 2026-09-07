from types import SimpleNamespace

import numpy as np
import pytest

from lewm_genesis.appearance_surface_development import surfaces, APPEARANCE_SEED, triangle_identity
from lewm_genesis.rgbd_motion_scene_development import independently_seeded_surfaces,build_scene_from_pack


BOXES=[dict(wall_id='oblique_wall',centre_xyz=[1.2,-.7,.3],size_xyz=[.08,2.,.6],yaw_rad=.31)]


@pytest.mark.parametrize('arm',['neutral','repeated','distinctive'])
def test_original_seed_reproduces_frozen_geometry_and_colors_exactly(arm):
    for (name,a),(other,b) in zip(surfaces(BOXES,arm),independently_seeded_surfaces(BOXES,arm,APPEARANCE_SEED),strict=True):
        assert name==other
        np.testing.assert_array_equal(a.vertices,b.vertices)
        np.testing.assert_array_equal(a.faces,b.faces)
        np.testing.assert_array_equal(a.visual.vertex_colors,b.visual.vertex_colors)


def test_new_appearance_seed_changes_only_distinctive_colors():
    a=independently_seeded_surfaces(BOXES,'distinctive',2026090607)
    b=independently_seeded_surfaces(BOXES,'distinctive',2026090608)
    for (_,left),(_,right) in zip(a,b,strict=True):
        assert triangle_identity(left.vertices,left.faces)==triangle_identity(right.vertices,right.faces)
        assert not np.array_equal(left.visual.vertex_colors,right.visual.vertex_colors)


@pytest.mark.parametrize('arm,seed',[('semantic_goal',7),('neutral',True),('repeated',-1)])
def test_invalid_appearance_rejected(arm,seed):
    with pytest.raises(ValueError):independently_seeded_surfaces(BOXES,arm,seed)


def test_non_upright_collision_geometry_not_silently_changed(tmp_path):
    pack=SimpleNamespace(static_objects=[SimpleNamespace(roll_rad=.1,pitch_rad=0)])
    with pytest.raises(ValueError,match='roll/pitch'):
        build_scene_from_pack(pack,output=tmp_path,appearance_arm='neutral',appearance_seed=7)


@pytest.mark.parametrize('override',[dict(n_envs=2),dict(backend='gpu'),dict(render_robot=True),dict(show_viewer=True)])
def test_unreviewed_native_modes_rejected_before_build(tmp_path,override):
    with pytest.raises(ValueError,match='one CPU'):
        build_scene_from_pack(None,output=tmp_path,appearance_arm='neutral',appearance_seed=7,**override)

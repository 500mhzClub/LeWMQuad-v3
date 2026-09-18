"""Pre-acquisition split, geometry and physical/visual construction contracts."""
from collections import Counter
from copy import deepcopy
import numpy as np
import pytest
from lewm import geometry_progress_layout_family_development as family
from lewm import geometry_progress_pilot_development as pilot
from lewm_genesis.variable_height_union_surface_development import wall_union_boundary


def test_complete_balanced_cohort_and_disjoint_parameter_clusters():
    cells=family.assignments();assert len(cells)==96 and tuple(cells)==family.TRIALS
    assert Counter(c['data_role'] for c in cells.values())=={'train':48,'geometry_transfer':48}
    assert len({(c['geometry'],c['appearance_seed'],c['action']) for c in cells.values()})==96
    assert set(c['cluster'] for c in cells.values() if c['data_role']=='train').isdisjoint(
        c['cluster'] for c in cells.values() if c['data_role']=='geometry_transfer')
    for name in family.layouts():
        assert Counter(c['action'] for c in cells.values() if c['geometry']==name)=={a:2 for a in pilot.ACTIONS}
    assert family.cohort_contract()['independent_maze_evaluation_layouts']==0


def test_every_actual_new_geometry_constructs_with_original_room_and_gait():
    for name in family.layouts():
        trial=next(t for t,c in family.assignments().items() if c['geometry']==name)
        spec=family.specification(trial);pack=family.pack(spec)
        surface=wall_union_boundary(spec['geometry']['wall_boxes'])
        assert surface['faces'] and pack.camera.near_m==.005
        assert spec['geometry']['wall_boxes'][1:]==pilot.geometry('left_open')['wall_boxes'][1:]
        assert spec['geometry']['wall_boxes'][0] not in [pilot.geometry(g)['wall_boxes'][0] for g in pilot.GEOMETRIES]
        assert pack.robot==pilot.pack(pilot.specification(pilot.TRIALS[0])).robot
        assert len(pack.static_objects)==5 and pack.physics_seed==family.PHYSICS_SEED


def test_mirrors_share_dimensions_and_reverse_only_panel_y():
    for cluster in family.CLUSTERS:
        a=family.geometry(cluster+'_left_open')['wall_boxes'][0]
        b=family.geometry(cluster+'_right_open')['wall_boxes'][0]
        assert a['size_xyz']==b['size_xyz']
        np.testing.assert_array_equal(np.asarray(a['centre_xyz'])*[1,-1,1],b['centre_xyz'])


def test_candidate_action_is_absent_from_scene_and_does_not_change_physical_spec():
    for name in family.layouts():
        specs=[family.specification(t) for t,c in family.assignments().items()
            if c['geometry']==name and c['appearance_seed']==family.APPEARANCES[0]]
        for s in specs:
            assert 'action' not in s and 'command' not in s
        normalized=[{k:v for k,v in s.items() if k not in ('trial','scene_id')} for s in specs]
        assert all(s==normalized[0] for s in normalized)
    assert family.candidate_commands is pilot.candidate_commands and family.decision is pilot.decision


def test_modified_spec_and_unknown_episode_rejected():
    spec=deepcopy(family.specification(family.TRIALS[0]));spec['geometry']['wall_boxes'][0]['size_xyz'][2]+=.01
    with pytest.raises(ValueError):family.pack(spec)
    with pytest.raises(ValueError):family.specification('episode_000')

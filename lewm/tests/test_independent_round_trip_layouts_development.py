from copy import deepcopy
import numpy as np
import pytest
from lewm import independent_round_trip_layouts_development as mod


@pytest.fixture(scope='module')
def inventory():return mod.build_inventory()


def test_exact_new_graphs_disjoint_from_explicit_source_registry(inventory):
    assert inventory==mod.build_inventory()
    assert len(inventory['prior_source_layouts'])==48 and len(inventory['layouts'])==8
    previous={r['abstract_topology_code'] for r in inventory['prior_source_layouts']}
    new={r['evaluation_layout']['abstract_topology_code'] for r in inventory['layouts']}
    assert len(new)==8 and not previous&new
    assert inventory['accepted_abstract_topology_groups']==8
    assert inventory['candidates_examined']==8+len(inventory['structural_rejections'])
    assert inventory['selection_used_runtime_outcomes'] is False
    assert inventory['physical_geometry_verified'] is False and inventory['final_evaluation'] is False
    for i,row in enumerate(inventory['layouts']):
        links=[tuple(map(tuple,e)) for e in row['evaluation_layout']['edges']]
        assert len(links)==15 and {c for e in links for c in e}==set(mod.CELLS)
        assert sum(mod.START in e for e in links)==1
        assert mod.edge(mod.START,(0,0)) in links
        identity=mod.identities(links)
        assert all(row['evaluation_layout'][k]==v for k,v in identity.items())
        route=row['evaluation_layout']['shortest_outbound_route']
        assert len(route)>=7 and all(mod.edge(tuple(a),tuple(b)) in links for a,b in zip(route,route[1:]))
        turns=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(route,route[1:])]
        assert sum(a!=b for a,b in zip(turns,turns[1:]))>=2
        assert row['layout_index']==i and row['procedural_seed']==mod.PHYSICS_SEED_BASE+i


def box_distance(a,b,wall):
    lo=np.asarray(wall['centre_xyz'][:2])-np.asarray(wall['size_xyz'][:2])/2
    hi=np.asarray(wall['centre_xyz'][:2])+np.asarray(wall['size_xyz'][:2])/2
    return np.linalg.norm(np.maximum(np.maximum(lo-np.maximum(a,b),np.minimum(a,b)-hi),0.))


def test_every_wall_opening_and_closed_edge_matches_graph(inventory):
    for spec in inventory['layouts']:
        links={mod.edge(tuple(a),tuple(b)) for a,b in spec['evaluation_layout']['edges']}
        walls=spec['geometry']['wall_boxes'];assert len({r['wall_id'] for r in walls})==len(walls)
        for a in mod.CELLS:
            for dx,dy in ((1,0),(0,1)):
                b=(a[0]+dx,a[1]+dy)
                if b not in mod.CELLS:continue
                distance=min(box_distance(np.array(a)*mod.PITCH_M,np.array(b)*mod.PITCH_M,w) for w in walls)
                if mod.edge(a,b) in links:assert distance>.45
                else:assert distance==0


def test_coordinate_only_public_missions_and_unchanged_pack_settings(inventory):
    for i,spec in enumerate(inventory['layouts']):
        public=mod.public_mission(i);before=deepcopy(spec);pack=mod.pack(spec)
        assert set(public)=={'goal_initial_body_xy_m','return_initial_body_xy_m','require_return_after_goal'}
        assert public['return_initial_body_xy_m']==[0.,0.] and public['require_return_after_goal']
        assert spec==before and pack.robot.spawn_xyz_m==(-1.3,0.,.375)
        assert pack.robot.spawn_quat_wxyz==(1.,0.,0.,0.) and pack.camera.near_m==.005
        assert pack.physics_seed==mod.PHYSICS_SEED_BASE+i
        assert pack.visual_seed==mod.APPEARANCE_SEED_BASE+i
        assert len(pack.static_objects)==len(spec['geometry']['wall_boxes'])
        assert pack.world_bounds_xy_m==((-2.1,-2.1),(3.4,3.4))


@pytest.mark.parametrize('fault',['wall','seed','route','identity','subset','order'])
def test_tampered_or_selected_inventory_rejected(inventory,fault):
    bad=deepcopy(inventory)
    if fault=='wall':bad['layouts'][0]['geometry']['wall_boxes'].pop()
    if fault=='seed':bad['layouts'][0]['procedural_seed']+=1
    if fault=='route':bad['layouts'][0]['evaluation_layout']['shortest_outbound_route'].pop()
    if fault=='identity':bad['layouts'][0]['evaluation_layout']['abstract_topology_sha256']='fake'
    if fault=='subset':bad['layouts'].pop()
    if fault=='order':bad['layouts'][0],bad['layouts'][1]=bad['layouts'][1],bad['layouts'][0]
    with pytest.raises(ValueError,match='exact'):mod.validate_inventory(bad)


@pytest.mark.parametrize('index',[-1,8,True,1.0])
def test_no_implicit_layout_or_runtime_resampling(index):
    with pytest.raises(ValueError):mod.specification(index)


def test_structural_candidate_budget_exhaustion_cannot_return_partial_set(monkeypatch):
    monkeypatch.setattr(mod,'MAXIMUM_CANDIDATES',1)
    with pytest.raises(ValueError,match='exhausted'):mod.build_inventory()

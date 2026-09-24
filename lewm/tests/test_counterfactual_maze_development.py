import numpy as np
import pytest

from lewm.counterfactual_maze_development import ACTIONS,branch_spec,corpus,horizon_labels,topology_identity
from lewm.multijunction_routes_development import connection


def test_corpus_has_disjoint_scene_roles_and_unique_topologies():
    layouts=corpus()
    assert len(layouts)==24
    assert sum(s['data_role']=='train' for s in layouts)==16
    assert sum(s['data_role']=='validation' for s in layouts)==8
    assert len({s['topology_sha256_dihedral'] for s in layouts})==24
    for layout in layouts:
        specs=[branch_spec(layout,i) for i in range(5)]
        assert len({s['scene_id'] for s in specs})==5
        assert all(s['geometry']==layout['geometry'] and s['procedural_seed']==layout['procedural_seed']
            and s['data_role']==layout['data_role'] for s in specs)


@pytest.mark.parametrize('index',range(24))
def test_layout_connected_prefix_leaf_junction_and_physical_openings(index):
    layout=corpus()[index]
    links=[connection(*edge) for edge in layout['graph_connections']]
    visited={(-1,0)}
    while True:
        expanded=visited | {b for edge in links if any(a in visited for a in edge) for b in edge}
        if expanded==visited: break
        visited=expanded
    assert len(visited)==16
    assert sum((-1,0) in e for e in links)==1
    assert sum((0,0) in e for e in links)>=3
    walls=layout['geometry']['wall_boxes']
    assert len({w['wall_id'] for w in walls})==len(walls)
    for a,b in links:
        midpoint=(np.array(a)+b)*layout['cell_pitch_m']/2
        assert not any(np.all(np.abs(midpoint-np.array(w['centre_xyz'][:2]))<np.array(w['size_xyz'][:2])/2) for w in walls)


def test_topology_identity_is_rotation_invariant():
    links=corpus()[0]['graph_connections']
    rotated=[[(1-c[1],c[0]) for c in edge] for edge in links]
    assert topology_identity(links)==topology_identity(rotated)


def raw_trace(last_time,contact=False):
    ns=np.arange(2_000_000,last_time+1,2_000_000)
    poses=np.tile([0,0,.3,0,0,0,1.],(len(ns),1))
    poses[:,0]=ns/1e9*.2
    flags=np.zeros(len(ns),dtype=bool)
    flags[-1]=contact
    return {'timestamp_s':ns/1e9,'base_pose_world':poses,'physics_contact':flags}


def test_contact_censoring_does_not_invent_future_motion():
    raw=raw_trace(750_000_000,contact=True)
    labels=horizon_labels(raw,49)  # branch origin at 0.1 s
    assert labels[0]['motion_valid'] and labels[0]['delta_xy_yaw_start_body'][0]==pytest.approx(.1)
    assert not labels[0]['contact_by_horizon']
    assert not labels[1]['motion_valid'] and labels[1]['delta_xy_yaw_start_body'] is None
    assert labels[1]['contact_valid'] and labels[1]['contact_by_horizon']


def test_noncontact_stop_leaves_future_contact_unknown():
    labels=horizon_labels(raw_trace(750_000_000),49)
    assert not labels[1]['motion_valid'] and not labels[1]['contact_valid']

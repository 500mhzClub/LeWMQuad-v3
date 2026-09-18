from copy import deepcopy
import numpy as np
import pytest
from lewm.novel_maze_round_trip_scene_development import (
    CELLS, START, PITCH_M, LAYOUT_COUNT, graph, edge, evaluator_route,
    specification, public_mission, pack)
from lewm.counterfactual_maze_development import corpus as older_source_corpus, topology_identity
from lewm.online_choice_maze_pilot_development import corpus as pilot_source_corpus


def test_fixed_topologies_are_connected_and_disjoint_from_32_prior_source_layouts():
    # These two reviewed generators only construct dictionaries; no runtime
    # artifact, manifest, checkpoint or protected material is read.
    previous = {s['topology_sha256_dihedral'] for s in older_source_corpus()+pilot_source_corpus()}
    current = {topology_identity(graph(i)) for i in range(LAYOUT_COUNT)}
    assert len(previous) == 32 and len(current) == LAYOUT_COUNT and not current & previous
    for i in range(LAYOUT_COUNT):
        links = graph(i); route = evaluator_route(links)
        assert len(links) == 15 and len({c for e in links for c in e}) == 16
        assert sum(START in e for e in links) == 1
        assert len(route) >= 7 and all(edge(a, b) in links for a, b in zip(route, route[1:]))
        directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(route, route[1:])]
        assert sum(a != b for a, b in zip(directions, directions[1:])) >= 2


def box_distance(a, b, wall):
    lo = np.asarray(wall['centre_xyz'][:2])-np.asarray(wall['size_xyz'][:2])/2
    hi = np.asarray(wall['centre_xyz'][:2])+np.asarray(wall['size_xyz'][:2])/2
    # The graph edges are axis-aligned, so their degenerate AABB is exact.
    return np.linalg.norm(np.maximum(np.maximum(lo-np.maximum(a, b), np.minimum(a, b)-hi), 0.))


def test_native_wall_boxes_preserve_every_graph_opening_and_close_every_nonedge():
    for i in range(LAYOUT_COUNT):
        s = specification(i); links = graph(i); walls = s['geometry']['wall_boxes']
        assert len({w['wall_id'] for w in walls}) == len(walls)
        for a in CELLS:
            for dx, dy in ((1, 0), (0, 1)):
                b = (a[0]+dx, a[1]+dy)
                if b not in CELLS: continue
                distances = [box_distance(np.array(a)*PITCH_M, np.array(b)*PITCH_M, w) for w in walls]
                if edge(a, b) in links: assert min(distances) > .45
                else: assert min(distances) == 0.


def test_public_mission_excludes_topology_and_scene_pack_uses_declared_physics():
    for i in range(LAYOUT_COUNT):
        s = specification(i); before = deepcopy(s); m = public_mission(i); p = pack(s)
        assert set(m) == {'goal_initial_body_xy_m', 'return_initial_body_xy_m', 'require_return_after_goal'}
        assert np.max(np.abs(m['goal_initial_body_xy_m'])) < 4.9
        assert m['return_initial_body_xy_m'] == [0., 0.] and m['require_return_after_goal']
        assert s == before and p.robot.spawn_xyz_m == (-PITCH_M, 0., .375)
        assert p.robot.spawn_quat_wxyz == (1., 0., 0., 0.)
        assert len(p.static_objects) == len(s['geometry']['wall_boxes'])
        assert p.camera.near_m == .005 and p.physics_seed == s['procedural_seed']


@pytest.mark.parametrize('index', [True, -1, LAYOUT_COUNT, 1.0])
def test_no_implicit_layout_or_resampling(index):
    with pytest.raises(ValueError): specification(index)

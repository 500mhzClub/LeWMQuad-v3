import numpy as np
import pytest

from lewm.multijunction_routes_development import MOTIFS,WIDTHS,route_spec,connection


@pytest.mark.parametrize('motif',MOTIFS)
@pytest.mark.parametrize('width',WIDTHS)
def test_routes_have_shared_scene_correct_ports_and_branching(motif,width):
    spec=route_spec(motif,width)
    walls=spec['geometry']['wall_boxes']
    assert len({w['wall_id'] for w in walls})==len(walls)
    assert all(g['wall_boxes']==walls for g in spec['route_geometries'])
    links=[connection(*edge) for edge in spec['graph_connections']]
    degrees={tuple(c):sum(tuple(c) in edge for edge in links) for c in spec['route_cells']}
    assert max(degrees.values())>=3
    for i,g in enumerate(spec['route_geometries']):
        edge=g['selected_directed_edge']
        opening=np.asarray(edge['opening_segment_world'])
        normal=np.asarray(edge['opening_normal_world'])
        assert np.linalg.norm(opening[1]-opening[0])==pytest.approx(width)
        assert np.linalg.norm(normal)==pytest.approx(1)
        assert np.dot(opening[1]-opening[0],normal)==pytest.approx(0)
        s,t=np.array(spec['route_cells'][i:i+2])*spec['cell_pitch_m']
        assert opening.mean(axis=0)==pytest.approx((s+t)/2)
        assert connection(spec['route_cells'][i],spec['route_cells'][i+1]) in links
        # Every declared crossing midpoint must be physically open.
        for wall in walls:
            delta=np.abs(opening.mean(axis=0)-np.array(wall['centre_xyz'][:2]))
            assert not np.all(delta < np.array(wall['size_xyz'][:2])/2)


def test_hairpin_adjacent_nonconnected_cells_are_separated():
    spec=route_spec('hairpin',1.2)
    midpoint=np.array([0,spec['cell_pitch_m']/2])
    assert any(np.allclose(w['centre_xyz'][:2],midpoint) for w in spec['geometry']['wall_boxes'])


def test_dead_end_return_is_reverse_directed_not_an_inferred_reverse():
    spec=route_spec('dead_end_return',1.2)
    before,after=spec['route_geometries'][1:3]
    assert np.asarray(before['selected_directed_edge']['opening_normal_world'])==pytest.approx(-np.asarray(after['selected_directed_edge']['opening_normal_world']))
    assert len(after['teacher_route_polyline_world'])==3
    assert len(spec['route_geometries'])==4


def test_eight_fresh_identities_and_fixed_seed_population():
    specs=[route_spec(m,w) for m in MOTIFS for w in WIDTHS]
    assert len({s['scene_id'] for s in specs})==8
    assert [s['procedural_seed'] for s in specs]==list(range(2026091000,2026091008))

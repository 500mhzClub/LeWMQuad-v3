"""Exact isomorphism, prospective roles and complete context/action construction."""
from collections import Counter
from copy import deepcopy
from itertools import permutations
import math
import random
import numpy as np
import pytest
from lewm.independent_layout_identity_development import canonical_graph,grid_edges,metric_identity,topology_identity,verify_role_separation
from lewm.independent_layout_inventory_development import build_inventory,validate_inventory,cell_types,CONTEXTS,HISTORIES,SUPPORTS


def permute(graph,order):
    inverse={v:i for i,v in enumerate(order)}
    return [sorted(inverse[j] for j in graph[v]) for v in order]


def brute_isomorphic(a,b):
    if len(a)!=len(b):return False
    for order in permutations(range(len(a))):
        if any(len(a[order[i]])!=len(b[i]) for i in range(len(a))):continue
        if all((order[j] in a[order[i]])==(j in b[i]) for i in range(len(a)) for j in range(i)):return True
    return False


def test_regular_graphs_not_collapsed_by_degree_or_refinement_signature():
    prism=[{1,2,3},{0,2,4},{0,1,5},{0,4,5},{1,3,5},{2,3,4}]
    bipartite=[{3,4,5} for _ in range(3)]+[{0,1,2} for _ in range(3)]
    assert not brute_isomorphic(prism,bipartite)
    assert canonical_graph(prism)['code']!=canonical_graph(bipartite)['code']
    for graph in (prism,bipartite):
        expected=canonical_graph(graph)['code']
        for order in permutations(range(6)):assert canonical_graph(permute(graph,order))['code']==expected


def test_exact_codes_agree_with_independent_small_graph_permutation_oracle():
    rng=random.Random(174);graphs=[]
    for _ in range(14):
        g=[set() for _ in range(6)]
        for i in range(6):g[i].add((i+1)%6);g[(i+1)%6].add(i)
        for _ in range(rng.randrange(5)):
            a,b=rng.sample(range(6),2)
            if len(g[a])<4 and len(g[b])<4:g[a].add(b);g[b].add(a)
        graphs.append(g)
    codes=[canonical_graph(g)['code'] for g in graphs]
    for i,g in enumerate(graphs):
        for j,h in enumerate(graphs[:i]):assert (codes[i]==codes[j])==brute_isomorphic(g,h)


def test_exact_search_budget_exhaustion_never_returns_an_identity():
    with pytest.raises(ValueError,match='unresolved'):canonical_graph([{1,3},{0,2},{1,3},{0,2}],maximum_states=1)
    with pytest.raises(ValueError):canonical_graph([{1},{0},{3},{2}])
    with pytest.raises(ValueError):canonical_graph([{1},{0,2},{0,1}])


def test_rigid_grid_copies_have_same_geometry_and_abstract_topology():
    e=build_inventory()['layouts'][0]['edges'];metric=metric_identity(e)['code'];top=topology_identity(e)['code']
    for swap in (False,True):
        for sx in (-1,1):
            for sy in (-1,1):
                move=lambda p:[sx*p[1]+17,sy*p[0]-23] if swap else [sx*p[0]+17,sy*p[1]-23]
                transformed=[[move(b),move(a)] for a,b in e[::-1]]
                assert metric_identity(transformed)['code']==metric and topology_identity(transformed)['code']==top


def test_different_bends_of_same_abstract_path_cannot_cross_roles():
    straight=[((0,0),(1,0)),((1,0),(2,0)),((2,0),(3,0))]
    bend=[((0,0),(1,0)),((1,0),(1,1)),((1,1),(1,2))]
    assert topology_identity(straight)['code']==topology_identity(bend)['code']
    assert metric_identity(straight)['code']!=metric_identity(bend)['code']
    with pytest.raises(ValueError,match='cannot cross'):
        verify_role_separation([dict(edges=straight,pitch_m=1.2,role='train'),dict(edges=bend,pitch_m=1.2,role='development_eval')])
    assert verify_role_separation([dict(edges=straight,pitch_m=1.2,role='train'),dict(edges=bend,pitch_m=1.2,role='train')])['topology_groups']==1


@pytest.mark.parametrize('edges',[
    [((0,0),(2,0))],[((0,0),(1,1))],[((0,0),(1,0)),((1,0),(0,0))],
    [((0,0),(1,0)),((3,0),(4,0))],[((False,0),(1,0))]])
def test_invalid_grid_passages_rejected(edges):
    with pytest.raises(ValueError):grid_edges(edges)


def test_inventory_is_prospective_balanced_and_graph_distinct():
    x=build_inventory();assert x==build_inventory();assert len(x['layouts'])==12 and len(x['episodes'])==1440
    assert x['role_counts']==dict(train=6,selection=3,development_eval=3)
    assert x['episode_role_counts']==dict(train=720,selection=360,development_eval=360)
    assert x['role_separation']['topology_groups']==x['role_separation']['metric_groups']==12
    assert not x['physics_collected'] and not x['model_trained'] and not x['final_evaluation']
    for role,count in (('train',2),('selection',1),('development_eval',1)):
        assert Counter(l['cycle_rank'] for l in x['layouts'] if l['role']==role)=={1:count,2:count,3:count}
    for layout in x['layouts']:
        assert len(layout['cells'])==16 and len(layout['edges'])-16+1==layout['cycle_rank']
        assert all(cell_types(layout['edges']).values())
        graph=[sorted(layout['cells'].index(list(q)) for q in adj) for _,adj in sorted(grid_edges(layout['edges'])[1].items())]
        shuffled=list(range(16));random.Random(8).shuffle(shuffled)
        assert canonical_graph(permute(graph,shuffled))['code']==layout['topology_canonical_code']


def test_every_context_has_all_actions_with_matched_construction_and_explicit_history():
    x=build_inventory();groups={}
    for ep in x['episodes']:
        key=tuple(ep[k] for k in ('layout_id','context_kind','history_kind','support'))
        groups.setdefault(key,[]).append(ep)
        assert ep['departure_ns']==2_300_000_000 and ep['departure_tick']==ep['warmup_ticks']==8
        assert ep['brake_ticks']==20 and ep['pulse_ticks']==(2,5)[ep['action_index']%2]
        assert ep['warmup_command']==([0.,0.,0.] if ep['history_kind']=='quiet' else [.12,0.,0.])
    assert len(groups)==12*len(CONTEXTS)*len(HISTORIES)*len(SUPPORTS)
    for episodes in groups.values():
        assert {e['action_index'] for e in episodes}==set(range(6))
        for name in ('physics_seed','appearance_seed','spawn_se2_world','spawn_z_m','warmup_command','friction_mu','role'):
            assert all(e[name]==episodes[0][name] for e in episodes)
    assert len({e['episode_id'] for e in x['episodes']})==1440
    x['episodes'][0]['spawn_se2_world'][0]+=1
    assert x['episodes'][0]['spawn_se2_world']!=x['episodes'][1]['spawn_se2_world']


def test_wall_roster_closes_exactly_nonpassage_boundaries():
    for layout in build_inventory()['layouts']:
        edges,adj=grid_edges(layout['edges']);passages={frozenset(e) for e in edges};walls=layout['wall_boxes']
        assert len({w['wall_id'] for w in walls})==len(walls)
        for x,y in adj:
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                mid=[(2*x+dx)*.6,(2*y+dy)*.6]
                matches=[w for w in walls if np.allclose(w['centre_xyz'][:2],mid,rtol=0,atol=1e-10)]
                assert len(matches)==(0 if frozenset(((x,y),(x+dx,y+dy))) in passages else 1)


def test_declared_spawns_face_the_context_geometry_and_are_not_setup_penetrations():
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.setup_snapshot_evaluation_development import check_setup_snapshot
    from scripts.pulse_context_setup_development import context_priors
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    geom=ArticulatedCollisionGeometry(URDF);q=np.repeat([0.,.8,-1.5],4);v,region=context_priors('0'*64,geom,q)
    for layout in build_inventory()['layouts']:
        _,adj=grid_edges(layout['edges']);walls=[dict(native_name=w['wall_id'],native_position=w['centre_xyz'],native_box_size=w['size_xyz']+[0.]*4,
            native_quaternion_wxyz=[1.,0.,0.,0.],native_collision_boxes=1,collision_enabled=True,fixed=True) for w in layout['wall_boxes']]
        for ctx in layout['contexts']:
            p=tuple(ctx['cell']);dx,dy=ctx['forward_grid'];ahead=(p[0]+dx,p[1]+dy)
            assert (ahead in adj[p])==(ctx['kind'] not in ('dead_end_approach','near_wall'))
            x,y,a=ctx['spawn_se2_world'];R=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
            score=check_setup_snapshot(v,region,identity=(0,0,0),measured_ns=1_500_000_000,position_world_m=[x,y,.35],
                rotation_world_from_initial_body=R,velocity_world_m_s=[0.,0.,0.],native_static_boxes=walls,
                expected_nonfloor_names=[w['native_name'] for w in walls],geometry=geom,joint_position=q)
            assert score['velocity_and_nonfloor_setup_checks_pass']
            # Synthetic nominal pose feasibility is NOT actual settled support.
            assert not score['support_established'] and not score['navigation_qualified']


@pytest.mark.parametrize('fault',['role','seed','action_bool','wall','context','missing','duplicate'])
def test_exact_manifest_rejects_mutated_roles_geometry_or_episode_schema(fault):
    x=build_inventory()
    if fault=='role':x['layouts'][0]['role']='development_eval'
    elif fault=='seed':x['episodes'][0]['physics_seed']+=1
    elif fault=='action_bool':x['episodes'][1]['action_index']=True
    elif fault=='wall':x['layouts'][0]['wall_boxes'][0]['centre_xyz'][0]+=.1
    elif fault=='context':x['episodes'][0]['warmup_command'][0]=.1
    elif fault=='missing':x['episodes'].pop()
    elif fault=='duplicate':x['episodes'][-1]=deepcopy(x['episodes'][0])
    with pytest.raises(ValueError):validate_inventory(x)


def test_independent_pairwise_bijection_checks_all66_pairs_and_catches_disguised_duplicate():
    from lewm.independent_layout_inventory_audit_development import audit_inventory,pairwise_isomorphic
    x=build_inventory();report=audit_inventory(x)
    assert report['layout_pairs_checked']==66 and report['planned_six_action_groups']==240
    assert not report['actual_prefix_matching_verified'] and not report['native_setup_verified']
    e=x['layouts'][0]['edges'];moved=[[[a[1]+8,-a[0]+3],[b[1]+8,-b[0]+3]] for a,b in e]
    assert pairwise_isomorphic(e,moved)['isomorphic']
    x['layouts'][-1]['edges']=moved
    with pytest.raises(ValueError,match='duplicate'):audit_inventory(x)
    with pytest.raises(ValueError,match='unresolved'):pairwise_isomorphic(e,moved,maximum_states=1)


def test_manifest_runner_is_source_only_and_does_not_launch_simulator():
    import inspect
    from scripts import build_go2_independent_layout_inventory_v1 as runner
    source=inspect.getsource(runner)
    assert 'initialize_genesis' not in source and 'command_tick(' not in source and 'optimizer' not in source
    assert runner.OUTPUT.name=='go2_independent_layout_inventory_v1_attempt_001'

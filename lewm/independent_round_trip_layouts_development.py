"""New development maze construction; topology and routes are evaluator-only.

Only pure reviewed source generators are consulted. This module opens no
dataset, checkpoint, runtime observation or protected benchmark material.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import random
from lewm.novel_maze_round_trip_scene_development import (
    CELLS, START, PITCH_M, WALL_THICKNESS_M, NEIGHBOURS, edge, evaluator_route,
    graph as old_graph, LAYOUT_COUNT as OLD_COUNT)
from lewm.counterfactual_maze_development import corpus as source_corpus
from lewm.online_choice_maze_pilot_development import corpus as choice_corpus
from lewm.independent_layout_inventory_development import build_inventory as context_inventory
from lewm.independent_layout_identity_development import topology_identity, metric_identity
from lewm_genesis.scene_loader import StaticObject
from scripts.probe_go2_rgbd_motion_scene_development_v1 import pack as base_pack

LAYOUT_COUNT=8
CONSTRUCTION_SEED=2026091027
MAXIMUM_CANDIDATES=10000
PHYSICS_SEED_BASE=2026092700
APPEARANCE_SEED_BASE=2026092800


def prior_graphs():
    """Explicit48-layout source registry, not a claim about every past maze."""
    rows=[dict(name=r['layout_id'],edges=r['graph_connections']) for r in source_corpus()+choice_corpus()]
    rows.extend(dict(name=f'novel_round_trip_v1_{i:02d}',edges=old_graph(i)) for i in range(OLD_COUNT))
    rows.extend(dict(name=r['layout_id'],edges=r['edges']) for r in context_inventory()['layouts'])
    if len(rows)!=48 or len({r['name'] for r in rows})!=48:
        raise ValueError('complete explicit48-layout source registry required')
    return rows


def candidate(rng):
    parent={c:c for c in CELLS};links={edge(START,(0,0))}
    def root(c):
        while parent[c]!=c:c=parent[c]
        return c
    parent[root(START)]=root((0,0))
    choices=[edge(c,(c[0]+dx,c[1]+dy)) for c in CELLS for dx,dy in ((1,0),(0,1))
        if (c[0]+dx,c[1]+dy) in parent and START not in (c,(c[0]+dx,c[1]+dy))]
    rng.shuffle(choices)
    for a,b in choices:
        if root(a)!=root(b):links.add(edge(a,b));parent[root(a)]=root(b)
    if len(links)!=15 or len({root(c) for c in CELLS})!=1:
        raise ValueError('connected sixteen-cell tree required')
    return tuple(sorted(links))


def identities(links):
    top=topology_identity(links)
    metric=metric_identity(links,pitch_m=PITCH_M,wall_thickness_m=WALL_THICKNESS_M,wall_height_m=1.4)
    return dict(abstract_topology_sha256=top['sha256'],abstract_topology_code=top['code'],
        exact_canonicalization_states=top['search_states'],metric_sha256=metric['sha256'],metric_code=metric['code'])


def make_spec(index,links,identity,candidate_index):
    route=evaluator_route(links);walls={}
    for x,y in CELLS:
        for dx,dy in NEIGHBOURS:
            if edge((x,y),(x+dx,y+dy)) in links:continue
            mx,my=2*x+dx,2*y+dy;key=(mx,my,abs(dx))
            walls[key]=dict(wall_id=f'independent_round_trip_wall_{mx}_{my}_{abs(dx)}',
                centre_xyz=[mx*PITCH_M/2,my*PITCH_M/2,.7],
                size_xyz=[WALL_THICKNESS_M,PITCH_M+WALL_THICKNESS_M,1.4] if dx
                    else [PITCH_M+WALL_THICKNESS_M,WALL_THICKNESS_M,1.4],
                yaw_rad=0.,material_id='NEUTRAL_WALL')
    return dict(scene_id=f'independent-round-trip-development-v1-{index:02d}',layout_index=index,
        family='INDEPENDENT_ROUND_TRIP_DEVELOPMENT',data_role='prospective_navigation_development',
        procedural_seed=PHYSICS_SEED_BASE+index,appearance_seed=APPEARANCE_SEED_BASE+index,
        appearance_arm='distinctive',friction_mu=1.,render_near_m=.005,
        visual_surface_contract='floor_first_variable_height_union_walls',
        geometry=dict(spawn_se2_world=[START[0]*PITCH_M,START[1]*PITCH_M,0.],
            wall_boxes=[walls[k] for k in sorted(walls)]),
        evaluation_layout=dict(cells=[list(c) for c in CELLS],edges=[[list(a),list(b)] for a,b in links],
            goal_cell=list(route[-1]),shortest_outbound_route=[list(c) for c in route],
            shortest_outbound_distance_m=(len(route)-1)*PITCH_M,pitch_m=PITCH_M,**identity),
        construction_candidate_index=candidate_index,native_execution=False,final_evaluation=False,navigation_qualified=False)


def build_inventory():
    previous=[dict(name=r['name'],**identities(r['edges'])) for r in prior_graphs()]
    old_top={r['abstract_topology_code'] for r in previous};old_metric={r['metric_code'] for r in previous}
    accepted_top=set();accepted_metric=set();rng=random.Random(CONSTRUCTION_SEED)
    layouts=[];rejections=[]
    for number in range(MAXIMUM_CANDIDATES):
        links=candidate(rng);identity=identities(links);route=evaluator_route(links)
        directions=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(route,route[1:])]
        turns=sum(a!=b for a,b in zip(directions,directions[1:]))
        reason=None
        if len(route)<7 or turns<2:reason='insufficient_declared_route_structure'
        elif identity['abstract_topology_code'] in old_top:reason='prior_abstract_topology'
        elif identity['metric_code'] in old_metric:reason='prior_grid_embedding'
        elif identity['abstract_topology_code'] in accepted_top:reason='repeated_abstract_topology'
        elif identity['metric_code'] in accepted_metric:reason='repeated_grid_embedding'
        if reason:
            rejections.append(dict(candidate_index=number,reason=reason,**identity));continue
        accepted_top.add(identity['abstract_topology_code']);accepted_metric.add(identity['metric_code'])
        layouts.append(make_spec(len(layouts),links,identity,number))
        if len(layouts)==LAYOUT_COUNT:break
    if len(layouts)!=LAYOUT_COUNT:raise ValueError('bounded structural construction exhausted; no runtime replacement')
    return dict(schema='independent_round_trip_layout_inventory_development.v1',construction_seed=CONSTRUCTION_SEED,
        maximum_candidates=MAXIMUM_CANDIDATES,candidates_examined=number+1,prior_source_layouts=previous,
        prior_source_layout_count=len(previous),prior_abstract_topology_groups=len(old_top),
        prior_embedding_comparison_dimensions=dict(pitch_m=PITCH_M,wall_thickness_m=WALL_THICKNESS_M,wall_height_m=1.4,
            actual_legacy_metric_geometry_reconstructed=False),
        layouts=layouts,structural_rejections=rejections,rejection_counts=dict(Counter(r['reason'] for r in rejections)),
        accepted_abstract_topology_groups=len(accepted_top),accepted_grid_embedding_groups=len(accepted_metric),
        disjoint_from_explicit_prior_registry=True,selection_used_runtime_outcomes=False,
        collection_performed=False,model_loaded=False,native_execution=False,final_evaluation=False,
        physical_geometry_verified=False,navigation_qualified=False,goal_achieved=False)


def specification(index):
    if type(index) is not int or not 0<=index<LAYOUT_COUNT:
        raise ValueError('one of the eight fixed new development layouts required')
    return build_inventory()['layouts'][index]


def public_mission(index):
    spec=specification(index);goal=spec['evaluation_layout']['goal_cell']
    return dict(goal_initial_body_xy_m=[(goal[i]-START[i])*PITCH_M for i in range(2)],
        return_initial_body_xy_m=[0.,0.],require_return_after_goal=True)


def validate_inventory(inventory):
    if inventory!=build_inventory():raise ValueError('complete exact source-defined new layout inventory required')


def pack(spec):
    if spec!=specification(spec['layout_index']):raise ValueError('exact new development layout specification required')
    original=base_pack()
    objects=tuple(StaticObject(object_id=b['wall_id'],kind='wall',center_xyz_m=tuple(b['centre_xyz']),
        size_xyz_m=tuple(b['size_xyz']),yaw_rad=b['yaw_rad'],material_id=b['material_id'])
        for b in spec['geometry']['wall_boxes'])
    identity=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return replace(original,scene_id=spec['scene_id'],family=spec['family'],
        difficulty_tier='MULTIJUNCTION_OUTBOUND_AND_RETURN',manifest_sha256=identity,
        static_objects=objects,physics_seed=spec['procedural_seed'],topology_seed=spec['procedural_seed'],
        visual_seed=spec['appearance_seed'],world_bounds_xy_m=((-2.1,-2.1),(3.4,3.4)),
        camera=replace(original.camera,near_m=.005),
        robot=replace(original.robot,spawn_xyz_m=(START[0]*PITCH_M,START[1]*PITCH_M,.375),
            spawn_quat_wxyz=(1.,0.,0.,0.)))

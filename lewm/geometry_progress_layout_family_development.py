"""Source-only next collection: disjoint training and geometry-transfer layouts.

Four parameter clusters, each with a mirrored pair, two matched appearances,
and the unchanged six-action bank. Mirrored layouts are dependent siblings;
these are local obstruction tasks, not independent complete maze evaluations.
No generated inputs, physics, training or evaluation are performed by import.
"""
from dataclasses import replace
import hashlib
import json
import random
from lewm.geometry_progress_pilot_development import (
    ACTIONS,geometry as pilot_geometry,specification as pilot_specification,
    pack as pilot_pack,TRIALS as PILOT_TRIALS,candidate_commands,timed_candidate,
    schedule,decision,progress_outcome,WARMUP_TICKS,HORIZON_TICKS)
from lewm_genesis.scene_loader import StaticObject

# Millimetre dimensions fixed in source before any new native outcomes.
# (role, panel x, length, signed inner edge, height)
CLUSTERS={
    'cluster_00':('train',630,1020,-60,760),
    'cluster_01':('train',690,940,-20,680),
    'cluster_02':('geometry_transfer',660,1100,-90,820),
    'cluster_03':('geometry_transfer',720,1060,-40,640),
}
APPEARANCES=(2026090940,2026090941)
PHYSICS_SEED=2026090910


def layouts():
    rows={}
    for cluster,(role,x,length,inner,height) in CLUSTERS.items():
        for side in ('left_open','right_open'):
            name=f'{cluster}_{side}'
            rows[name]=dict(layout_id=name,cluster=cluster,role=role,opening=side,
                panel_x_mm=x,panel_length_mm=length,panel_inner_edge_mm=inner,panel_height_mm=height)
    return rows


def assignments():
    rows=[dict(geometry=name,cluster=cell['cluster'],data_role=cell['role'],
        opening=cell['opening'],appearance_seed=seed,action=a)
        for name,cell in layouts().items() for seed in APPEARANCES for a in ACTIONS]
    random.Random(2026090942).shuffle(rows)
    return {f'family_episode_{i:03d}':r for i,r in enumerate(rows)}


TRIALS=tuple(assignments())


def geometry(name):
    if name not in layouts():raise ValueError('exact prospective family layout required')
    cell=layouts()[name];g=pilot_geometry(cell['opening']);panel=g['wall_boxes'][0]
    sign=-1 if cell['opening']=='left_open' else 1
    panel['centre_xyz']=[cell['panel_x_mm']/1000,
        sign*(cell['panel_length_mm']/2+cell['panel_inner_edge_mm'])/1000,cell['panel_height_mm']/2000]
    panel['size_xyz']=[.08,cell['panel_length_mm']/1000,cell['panel_height_mm']/1000]
    return g


def specification(trial):
    if trial not in TRIALS:raise ValueError('exact randomized family episode required')
    cell=assignments()[trial]
    return pilot_specification(PILOT_TRIALS[0])|dict(scene_id='geometry-progress-family-v1-'+trial,
        family='GEOMETRY_PROGRESS_LAYOUT_TRANSFER_DEVELOPMENT',trial=trial,
        layout_id=cell['geometry'],data_role=cell['data_role'],procedural_seed=PHYSICS_SEED,
        appearance_seed=cell['appearance_seed'],geometry=geometry(cell['geometry']),
        render_near_m=.005,visual_surface_contract='floor_first_variable_height_union_walls')


def pack(spec):
    if spec!=specification(spec['trial']):raise ValueError('exact new family specification required')
    base=pilot_pack(pilot_specification(PILOT_TRIALS[0]))
    objects=tuple(StaticObject(object_id=b['wall_id'],kind='wall',center_xyz_m=tuple(b['centre_xyz']),
        size_xyz_m=tuple(b['size_xyz']),yaw_rad=b['yaw_rad'],material_id=b['material_id']) for b in spec['geometry']['wall_boxes'])
    return replace(base,scene_id=spec['scene_id'],family=spec['family'],static_objects=objects,
        physics_seed=PHYSICS_SEED,topology_seed=PHYSICS_SEED,visual_seed=spec['appearance_seed'],
        camera=replace(base.camera,near_m=.005),
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())


def cohort_contract():
    return dict(episodes=len(TRIALS),layouts=8,parameter_clusters=4,
        training_layouts=4,training_parameter_clusters=2,training_episodes=48,
        geometry_transfer_layouts=4,geometry_transfer_parameter_clusters=2,geometry_transfer_episodes=48,
        matched_appearance_seeds=list(APPEARANCES),actions=list(ACTIONS),
        mirrored_pairs_are_independent=False,independent_maze_evaluation_layouts=0,
        native_collection_launched=False,model_trained=False,navigation_qualified=False)

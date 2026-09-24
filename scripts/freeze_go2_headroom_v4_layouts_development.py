"""One geometry-only construction from the declared seed; no simulator/models."""
import hashlib
import json
from pathlib import Path
from lewm import dense_world_model_maze_layouts_development as exposed
from lewm import maze_view_transfer_layouts_development as transfer
from lewm.eligible_floor_registration_development import bind

generator=exposed.generator

def prior_graphs():
    return transfer.prior_graphs()+[dict(name=s['scene_id'],edges=s['evaluation_layout']['edges']) for s in transfer.build_inventory()['layouts']]

def make_spec(index,links,identity,candidate_index):
    spec=bind(generator.make_spec,PHYSICS_SEED_BASE=2026102800,APPEARANCE_SEED_BASE=2026102900)(index,links,identity,candidate_index)
    return spec|dict(scene_id=f'decision-headroom-v4-unexamined-{index:02d}',family='DECISION_HEADROOM_SAME_MAZE_FAMILY_DEVELOPMENT')

def main():
    new=bind(generator.build_inventory,prior_graphs=prior_graphs,make_spec=make_spec,LAYOUT_COUNT=4,CONSTRUCTION_SEED=2026092307)()
    rows=[]
    for index,spec in enumerate(exposed.build_inventory()['layouts']+new['layouts']):
        spec=spec|dict(layout_index=index)
        rows.append(dict(audit_layout=index,specification=spec,
            specification_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            exposure='exposed' if index<4 else 'runtime_unexamined_at_freeze',
            future_role=('fit_eligible' if index<2 else 'selection' if index<4 else 'evaluation_only'),
            exposure_note='Prior development/cohort; pilot also used 00 and 02' if index<4 else 'Geometry identity exposed by freeze; no runtime, model or outcome examination',
            final_sealed_benchmark=False))
    out=dict(schema='headroom_v4_frozen_layouts.v1',construction_seed=2026092307,layouts=rows,construction=new,
             physics_executed=False,model_loaded=False,scope='Same generator; novelty against explicit source registry including both readout training and transfer layouts')
    p=Path('docs/go2_decision_headroom_v4_layouts_2026-09-23.json')
    with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps([dict(layout=r['audit_layout'],sha256=r['specification_sha256'],topology=r['specification']['evaluation_layout']['abstract_topology_sha256']) for r in rows],indent=2))

if __name__=='__main__':main()

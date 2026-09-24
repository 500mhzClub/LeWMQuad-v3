#!/usr/bin/env python3
"""RGB floor baseline versus evaluation-only visible-surface geometry labels."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT,CORPUS_ROOT
from lewm.counterfactual_maze_development import corpus
from lewm.floor_visibility_reference_development import visible_floor,confusion
from lewm.rgb_floor_evidence_development import observe_floor,palette_floor_mask
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_rgb_floor_evidence_development_v1_attempt_001'
NEW_SOURCES=('lewm/rgb_floor_evidence_development.py','lewm/floor_visibility_reference_development.py',
    'lewm/tests/test_rgb_floor_evidence_development.py','scripts/analyze_go2_rgb_floor_evidence_development_v1.py',
    'docs/go2_rgb_floor_evidence_development_v1_2026-09-05.md')


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh perception diagnostic required')
    inputs={str((DERIVATION_ROOT/name).relative_to(ROOT)):sha for name,sha in {
        'result.json':'a224d2b212b7a544a34e8f7d62c03bf831f9bdf795e1fd726b26fea5b5eee76b',
        'raw_artifact_audit.json':'a172810608cc96a8268b2470ce429dfcfdefff2e8be51c8becaaa2fc1e1f7b01',
        'windows.json':'8c229c5e12b08dda2b98caac9f79b33f1b3b539a99013b802cd9ec4b3a2b6131'}.items()}
    inputs.update({str((CORPUS_ROOT/name).relative_to(ROOT)):sha for name,sha in {
        'result.json':'bb7307e4a55ee896ad91753f03c309f39d37e7ec497fd95d7b59cf098d387015',
        'raw_artifact_audit.json':'fea689e4211b2bff02d76149c60150e9ca65b2c2a6ccc66879dcecfec8834eb6'}.items()})
    # Bind the independently reviewed source dependency witness, not any new
    # incomplete physical outcomes from the still-running collection.
    previous=ROOT/'.generated/go2_successive_choice_maze_development_v1_attempt_001'
    inputs[str((previous/'full_audit_source_dependency_witness_clock_boundary_v2.json').relative_to(ROOT))]='f91c7537ad8897fdf835545a05fe557d73acc56c4fe568ad8da8c73767e28398'
    verify(inputs)
    witness=json.loads((previous/'full_audit_source_dependency_witness_clock_boundary_v2.json').read_text())
    old_launch=json.loads((previous/'launch.json').read_text())
    sources=old_launch['source_sha256']|witness['source_sha256']|{p:digest(ROOT/p) for p in NEW_SOURCES}
    sources['lewm_worlds/lewm_worlds/randomization.py']=digest(ROOT/'lewm_worlds/lewm_worlds/randomization.py'); verify(sources)
    data=AuditedSubtrajectoryDataset(DERIVATION_ROOT,'train'); windows=json.loads((DERIVATION_ROOT/'windows.json').read_text())
    unique={}
    for row in windows:
        key=(row['context_scene_id'],row['history_observation_indices'][-1])
        unique.setdefault(key,[]).append(row)
    if len(unique)!=818: raise ValueError('fixed818 unique causal context frames expected')
    layouts={s['layout_id']:s for s in corpus()}; cameras={}
    for scene,_ in unique:
        if scene in cameras: continue
        path=data.corpus.paths[scene]/'camera_audit.json'; expected=data.corpus.members[scene]['result']['artifact_sha256']['camera_audit.json']
        if digest(path)!=expected: raise ValueError('audited camera evidence changed')
        cameras[scene]=json.loads(path.read_text()); inputs[str(path.relative_to(ROOT))]=expected
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'frames':[{'scene_id':s,'observation_index':i,'window_ids':[r['window_id'] for r in rows]} for (s,i),rows in unique.items()],
        'scope':'fixed palette RGB observation baseline and separate privileged surface labels; no training, clearance, identity or navigation claim'})
    results=[]; labels=[]; valids=[]; interiors=[]
    try:
        for (scene,index),windows in unique.items():
            row=windows[0]; member=data.corpus.members[scene]['result']; spec=layouts[row['layout_id']]
            if member['layout_id']!=row['layout_id'] or member['data_role']!=row['data_role']: raise ValueError('context crossed layout role')
            packet=load_route_observation(data.corpus.paths[scene],index); ns=packet['image']['measured_ns']
            if ns!=row['decision_ns']: raise ValueError('actual current image clock')
            truth=visible_floor(cameras[scene][index]['world_from_optical'],spec['geometry']['wall_boxes'])
            observed=observe_floor(packet,now_ns=ns); rgb=packet['image']['rgb']; gray=np.repeat(rgb.mean(-1,keepdims=True).astype(np.uint8),3,axis=-1)
            predictions={'original_rgb':observed['floor_evidence_mask'],'grayscale_control':palette_floor_mask(gray),
                'red_blue_swap_control':palette_floor_mask(rgb[...,::-1].copy())}
            metrics={}
            for name,mask in predictions.items():
                predicted=mask[np.ix_(truth['rows'],truth['columns'])]
                metrics[name]={which:confusion(predicted,truth['visible_floor'],truth[which]) for which in ('valid','interior')}
            results.append({'scene_id':scene,'observation_index':index,'decision_ns':ns,'layout_id':row['layout_id'],
                'data_role':row['data_role'],'source_rgb_sha256':member['artifact_sha256'][f'rgb_{index:04d}.png'],
                'metrics':metrics,'envelope_valid_columns':int(observed['valid_columns'].sum())})
            labels.append(truth['visible_floor']); valids.append(truth['valid']); interiors.append(truth['interior'])
            if len(results)%50==0: print(json.dumps({'event':'floor_frames_checked','completed':len(results),'planned':818}),flush=True)
        aggregates=[]
        for layout_id in layouts:
            selected=[r for r in results if r['layout_id']==layout_id]
            for variant in ('original_rgb','grayscale_control','red_blue_swap_control'):
                for scope in ('valid','interior'):
                    counts={k:sum(r['metrics'][variant][scope][k] for r in selected) for k in ('true_positive','false_positive','false_negative','true_negative')}
                    tp,fp,fn=counts['true_positive'],counts['false_positive'],counts['false_negative']
                    aggregates.append({'layout_id':layout_id,'data_role':layouts[layout_id]['data_role'],'variant':variant,'scope':scope,
                        'frames':len(selected),**counts,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None})
        np.savez_compressed(OUTPUT/'evaluation_floor_labels.npz',visible_floor=np.stack(labels),valid=np.stack(valids),
            interior=np.stack(interiors),rows=truth['rows'],columns=truth['columns'])
        verify(sources|inputs)
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','frames':len(results),'layouts':24,'frame_results':results,
            'by_layout':aggregates,'label_artifact_sha256':digest(OUTPUT/'evaluation_floor_labels.npz'),
            'launch_sha256':digest(OUTPUT/'launch.json'),'metric_clearance_qualified':False,
            'scope':'palette-specific visible-floor evidence; no place/exit identity, traversability guarantee, training or physical navigation'})
        print(json.dumps({'status':'COMPLETE','frames':len(results),'layouts':24}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'FAIL','error':str(error),'frames':len(results),'frame_results':results,
            'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()

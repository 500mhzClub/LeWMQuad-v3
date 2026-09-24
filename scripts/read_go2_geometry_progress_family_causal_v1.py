"""Complete role accounting and separately frozen moving-context derivation."""
import argparse
from collections import Counter
import json
import shutil
import numpy as np
import torch
from lewm.geometry_progress_layout_family_development import TRIALS,assignments
from lewm.geometry_progress_family_accounting_development import summarize
from lewm.geometry_progress_family_causal_windows_development import derive,materialize,OFFSETS_TICKS
from lewm.geometry_progress_family_learning_sample_development import materialize as initial_materialize
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_geometry_progress_family_v1 import OUTPUT as INPUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.geometry_progress_family_plot_development import plot_paths

OUTPUT=BASE/'go2_geometry_progress_family_causal_v1_attempt_001'
PROTOCOL='docs/go2_geometry_progress_family_causal_v1_2026-09-08.md'


def counts(windows):
    if [(r['trial'],r['offset_ticks']) for r in windows]!=[(t,o) for t in TRIALS for o in OFFSETS_TICKS]:
        raise ValueError('complete ordered 768 planned causal windows required')
    expected=assignments()
    if any(any(r.get(k)!=v for k,v in expected[r['trial']].items()) for r in windows):
        raise ValueError('causal role/cluster assignments changed')
    result={}
    for role in ('train','geometry_transfer'):
        rows=[r for r in windows if r['data_role']==role];ready=[r for r in rows if r['available']]
        targets=[t for r in ready for t in r['targets']]
        result[role]=dict(planned_episodes=48,planned_windows=len(rows),available_windows=len(ready),
            available_initial_windows=sum(r['offset_ticks']==0 for r in ready),
            available_moving_windows=sum(r['offset_ticks']>0 for r in ready),
            unavailable_reasons=dict(Counter(r['reason'] for r in rows if not r['available'])),
            planned_known_horizon_slots=sum(r['remaining_ticks']//5 for r in rows),
            recorded_target_slots=len(targets),recorded_known_horizon_slots=sum(t['in_plan'] for t in targets),
            motion_valid=sum(t['motion_valid'] for t in targets),future_image_valid=sum(t['future_image_valid'] for t in targets),
            contact_valid=sum(t['contact_valid'] for t in targets),contact_positive=sum(t['contact']==1. for t in targets),
            contact_positive_without_future_image=sum(t['contact']==1. and not t['future_image_valid'] for t in targets),
            contact_positive_without_motion=sum(t['contact']==1. and not t['motion_valid'] for t in targets))
    return result


def compare_initial(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('initial tensor schema changed')
        for k in a:compare_initial(a[k],b[k])
    else:torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--collection-result-sha256',required=True);args=parser.parse_args()
    validate_root(INPUT);validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive causal derivation; no retry/resume')
    ids={'result.json':args.collection_result_sha256};verify_artifacts(INPUT,ids);collected=read_json(INPUT,'result.json')
    if collected['status']!='GEOMETRY_PROGRESS_FAMILY_COLLECTION_AND_AUDIT_COMPLETE':raise ValueError('complete prospective family audit required')
    ids|=collected['artifact_sha256'];verify_artifacts(INPUT,ids);launch=read_json(INPUT,'launch.json');verify(launch)
    reports=[read_json(INPUT,t+'_audit.json') for t in TRIALS];accounting=summarize(reports)
    if any(collected[k]!=v for k,v in accounting.items()):raise ValueError('complete raw role accounting changed')
    sources=discover_sources((PROTOCOL,'scripts/read_go2_geometry_progress_family_causal_v1.py',
        'lewm/tests/test_geometry_progress_family_causal_windows_development.py'),launch['source_sha256'])
    definition=launch|dict(source_sha256=sources);verify(definition)
    if shutil.disk_usage(BASE.parent).free<40*1024**3+256*1024**2:raise ValueError('derivation budget plus40GiB reserve required')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',dict(input_sha256=ids,source_sha256=sources,
        protocol=PROTOCOL,scope='separate causal suffix label derivation and tensor readout; no optimizer or policy execution'))
    try:
        windows=[];index=[]
        for report in reports:
            trial=report['trial'];directory=INPUT/trial
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            rows=derive(raw,read_json(directory,'camera_audit.json'),report);reader=IntentReturnRGBDReplay(directory)
            for row in rows:
                windows.append(row)
                if not row['available']:
                    index.append(dict(window_id=row['window_id'],materialized=False,reason=row['reason']));continue
                sample=materialize(reader,row)
                if row['offset_ticks']==0:compare_initial(sample,initial_materialize(reader,report))
                history=sample['inputs']['observation_history']
                index.append(dict(window_id=row['window_id'],materialized=True,
                    history_shapes={k:list(v.shape) for k,v in history.items()},
                    history_sha256={k:fingerprint(v.numpy()) for k,v in history.items()},
                    known_action_sha256=fingerprint(sample['inputs']['known_action_blocks'].numpy()),
                    known_action_valid_sha256=fingerprint(sample['inputs']['known_action_valid'].numpy())))
            print('FAMILY_CAUSAL',trial,sum(r['available'] for r in rows),flush=True)
        write_json(OUTPUT/'windows.json',windows);write_json(OUTPUT/'tensor_index.json',index)
        plot_paths(reports,input_root=INPUT,output=OUTPUT)
        timing=np.asarray([v for r in reports for v in r['observation_and_control_wall_ms']])
        footprints=[f for r in reports for f in r['footprint_checks']]
        initial=[idx for row,idx in zip(windows,index,strict=True) if row['offset_ticks']==0 and idx['materialized']]
        result=dict(status='GEOMETRY_PROGRESS_FAMILY_CAUSAL_READOUT_COMPLETE',collection_sha256=args.collection_result_sha256,
            collection_root=str(INPUT),cohort=accounting,window_counts=counts(windows),
            total_planned_windows=len(windows),total_materialized_windows=sum(i['materialized'] for i in index),
            initial_tensor_equivalence_checked=len(initial),
            initial_context_counts={k:len({i['history_sha256'][k] for i in initial}) for k in ('rgb','body','control')},
            raw_accounting=dict(physics_samples=sum(r['physics_samples'] for r in reports),frames=sum(r['frames'] for r in reports),
                decisions=sum(r['decisions'] for r in reports),footprint_frames=len(footprints),
                stable_interior_failed_frames=sum(not f['stable_interior_metric_pass'] for f in footprints),
                near_occlusion_failed_frames=sum(f['near_occlusion_failure'] for f in footprints)),
            observation_and_control_wall_ms=dict(median=float(np.median(timing)),maximum=float(timing.max()),
                above100ms=int((timing>100).sum()),count=len(timing),includes_physics_tick=False,real_time_qualified=False),
            ready_for_separately_frozen_learning=bool(accounting['all_measurement_gates_pass'] and accounting['training_design_informative']),
            moving_contexts_do_not_prove_replanning=True,model_trained=False,navigation_qualified=False,goal_achieved=False,
            source_sha256=sources,artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','windows.json','tensor_index.json','native_paths.png','native_paths.svg')})
        verify(definition);verify_artifacts(INPUT,ids);verify_artifacts(OUTPUT,result['artifact_sha256'])
        write_json(OUTPUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','cohort')},indent=2),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAMILY_CAUSAL_DERIVATION_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

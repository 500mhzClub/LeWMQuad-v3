"""Full recorded controller equivalence and paired uninstrumented CPU timing."""
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from lewm.exact_mission_target_goal_probe_development import ExactMissionTargetGoalProbe
from lewm.batched_exact_mission_target_development import BatchedExactMissionTargetGoalProbe
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.auxiliary_downward45_packet_replay_development import packet,public_acquisition
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_exact_mission_target_goal_probe_v1 import OUTPUT as INPUT,CASES,CORRECTION,FITS
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing

OUTPUT=BASE/'go2_batched_exact_target_controller_v1_attempt_001'
PROTOCOL='docs/go2_batched_exact_target_controller_v1_2026-09-08.md'


def replay(admission,case,implementation,reader,rows,acquisitions):
    name,trial,variant,condition,model_name=case
    model,c,v=load_assigned(admission,model_name);assert (c,v)==(condition,variant)
    cls={'original':ExactMissionTargetGoalProbe,'batched':BatchedExactMissionTargetGoalProbe}[implementation]
    controller=cls(model,ArticulatedCollisionGeometry(URDF),condition=c,variant=v,persistent=True)
    before=state_digest(model.state_dict());times=[];signature=hashlib.sha256()
    for i,row in enumerate(rows):
        assert row['tick']==row['observation_index']==i
        p,d,f,now=reader.packet(i)
        aux=packet(INPUT/name,i,p,public_acquisition(acquisitions[i]),now_ns=now)
        start=time.perf_counter_ns()
        result=controller.observe(p,d,f,now_ns=now,auxiliary_depth=aux)
        elapsed=(time.perf_counter_ns()-start)/1e6
        encoded=json.dumps(result,sort_keys=True,separators=(',',':'))
        assert json.loads(encoded)==row['decision'],('full recorded decision differs',implementation,name,i)
        signature.update(encoded.encode());signature.update(b'\n')
        times.append(dict(frame=i,controller_wall_ms=elapsed,
            new_selection=result['new_selection'] is not None,terminal=result['terminal']))
        if i%50==0:print('EXACT_CONTROLLER_REPLAY',name,implementation,i,flush=True)
    assert state_digest(model.state_dict())==before and all(p.grad is None for p in model.parameters())
    return dict(implementation=implementation,frames=len(rows),all_decisions_exact=True,
        decision_stream_sha256=signature.hexdigest(),model_state_sha256=before,model_state_unchanged=True,
        times=times,all_frame_timing=timing([r['controller_wall_ms'] for r in times]),
        selection_frame_timing=timing([r['controller_wall_ms'] for r in times if r['new_selection']]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive full-controller benchmark required')
    verify_artifacts(INPUT,{'result.json':args.probe_result_sha256});result=read_json(INPUT,'result.json')
    assert result['status']=='EXACT_MISSION_TARGET_GOAL_PROBE_COMPLETE' and result['cases']==[list(c) for c in CASES]
    ids={'result.json':args.probe_result_sha256,**result['artifact_sha256']};verify_artifacts(INPUT,ids)
    old=read_json(INPUT,'launch.json');verify(old);admission=old['correction_admission']
    sources=discover_sources((PROTOCOL,'scripts/benchmark_go2_batched_exact_target_controller_v1.py',
        'lewm/tests/test_batched_exact_mission_target_development.py',
        'lewm/tests/test_batched_sample_bounds_development.py',
        'docs/go2_batched_sample_bounds_source_candidate_2026-09-08.md'),old['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('bounded benchmark resources unavailable')
    orders=[['original','batched'],['batched','original']]
    launch=old|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        benchmark_input_artifact_sha256=ids,implementation_order_per_case=orders,
        native_execution=False,model_training=False,controller_semantics_changed=False,
        controller_observe_only_timing=True,acquisition_and_serialization_timed=False,
        workers=1,numerical_threads=1,native_adoption=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch);start=time.perf_counter()
    print('BATCHED_EXACT_TARGET_CONTROLLER_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        reports=[]
        for case,order in zip(CASES,orders,strict=True):
            reader=IntentReturnRGBDReplay(INPUT/case[0]);rows=read_json(INPUT/case[0],'context_decisions.json')
            acquisitions=read_json(INPUT/case[0],'auxiliary_camera_audit.json')
            assert 0<len(rows)==len(reader.frames)==len(acquisitions)<=254
            runs={k:replay(admission,case,k,reader,rows,acquisitions) for k in order}
            a,b=runs['original'],runs['batched']
            assert a['decision_stream_sha256']==b['decision_stream_sha256'] and a['model_state_sha256']==b['model_state_sha256']
            assert all(x['frame']==y['frame'] and x['new_selection']==y['new_selection'] for x,y in zip(a['times'],b['times'],strict=True))
            ratios=[x['controller_wall_ms']/y['controller_wall_ms'] for x,y in zip(a['times'],b['times'],strict=True) if x['new_selection']]
            reports.append(dict(case=case[0],model=case[4],implementation_order=order,runs=runs,
                median_paired_selection_frame_speed_ratio=float(np.median(ratios)),
                full_recorded_decisions_exact=True,whole_loop_speedup_established=False))
        verify(launch);verify_artifacts(INPUT,ids)
        verify_artifacts(CORRECTION,admission['correction_artifact_sha256'])
        verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
        write_json(OUTPUT/'result.json',dict(status='BATCHED_EXACT_TARGET_CONTROLLER_COMPLETE',
            source_sha256=sources,probe_result_sha256=args.probe_result_sha256,
            artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')},conditions=reports,
            wall_s=time.perf_counter()-start,hardware_after=hardware(),
            native_execution=False,model_training=False,native_adoption=False,
            whole_loop_speedup_established=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
        print('BATCHED_EXACT_TARGET_CONTROLLER_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_BATCHED_CONTROLLER_BENCHMARK_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

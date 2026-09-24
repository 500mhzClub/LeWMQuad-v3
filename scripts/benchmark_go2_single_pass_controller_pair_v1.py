"""Paired complete controller and receipt costs on a declared early prefix."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from itertools import islice
import json
import time
import cv2
import numpy as np
import torch
from lewm.later_floor_resolution_controller_development import LaterFloorResolutionRoundTripController
from lewm.single_pass_later_floor_controller_development import SinglePassLaterFloorController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows, writer
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.replay_go2_single_pass_maze_controller_v1 import (
    OUTPUT as REPLAY, INPUT, CASE, verify_all as verify_replay)

OUTPUT=BASE/'go2_single_pass_controller_pair_v1_attempt_001'
PROTOCOL='docs/go2_single_pass_controller_pair_v1_2026-09-09.md'
FRAMES=256
IMPLEMENTATIONS=('original','single_pass')
ALLOWANCE=1024**3


def order(frame):
    return IMPLEMENTATIONS if frame%2==0 else IMPLEMENTATIONS[::-1]


def admit(result):
    if (result['status']!='SINGLE_PASS_MAZE_CONTROLLER_REPLAY_COMPLETE'
            or result['frames']!=1881 or result['first_terminal_frame']!=1870):
        raise ValueError('completed full original-population equivalence required')
    for key in ('all_complete_decisions_exact','model_state_unchanged','original_failure_preserved'):
        if result[key] is not True:raise ValueError('full replay invariant required: '+key)
    if result['native_adoption'] is not False or result['native_execution'] is not False:
        raise ValueError('saved equivalence evidence only; no native adoption')


def measure(controller, inputs, frame, append, *, clock=time.perf_counter_ns, cpu_clock=time.process_time_ns):
    p,d,f,auxiliary,now=inputs
    cpu_start=cpu_clock();start=clock()
    decision=controller.observe(p,d,f,now_ns=now,auxiliary_depth=auxiliary)
    controlled=clock()
    # The same production gzip writer serializes, compresses and flushes both
    # arms. This small benchmark row contains no copied historical wall times.
    append(dict(tick=frame,decision=decision))
    finished=clock();cpu_end=cpu_clock()
    return decision,dict(controller_wall_ms=(controlled-start)/1e6,
        receipt_wall_ms=(finished-controlled)/1e6,
        controller_and_receipt_wall_ms=(finished-start)/1e6,
        controller_and_receipt_process_cpu_ms=(cpu_end-cpu_start)/1e6)


def verify_inputs(launch):
    verify(launch);verify_artifacts(REPLAY,launch['replay_artifact_sha256'])
    verify_replay(read_json(REPLAY,'launch.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay-result-sha256',required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fresh paired benchmark required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    verify_artifacts(REPLAY,{'result.json':args.replay_result_sha256})
    result=read_json(REPLAY,'result.json');admit(result)
    ids={'result.json':args.replay_result_sha256,**result['artifact_sha256']}
    old=read_json(REPLAY,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/benchmark_go2_single_pass_controller_pair_v1.py',
        'lewm/tests/test_single_pass_controller_pair_development.py'),result['source_sha256'])
    resources=hardware()
    launch=old|dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,
        replay_artifact_sha256=ids,hardware=resources,frames=FRAMES,
        implementation_classes=dict(original='LaterFloorResolutionRoundTripController',
            single_pass='SinglePassLaterFloorController'),
        independent_controller_and_model_states=True,alternating_first_implementation_each_frame=True,
        shared_decoded_inputs_copied_per_arm_outside_timing=True,input_mutation_checked=True,
        controller_observe_and_production_receipt_writer_timed=True,
        decoding_acquisition_physics_and_comparison_timed=False,
        benchmark_receipt_contains_tick_and_complete_decision=True,
        cpu_processes=1,numerical_threads=1,minimum_available_ram_bytes=16*1024**3,
        output_allowance_bytes=ALLOWANCE,os_resource_limits_enforced=False,
        controlled_speed_comparison=True,native_execution=False,model_training=False,native_adoption=False,
        complete_trajectory_timing=False,whole_loop_speedup_established=False,
        concurrency_reason='one paired CPU benchmark beside the existing native settling scene; shared-machine timing')
    verify_inputs(launch)
    memory_ok=resources['memory_available_bytes']>=16*1024**3
    storage_ok=resources['artifact_free_bytes']>=RESERVE_BYTES+ALLOWANCE
    if args.preflight_only:
        print('SINGLE_PASS_PAIR_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('paired controller benchmark resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('SINGLE_PASS_PAIR_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        name,index,variant,condition,model_name=CASE
        controllers={};models={}
        for implementation,cls in zip(IMPLEMENTATIONS,
                (LaterFloorResolutionRoundTripController,SinglePassLaterFloorController),strict=True):
            model,c,v=load_assigned(old['correction_admission'],model_name)
            assert(c,v)==(condition,variant) and state_digest(model.state_dict())==result['model_state_sha256']
            models[implementation]=model
            controllers[implementation]=cls(model,ArticulatedCollisionGeometry(URDF),
                public_mission=public_mission(index),navigation_ticks=NAVIGATION_TICKS,
                persistent=True,condition=c,variant=v)
        reader=IntentReturnRGBDReplay(INPUT/name)
        acquisitions=read_json(INPUT/name,'auxiliary_camera_audit.json');measurements=[]
        with ExitStack() as stack:
            writers={}
            for implementation in IMPLEMENTATIONS:
                directory=OUTPUT/implementation;directory.mkdir()
                writers[implementation]=stack.enter_context(writer(directory))
            for i,saved in enumerate(islice(read_rows(INPUT/name),FRAMES)):
                if saved['tick']!=i:raise ValueError('ordered complete prefix required')
                p,d,f,now=reader.packet(i)
                auxiliary=packet(INPUT/name,i,p,public_acquisition(acquisitions[i]),now_ns=now)
                inputs=(p,d,f,auxiliary,now);input_sha=fingerprint(inputs);row={}
                for implementation in order(i):
                    private=deepcopy(inputs)
                    if fingerprint(private)!=input_sha:raise ValueError('copied input changed')
                    decision,elapsed=measure(controllers[implementation],private,i,writers[implementation])
                    if fingerprint(private)!=input_sha:raise ValueError('controller mutated its public inputs')
                    if json.loads(json.dumps(decision,allow_nan=False))!=saved['decision']:
                        raise ValueError('complete decision mismatch at '+str(i)+' in '+implementation)
                    row[implementation]=elapsed
                measurements.append(dict(frame=i,order=order(i),new_selection=decision['new_selection'] is not None,
                    terminal=decision['terminal'],runs=row))
                if sum((OUTPUT/k/'context_decisions.jsonl.gz').stat().st_size for k in IMPLEMENTATIONS)>ALLOWANCE:
                    raise ValueError('bounded paired receipt allowance exceeded')
                if i%32==0:print('SINGLE_PASS_PAIR_FRAME',i,flush=True)
        assert len(measurements)==FRAMES
        for model in models.values():
            assert state_digest(model.state_dict())==result['model_state_sha256']
            assert all(p.grad is None for p in model.parameters())
        artifacts={'launch.json':digest(OUTPUT/'launch.json')}
        for implementation in IMPLEMENTATIONS:
            filename=implementation+'/context_decisions.jsonl.gz';artifacts[filename]=digest(OUTPUT/filename)
        assert artifacts['original/context_decisions.jsonl.gz']==artifacts['single_pass/context_decisions.jsonl.gz']
        metrics={k:{metric:timing([r['runs'][k][metric] for r in measurements])
            for metric in measurements[0]['runs'][k]} for k in IMPLEMENTATIONS}
        ratios={metric:float(np.median([r['runs']['original'][metric]/r['runs']['single_pass'][metric]
            for r in measurements])) for metric in ('controller_wall_ms','controller_and_receipt_wall_ms')}
        deadlines={k:{metric:sum(r['runs'][k][metric]>100. for r in measurements)
            for metric in ('controller_wall_ms','controller_and_receipt_wall_ms')} for k in IMPLEMENTATIONS}
        verify_inputs(launch);verify_artifacts(OUTPUT,artifacts)
        write_json(OUTPUT/'result.json',dict(status='SINGLE_PASS_CONTROLLER_PAIR_COMPLETE',
            source_sha256=sources,artifact_sha256=artifacts,frames=FRAMES,measurements=measurements,
            metrics=metrics,median_paired_speed_ratios=ratios,deadline_misses_100ms=deadlines,
            complete_original_decisions_exact=True,compressed_decision_streams_identical=True,
            model_state_unchanged=True,model_state_sha256=result['model_state_sha256'],
            input_mutation_checked=True,hardware_after=hardware(),wall_s=time.perf_counter()-started,
            native_execution=False,native_adoption=False,model_training=False,
            complete_trajectory_timing=False,whole_loop_speedup_established=False,
            real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
        print('SINGLE_PASS_PAIR_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()

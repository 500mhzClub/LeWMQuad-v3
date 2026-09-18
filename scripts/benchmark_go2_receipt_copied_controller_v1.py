"""Paired complete-decision replay of original and receipt-copied controllers."""
from itertools import islice
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from lewm.receipt_copied_selector_development import ReceiptCopiedMeasuredFloorController
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.replay_go2_residual_first_interval_prefix_v1 import INPUT, INPUT_SHA, CASE, admit, verify_inputs
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_receipt_copied_controller_benchmark_v1_attempt_001'
PROTOCOL='docs/go2_receipt_copied_controller_benchmark_v1_2026-09-09.md'
FRAMES=514
LABELS=('original','receipt_copied')


def paired_step(controllers,inputs,original,*,frame,clock=time.perf_counter_ns):
    policy,depth,fast,auxiliary,image,now=inputs
    if (original['tick']!=frame or original['observation_index']!=frame
            or original['pre_sample_index']!=749+50*frame or now!=1_500_000_000+frame*100_000_000):
        raise ValueError('exact ordered original observation required')
    before=fingerprint(inputs); elapsed={}; order=(0,1) if frame%2==0 else (1,0)
    for index in order:
        started=clock()
        result=controllers[index].observe(policy,depth,fast,now_ns=now,auxiliary_depth=auxiliary,auxiliary_rgb=image)
        duration=clock()-started
        if duration<0: raise ValueError('monotone controller timer required')
        elapsed[LABELS[index]]=duration/1e6
        if fingerprint(inputs)!=before: raise ValueError('controller mutated shared public inputs')
        normalized=json.loads(json.dumps(result,allow_nan=False))
        if normalized!=original['decision']:
            raise ValueError('complete '+LABELS[index]+' decision differs at frame '+str(frame))
    payload=json.dumps(original['decision'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(frame=frame,order=[LABELS[i] for i in order],controller_wall_ms=elapsed,
        complete_original_and_candidate_decisions_exact=True,public_inputs_unchanged=True,
        decision_sha256=hashlib.sha256(payload).hexdigest(),terminal=original['decision']['terminal'],
        warmup=frame<3)


def aggregate(rows):
    selected=[r for r in rows if not r['warmup'] and r['terminal'] is None]
    if not selected: raise ValueError('non-warmup active observations required for timing comparison')
    def values(group):
        if not group: return None
        old=np.asarray([r['controller_wall_ms']['original'] for r in group])
        new=np.asarray([r['controller_wall_ms']['receipt_copied'] for r in group])
        return dict(observations=len(group),original_median_ms=float(np.median(old)),
            receipt_copied_median_ms=float(np.median(new)),paired_median_reduction_ms=float(np.median(old-new)),
            original_mean_ms=float(old.mean()),receipt_copied_mean_ms=float(new.mean()),
            original_over_100ms=int((old>100).sum()),receipt_copied_over_100ms=int((new>100).sum()))
    return dict(active=values(selected),by_first_controller={label:values([r for r in selected if r['order'][0]==label])
        for label in LABELS},warmup_observations=sum(r['warmup'] for r in rows),
        terminal_observations=sum(r['terminal'] is not None for r in rows),
        acquisition_and_receipt_io_timed=False,profiling_instrumentation_used=False,
        alternating_order_per_observation=True,concurrent_native_job_possible=True,
        end_to_end_speedup_established=False,real_time_qualified=False)


def replay(launch):
    models=[]; controllers=[]
    for cls in (MeasuredFloorTransportController,ReceiptCopiedMeasuredFloorController):
        model,c,v=load_assigned(launch['correction_admission'],CASE[4])
        if (c,v)!=(CASE[3],CASE[2]) or state_digest(model.state_dict())!=MODEL_STATE:
            raise ValueError('same original assigned corrected model required')
        if any(model is prior for prior in models): raise ValueError('separate model instances required')
        models.append(model)
        controllers.append(cls(model,ArticulatedCollisionGeometry(URDF),public_mission=public_mission(2),
            navigation_ticks=NAVIGATION_TICKS,persistent=True,condition=c,variant=v))
    directory=INPUT/CASE[0]; reader=IntentReturnRGBDReplay(directory)
    acquisitions=read_json(directory,'auxiliary_camera_audit.json'); tape=read_json(directory,'command_tape.json')
    if len(reader.frames)!=FRAMES or len(acquisitions)!=FRAMES or len(tape)!=FRAMES-1:
        raise ValueError('complete fixed original episode population required')
    timings=[]
    with (OUTPUT/'paired_timings.jsonl').open('x') as stream,(OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
        for i,original in enumerate(islice(read_rows(directory),FRAMES)):
            if i<len(tape):
                if tape[i]['requested_command']!=original['decision']['requested_command'] or not tape[i]['completed']:
                    raise ValueError('original completed dispatched command required')
            elif original['decision']['terminal'] is None:
                raise ValueError('final observation without a command must already be terminal')
            p,d,f,now=reader.packet(i)
            image,auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
            row=paired_step(controllers,(p,d,f,auxiliary,image,now),original,frame=i)
            timings.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            if i%64==0:
                resources=hardware(); monitor.write(json.dumps(dict(frame=i,**resources))+'\n'); monitor.flush()
                if resources['artifact_free_bytes']<RESERVE_BYTES+64*1024**2:
                    raise ValueError('benchmark storage reserve unavailable')
                print('RECEIPT_COPIED_BENCHMARK_FRAME',i,flush=True)
    if len(timings)!=FRAMES: raise ValueError('all fixed original observations required')
    for model in models:
        if state_digest(model.state_dict())!=MODEL_STATE or any(p.grad is not None for p in model.parameters()):
            raise ValueError('both model states must remain unchanged without gradients')
    return dict(frames=len(timings),complete_original_and_candidate_decisions_exact=True,
        public_inputs_unchanged=True,both_model_states_unchanged=True,model_state_sha256=MODEL_STATE,
        paired_timing=aggregate(timings),native_execution=False,model_training=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive paired receipt-copy benchmark required')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA}); result=read_json(INPUT,'result.json')
    ids={'result.json':INPUT_SHA,**result['artifact_sha256']}; verify_artifacts(INPUT,ids)
    old=read_json(INPUT,'launch.json'); admit(result,read_json(INPUT,CASE[0]+'_audit.json'),old)
    sources=discover_sources((PROTOCOL,'scripts/benchmark_go2_receipt_copied_controller_v1.py',
        'lewm/tests/test_receipt_copied_selector_development.py','lewm/tests/test_receipt_copy_development.py',
        'lewm/tests/test_receipt_copied_controller_benchmark_development.py',
        'docs/go2_controller_phase_timing_result_2026-09-09.md',
        'docs/go2_frame_cache_and_copy_progress_2026-09-09.md'),old['source_sha256'])
    launch=old|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),replay_input_bindings=ids,
        planned_case=list(CASE),maximum_frames=FRAMES,controller_variants=list(LABELS),
        native_execution=False,native_scene_workers=0,model_loaded=True,model_training=False,
        numerical_threads=1,replay_workers=1,separate_model_and_controller_per_variant=True,
        order='original first on even observations, receipt-copied first on odd observations',
        minimum_available_ram_bytes=16*1024**3,output_allowance_bytes=64*1024**2,
        os_resource_limits_enforced=False,controller_or_model_selection_performed=False,
        real_time_qualified=False,goal_achieved=False)
    verify_inputs(launch); resources=hardware(); launch['hardware']=resources
    if resources['memory_available_bytes']<16*1024**3 or resources['artifact_free_bytes']<RESERVE_BYTES+64*1024**2:
        raise ValueError('paired controller replay resource allowance unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch); started=time.perf_counter()
    print('RECEIPT_COPIED_BENCHMARK_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        report=replay(launch); verify_inputs(launch)
        bindings={n:digest(OUTPUT/n) for n in ('launch.json','paired_timings.jsonl','resource_monitor.jsonl')}
        verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='RECEIPT_COPIED_CONTROLLER_BENCHMARK_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=bindings,report=report,input_result_sha256=INPUT_SHA,
            wall_s=time.perf_counter()-started,hardware_after=hardware(),native_execution=False,
            model_training=False,new_navigation_outcomes=False,real_time_qualified=False,goal_achieved=False))
        print('RECEIPT_COPIED_BENCHMARK_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RECEIPT_COPIED_BENCHMARK_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()

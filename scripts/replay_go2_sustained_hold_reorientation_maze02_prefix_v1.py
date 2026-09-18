"""Raw paired replay to the first changed request; no successor outcome consumed."""
import argparse
from itertools import islice
import json
import os
from pathlib import Path
import time
import cv2
import torch
from lewm.hold_reorientation_controller_development import HoldReorientationController
from lewm.sustained_hold_reorientation_controller_development import SustainedHoldReorientationController
from lewm.sustained_hold_reorientation_prefix_development import compare_step
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import check_go2_sustained_hold_reorientation_saved_prefix_v1 as saved
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json,verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows,writer,NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet,public_acquisition
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live

SOURCE='scripts/replay_go2_sustained_hold_reorientation_maze02_prefix_v1.py'
PROTOCOL='docs/go2_sustained_hold_reorientation_raw_prefix_v1_2026-09-11.md'
TEST='lewm/tests/test_sustained_hold_reorientation_raw_prefix_development.py'
OUTPUT=BASE/'go2_sustained_hold_reorientation_maze02_prefix_v1_attempt_001'
SAVED_SHA='a58a89afd262ffaef63c5adac6a7ae6f8259f5b93a21f19a5dc4da4e4e9f6177'
MODEL_SHA='35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a'
FRAMES=407
original=saved.prior.wait.native
PREVIOUS_CPU_OWNER=dict(pid=2796867,created=1789098212.37,command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python','-B',
    'scripts/replay_go2_receipt_copied_footprint_late_history_v1.py'])


def saved_inputs():
    if digest(saved.OUTPUT)!=SAVED_SHA:raise ValueError('exact saved prefix required')
    result=json.loads(saved.OUTPUT.read_text());verify(result['source_sha256'])
    if (result['status']!='SUSTAINED_HOLD_REORIENTATION_SAVED_PREFIX_COMPLETE'
            or result['frames']!=FRAMES or result['first_changed_requested_command_frame']!=FRAMES-1
            or result['native_result_sha256']!=saved.prior.NATIVE_SHA
            or result['candidate_post_intervention_observations_consumed'] is not False):
        raise ValueError('original fixed prospective saved prefix required')
    comparisons=result['comparisons']
    if len(comparisons)!=FRAMES or any(r['frame']!=i or r['changed'] is not (i==FRAMES-1) for i,r in enumerate(comparisons)):
        raise ValueError('complete fixed saved comparison population required')
    return result


def resources_for(resources):
    if resources['memory_available_bytes']<32*1024**3 or resources['artifact_free_bytes']<41*1024**3:
        raise ValueError('32GiB RAM and 40+1GiB artifact envelope required')


def require_previous_owners_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=saved.prior.wait.BOOT:
        raise ValueError('original boot required')
    if owner_live(PREVIOUS_CPU_OWNER) or owner_live(saved.prior.OWNER):
        raise ValueError('original CPU replay and source native waiter must have ended')


def admit(sources):
    prior=saved.prior;wait=prior.wait
    require_previous_owners_ended()
    verify_artifacts(wait.OUTPUT,{'result.json':prior.WAIT_SHA,'launch.json':prior.WAIT_LAUNCH})
    result=read_json(wait.OUTPUT,'result.json')
    if (wait.OUTPUT/'failure.json').exists() or (original.OUTPUT/'failure.json').exists():
        raise ValueError('original failure must be preserved')
    verify_artifacts(wait.OUTPUT,result['artifact_sha256'])
    completion=wait.authenticate_completed(sources,read_json(wait.OUTPUT,'input_completion.json'))
    if completion!=result['report'] or completion!=read_json(wait.OUTPUT,'native_completion.json'):
        raise ValueError('complete original native admission must reconstruct')
    verify_artifacts(original.OUTPUT,{'result.json':prior.NATIVE_SHA,'launch.json':prior.NATIVE_LAUNCH})
    launch=read_json(original.OUTPUT,'launch.json')
    if launch['model_state_sha256']!=MODEL_SHA:raise ValueError('exact original assigned model required')
    return dict(native_result_sha256=prior.NATIVE_SHA,native_launch_sha256=prior.NATIVE_LAUNCH,
        waiter_result_sha256=prior.WAIT_SHA,original_completion=completion,
        original_completion_verifier_reexecuted=True,full_training_ancestry_reexecuted=False,
        actual_assigned_model_loader_required=True)


def replay(expected):
    launch=read_json(original.OUTPUT,'launch.json')
    models=[original.assigned_model(launch) for _ in range(2)]
    if any(state_digest(m.state_dict())!=MODEL_SHA for m in models):raise ValueError('two exact fresh original model copies required')
    options=dict(public_mission=public_mission(2),navigation_ticks=NAVIGATION_TICKS,
        condition='jepa',variant='full',persistent=True)
    geometry=ArticulatedCollisionGeometry(URDF)
    controllers=(HoldReorientationController(models[0],geometry,**options),
        SustainedHoldReorientationController(models[1],geometry,**options))
    directory=original.OUTPUT/original.CASE[0];reader=IntentReturnRGBDReplay(directory)
    acquisitions=read_json(directory,'auxiliary_camera_audit.json');tape=read_json(directory,'command_tape.json')
    count=forecasts=0;last=None
    with writer(OUTPUT) as append:
        for row in islice(read_rows(directory),FRAMES):
            frame=row['tick'];comparison=expected['comparisons'][frame]
            if (frame!=count or saved.identity(row)!=comparison['original_row_sha256']
                    or not tape[frame]['completed'] or tape[frame]['tick']!=frame
                    or tape[frame]['pre_sample_index']!=749+50*frame or tape[frame]['post_sample_index']!=799+50*frame):
                raise ValueError('exact completed original public and physical prefix required')
            if controllers[0].residual.pending!=controllers[1].residual.pending:
                raise ValueError('identical preceding pending model forecasts required')
            p,d,fast,now=reader.packet(frame)
            image,auxiliary=packet(directory,frame,p,public_acquisition(acquisitions[frame]),now_ns=now)
            public_hash=fingerprint((p,d,fast,image,auxiliary,now))
            decisions=[c.observe(p,d,fast,now_ns=now,auxiliary_depth=auxiliary,auxiliary_rgb=image) for c in controllers]
            old,new=[json.loads(json.dumps(x)) for x in decisions]
            if old!=row['decision']:raise ValueError('complete raw original decision does not reproduce: '+str(frame))
            check=compare_step(old,new,tape[frame]['requested_command'],frame=frame,
                expected_selection_sha256=comparison['candidate_selection_sha256'])
            if check['requested_command_changed'] is not (frame==FRAMES-1):
                raise ValueError('only fixed first continuation boundary may change')
            if public_hash!=fingerprint((p,d,fast,image,auxiliary,now)):raise ValueError('public input arrays changed')
            contacts=[fingerprint(state_tree(c.memory)) for c in controllers]
            if (contacts[0]!=contacts[1] or controllers[0].mapper.floor!=controllers[1].mapper.floor
                    or controllers[0].mapper.occupied!=controllers[1].mapper.occupied):
                raise ValueError('complete retained contact and observed planning map must match')
            append(dict(tick=frame,decision=new,original_requested_command=tape[frame]['requested_command'],
                comparison=check,public_input_sha256=public_hash,public_input_arrays_unchanged=True,
                complete_retained_contact_state_sha256=contacts[0],complete_retained_contact_state_equal=True,
                original_complete_decision_reconstructed=True))
            count+=1;forecasts+=int(check['raw_model_forecasts_compared']);last=new
            if frame%50==0 or frame==FRAMES-1:print('SUSTAINED_RAW_FRAME',frame,flush=True)
    if count!=FRAMES or last['requested_command']!=expected['first_boundary']['candidate_requested_command']:
        raise ValueError('complete fixed raw prefix required')
    if any(state_digest(m.state_dict())!=MODEL_SHA or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('original model state or gradients changed')
    return dict(frames=count,raw_model_forecast_comparisons=forecasts,first_changed_command_frame=FRAMES-1,
        original_requested_command=expected['first_boundary']['original_requested_command'],
        candidate_requested_command=last['requested_command'],original_complete_decisions_reconstructed=True,
        candidate_matches_every_saved_selection=True,observed_map_and_contact_state_exact=True,
        prior_pending_model_forecasts_exact=True,model_state_sha256=MODEL_SHA,model_state_unchanged=True,
        no_observation_after_changed_request_consumed=True,changed_command_executed=False,
        native_execution=False,unexecuted_outcomes_inferred=False,navigation_verified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-preflight-only',action='store_true');args=parser.parse_args()
    env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',OPENCV_OPENCL_RUNTIME='disabled')
    if any(os.environ.get(k)!=v for k,v in env.items()) or cv2.ocl.useOpenCL():raise ValueError('fixed CPU environment required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive raw prefix; no retry or resume')
    expected=saved_inputs()
    sources=discover_sources((SOURCE,PROTOCOL,TEST),expected['source_sha256']|{str(saved.OUTPUT.relative_to(ROOT)):SAVED_SHA})
    verify(sources);require_previous_owners_ended();resources=hardware();resources_for(resources)
    if args.source_preflight_only:
        print('SUSTAINED_RAW_PREFLIGHT_PASS',len(sources),json.dumps(resources),flush=True);return
    admission=admit(sources);resources_for(hardware());verify(sources);require_previous_owners_ended();create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_admission=admission,
        saved_prefix_sha256=SAVED_SHA,model_state_sha256=MODEL_SHA,frames=FRAMES,hardware=resources,
        previous_cpu_owner=PREVIOUS_CPU_OWNER,boot_id=saved.prior.wait.BOOT,owner_pid=os.getpid(),
        protocol=PROTOCOL,environment=env,native_execution=False,model_training=False,automatic_retry=False))
    print('SUSTAINED_RAW_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter()
    try:
        report=replay(expected)
        if admit(sources)!=admission or saved_inputs()!=expected:raise ValueError('bound inputs changed')
        verify(sources);ids={n:digest(OUTPUT/n) for n in ('launch.json',NAME)};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='SUSTAINED_HOLD_REORIENTATION_RAW_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-start,
            native_execution=False,goal_achieved=False))
        print('SUSTAINED_RAW_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SUSTAINED_RAW_PREFIX_FAILURE',
            reason=repr(error),automatic_retry=False,evidence_preserved=True));raise


if __name__=='__main__':main()

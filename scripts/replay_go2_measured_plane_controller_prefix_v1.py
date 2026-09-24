"""Full learned-controller comparison, stopping at the first changed command."""
import argparse
from contextlib import closing
from itertools import islice
import json
import time
import torch
import psutil

from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import verify_go2_measured_plane_observer_history_v1 as completed
from scripts import admit_go2_completed_extended_budget_worker_v1 as worker
from scripts.measured_plane_controller_prefix_comparison_development import compare
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import URDF

run, native = completed.run, worker.native
SOURCE = 'scripts/replay_go2_measured_plane_controller_prefix_v1.py'
TEST = 'lewm/tests/test_measured_plane_controller_prefix_runner_development.py'
COMPARISON_TEST = 'lewm/tests/test_measured_plane_controller_prefix_comparison_development.py'
PROTOCOL = 'docs/go2_measured_plane_controller_prefix_v1_2026-09-11.md'
OUTPUT = run.BASE/'go2_measured_plane_controller_prefix_v1_attempt_001'
WAIT_ROOT = run.BASE/'go2_measured_plane_observer_completion_wait_v1_attempt_001'
WAIT_LAUNCH_SHA = '188bbb617aad2f4cbd8c312b164d9e47f3ff5e7c5bdaf890f89777f62ce3c93c'
WORKER_ADMISSION_SHA = '2e748a012979225e1871d79a7240fe9bf1a14da54a2aeca3ea7d6db9b1fbeedf'
MODEL_SHA = '56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd'
MAX_OUTPUT_BYTES = 2*1024**3


def require_positive_observer(proof):
    report = proof['report']
    if (proof['status'] != 'MEASURED_PLANE_OBSERVER_HISTORY_COMPLETION_VERIFIED'
            or proof['original_launch_sha256'] != completed.LAUNCH_SHA
            or proof['original_owner_ended'] is not True
            or proof['complete_output_stream_checked'] is not True
            or proof['actual_consumed_raw_packets_reconstructed'] != run.FRAMES
            or proof['native_completion_admitted'] is not False
            or report['frames'] != run.FRAMES or report['planned_frames'] != run.FRAMES
            or report['complete_planned_history'] is not True
            or report['stop_reason'] != 'FIXED_HISTORY_END'
            or report['candidate_failure_preserved'] is not False
            or report['final_comparison']['candidate_visual_failure'] is not None
            or report['final_comparison']['candidate_floor_failure'] is not None
            or report['navigation_recovered'] is not False):
        raise ValueError('verified complete positive observer history required before controller replay')


def prepare(proof_sha, waiter_sha):
    run.verify({str(completed.OUTPUT.relative_to(run.ROOT)):proof_sha,
        str(worker.OUTPUT.relative_to(run.ROOT)):WORKER_ADMISSION_SHA})
    proof = json.loads(completed.OUTPUT.read_text())
    require_positive_observer(proof)
    run.verify_artifacts(WAIT_ROOT, {'launch.json':WAIT_LAUNCH_SHA,'result.json':waiter_sha})
    wait_launch = run.read_json(WAIT_ROOT,'launch.json'); waited = run.read_json(WAIT_ROOT,'result.json')
    if (run.owner_live(wait_launch['owner'])
            or waited['status'] != 'MEASURED_PLANE_OBSERVER_COMPLETION_WAIT_V1_COMPLETE'
            or waited['completion_sha256'] != proof_sha
            or waited['completion_receipt'] != str(completed.OUTPUT.relative_to(run.ROOT))
            or waited['original_result_sha256'] != proof['result_sha256']
            or waited['source_sha256'] != wait_launch['source_sha256']
            or waited['verification_executed_once'] is not True):
        raise ValueError('exact observer verification waiter must have completed and ended')
    run.verify_artifacts(WAIT_ROOT, waited['artifact_sha256'])
    observer_launch = run.read_json(run.OUTPUT,'launch.json')
    completed.original_owner_ended(observer_launch)
    native_proof = json.loads(worker.OUTPUT.read_text())
    if (native_proof['status'] != 'EXTENDED_BUDGET_COMPLETED_WORKER_ADMITTED'
            or native_proof['worker_terminal_sha256'] != worker.WORKER_SHA
            or native_proof['model_state_sha256'] != MODEL_SHA
            or native_proof['model_state_unchanged'] is not True
            or native_proof['original_worker_ended'] is not True
            or run.owner_live(native_proof['original_worker'])):
        raise ValueError('same ended raw native worker and assigned corrected model required')
    inherited = {}
    for bindings in (proof['source_sha256'],waited['source_sha256'],native_proof['source_sha256']):
        for name,sha in bindings.items():
            if name in inherited and inherited[name] != sha: raise ValueError('source ancestry conflict')
            inherited[name] = sha
    seeds = (SOURCE,TEST,COMPARISON_TEST,PROTOCOL,str(completed.OUTPUT.relative_to(run.ROOT)),
        str(worker.OUTPUT.relative_to(run.ROOT)))
    sources = discover_sources(seeds,inherited)
    verify_inputs(sources,proof,native_proof,proof_sha,waiter_sha)
    return sources,proof,native_proof


def verify_inputs(sources,proof,native_proof,proof_sha,waiter_sha):
    run.verify(sources)
    run.verify({str(completed.OUTPUT.relative_to(run.ROOT)):proof_sha,
        str(worker.OUTPUT.relative_to(run.ROOT)):WORKER_ADMISSION_SHA})
    run.verify_artifacts(run.OUTPUT,proof['artifact_sha256'] | {'result.json':proof['result_sha256']})
    run.verify_artifacts(native.OUTPUT,native_proof['artifact_sha256'])
    run.verify_artifacts(WAIT_ROOT,{'launch.json':WAIT_LAUNCH_SHA,'result.json':waiter_sha})
    launch = run.read_json(WAIT_ROOT,'launch.json')
    if run.owner_live(launch['owner']) or run.owner_live(native_proof['original_worker']):
        raise ValueError('original verification and native worker owners must remain ended')
    completed.original_owner_ended(run.read_json(run.OUTPUT,'launch.json'))


def resources():
    hw = run.hardware()
    if hw['memory_available_bytes'] < 64*1024**3 or hw['artifact_free_bytes'] < 41*1024**3+MAX_OUTPUT_BYTES:
        raise ValueError('64 GiB RAM and 43 GiB disk required for paired controllers and native reserve')
    return hw


def command_endpoint(row, tape, frame):
    if (row['tick'] != frame or row['observation_index'] != frame or row['pre_sample_index'] != 749+50*frame
            or tape['tick'] != frame or tape['completed'] is not True
            or tape['pre_sample_index'] != 749+50*frame or tape['post_sample_index'] != 799+50*frame
            or row['decision']['requested_command'] != tape['requested_command']):
        raise ValueError('actual original command and physical sample endpoints required')


def replay():
    launch = run.read_json(native.OUTPUT,'launch.json')
    models = [native.assigned_model(launch),native.assigned_model(launch)]
    if models[0] is models[1] or any(state_digest(m.state_dict()) != MODEL_SHA or m.training for m in models):
        raise ValueError('two fresh exact originally assigned evaluation models required')
    options = dict(public_mission=public_mission(2),navigation_ticks=4000,
        condition=native.CASE[3],variant=native.CASE[2],persistent=True)
    baseline = ResidualAnchoredContinuationController(models[0],ArticulatedCollisionGeometry(URDF),**options)
    candidate = MeasuredPlaneResidualController(models[1],ArticulatedCollisionGeometry(URDF),**options)
    directory = native.OUTPUT/native.CASE[0]
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory,'auxiliary_camera_audit.json')
    tape = run.read_json(directory,'command_tape.json')
    count = forecasts = 0
    stopped = False
    with run.pipeline.writer(OUTPUT) as append, closing(run.pipeline.read_rows(directory)) as originals, \
            closing(run.pipeline.read_rows(run.OUTPUT)) as observers:
        for recorded,observer in zip(islice(originals,run.FRAMES),observers,strict=True):
            frame = count
            command_endpoint(recorded,tape[frame],frame)
            if observer['tick'] != frame: raise ValueError('same complete observer frame required')
            p,d,f,now = reader.packet(frame)
            image,aux = run.pipeline.rgb_packet(directory,frame,p,
                run.public_acquisition(acquisitions[frame]),now_ns=now)
            public = p,d,f,image,aux
            before = run.fingerprint(public)
            if before != observer['comparison']['raw_packet_sha256']:
                raise ValueError('same verified public observation packet required')
            old = baseline.observe(p,d,f,auxiliary_rgb=image,auxiliary_depth=aux,now_ns=now)
            if run.fingerprint(public) != before: raise ValueError('original controller changed public inputs')
            live = candidate.observe(p,d,f,auxiliary_rgb=image,auxiliary_depth=aux,now_ns=now)
            old,live = json.loads(run.canonical(old)),json.loads(run.canonical(live))
            try:
                if run.fingerprint(public) != before: raise ValueError('candidate controller changed public inputs')
                check = compare(old,live,recorded['decision'],observer,frame=frame)
            except Exception as error:
                append(dict(tick=frame,original=old,decision=live,comparison_failure=repr(error)))
                raise
            append(dict(tick=frame,original=old,decision=live,comparison=check,
                public_packet_sha256=before,public_inputs_unchanged=True,
                original_requested_command=tape[frame]['requested_command']))
            count += 1; forecasts += int(check['original_forecast_compared'])
            if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError('bounded controller output allowance exceeded')
            if frame % 50 == 0 or check['stop']:
                print('MEASURED_PLANE_CONTROLLER_FRAME',frame,check['stop_reason'],flush=True)
            if check['stop']:
                stopped = True
                break
    if not count or not stopped: raise ValueError('explicit command or terminal comparison boundary required')
    if any(state_digest(m.state_dict()) != MODEL_SHA or any(p.grad is not None for p in m.parameters()) for m in models):
        raise ValueError('both assigned model states and absent gradients must remain unchanged')
    return dict(frames=count,raw_forecast_comparisons=forecasts,boundary_comparison=check,
        boundary_original=old,boundary_candidate=live,model_state_sha256=MODEL_SHA,
        two_fresh_models=True,model_states_unchanged=True,original_controller_fully_reproduced=True,
        candidate_observer_and_floor_fully_reproduced=True,source_trajectory_is_development_only=True,
        following_changed_command_outcome_consumed=False,changed_command_executed=False,
        native_parent_completion_admitted=False,full_training_ancestry_reexecuted=False,
        assigned_snapshot_and_correction_reverified_on_load=True,
        native_execution=False,navigation_recovered=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)


def main(proof_sha,waiter_sha,preflight=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items()) or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    run.validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive controller prefix; no retry')
    sources,proof,native_proof = prepare(proof_sha,waiter_sha)
    hw = resources()
    if preflight:
        print('MEASURED_PLANE_CONTROLLER_PREFLIGHT',len(sources),flush=True)
        return
    run.cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    run.create_output(OUTPUT)
    process = psutil.Process()
    run.write_json(OUTPUT/'launch.json',dict(source_sha256=sources,observer_completion_sha256=proof_sha,
        observer_waiter_result_sha256=waiter_sha,worker_admission_sha256=WORKER_ADMISSION_SHA,
        observer_artifact_sha256=proof['artifact_sha256'] | {'result.json':proof['result_sha256']},
        worker_artifact_sha256=native_proof['artifact_sha256'],model_state_sha256=MODEL_SHA,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline()),
        hardware=hw,environment=run.ENV,protocol=PROTOCOL,maximum_frames=run.FRAMES,
        maximum_output_bytes=MAX_OUTPUT_BYTES,stop_at_first_changed_command_or_terminal=True,
        native_execution=False,automatic_retry=False))
    print('MEASURED_PLANE_CONTROLLER_LAUNCHED',run.digest(OUTPUT/'launch.json'),len(sources),flush=True)
    start = time.perf_counter()
    try:
        report = replay()
        verify_inputs(sources,proof,native_proof,proof_sha,waiter_sha)
        run.write_json(OUTPUT/'report.json',report)
        ids = {n:run.digest(OUTPUT/n) for n in ('launch.json','context_decisions.jsonl.gz','report.json')}
        run.verify_artifacts(OUTPUT,ids)
        run.write_json(OUTPUT/'result.json',dict(status='MEASURED_PLANE_CONTROLLER_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,report=report,wall_s=time.perf_counter()-start,
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('MEASURED_PLANE_CONTROLLER_COMPLETE',run.digest(OUTPUT/'result.json'),report['boundary_comparison'],flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MEASURED_PLANE_CONTROLLER_PREFIX_FAILURE',
            reason=repr(error),automatic_retry=False,original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--observer-completion-sha256',required=True)
    parser.add_argument('--observer-waiter-result-sha256',required=True)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    main(args.observer_completion_sha256,args.observer_waiter_result_sha256,args.preflight)

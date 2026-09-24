"""Full original learned controller through a verified anchor reacquisition boundary."""
import argparse
from contextlib import closing
from itertools import islice
import json
import os
import time

import cv2
import torch

from lewm.chained_anchor_residual_controller_development import ChainedAnchorResidualController
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import verify_go2_chained_anchor_observer_completion_v2 as completed
from scripts.chained_anchor_controller_comparison_development import compare
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

observer = completed.run
native = observer.native
SOURCE = 'scripts/replay_go2_chained_anchor_controller_prefix_v1.py'
TEST = 'lewm/tests/test_chained_anchor_controller_comparison_development.py'
PROTOCOL = 'docs/go2_chained_anchor_controller_prefix_v1_2026-09-11.md'
OUTPUT = BASE/'go2_chained_anchor_controller_prefix_v1_attempt_001'
MODEL_SHA = 'fb6f1aba8830a53d67cd6c284fb24199966d5f0c63db3b2a107ab833c81c266f'
BOUNDARY = 853
MAX_OUTPUT_BYTES = 1024**3


def prepare(verification_sha):
    if observer.owner_live(completed.OWNER) or observer.owner_live(observer.CPU_OWNER):
        raise ValueError('both preceding full CPU owners must be ended')
    verify({str(completed.OUTPUT): verification_sha})
    proof = json.loads(completed.OUTPUT.read_text())
    if (proof['status'] != 'CHAINED_ANCHOR_OBSERVER_COMPLETION_VERIFIED'
            or proof['result_sha256'] != completed.RESULT_SHA or proof['launch_sha256'] != completed.LAUNCH_SHA
            or proof['owner'] != completed.OWNER or proof['owner_ended'] is not True
            or proof['public_packets_reconstructed'] != BOUNDARY+1
            or proof['original_visual_rows_reconstructed'] != BOUNDARY+1
            or proof['report']['frames'] != BOUNDARY+1
            or proof['report']['candidate_exact_original_frames'] != BOUNDARY
            or proof['report']['boundary']['frame'] != BOUNDARY
            or proof['report']['boundary_fallback']['accepted'] is not True
            or proof['report']['following_recorded_observations_consumed'] is not False):
        raise ValueError('complete fixed public-packet and observer prefix verification required')
    sources = discover_sources((SOURCE, TEST, PROTOCOL, str(completed.OUTPUT)), proof['source_sha256'])
    verify(sources)
    verify_artifacts(observer.OUTPUT, proof['observer_artifact_sha256'])
    if proof['observer_artifact_sha256'].get('result.json') != completed.RESULT_SHA:
        raise ValueError('exact verified observer result binding required')
    return sources, proof


def resources():
    hw = observer.resources()
    if hw['memory_available_bytes'] < 48*1024**3 or hw['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('16GiB controller replay plus 32GiB native reserve and 41GiB disk required')
    return hw


def verify_inputs(sources, proof, verification_sha):
    verify(sources)
    verify({str(completed.OUTPUT): verification_sha})
    verify_artifacts(observer.OUTPUT, proof['observer_artifact_sha256'])
    if observer.admit_worker(sources) != proof['worker_artifact_sha256']:
        raise ValueError('same completed negative native worker and input bindings required')


def replay():
    original_launch = read_json(native.OUTPUT, 'launch.json')
    model = native.assigned_model(original_launch)
    if state_digest(model.state_dict()) != MODEL_SHA: raise ValueError('original assigned no-RGB JEPA model required')
    controller = ChainedAnchorResidualController(model, ArticulatedCollisionGeometry(URDF),
        public_mission=public_mission(2), navigation_ticks=NAVIGATION_TICKS,
        condition=native.CASE[3], variant=native.CASE[2], persistent=True)
    directory = native.OUTPUT/native.CASE[0]
    reader = observer.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    tape = read_json(directory, 'command_tape.json')
    frames = forecasts = 0
    with observer.writer(OUTPUT) as append, closing(observer.read_rows(directory)) as originals, closing(observer.read_rows(observer.OUTPUT)) as expected:
        for row, witness in zip(islice(originals, BOUNDARY+1), expected, strict=True):
            frame = frames
            if (row['tick'] != frame or witness['tick'] != frame or row['observation_index'] != frame
                    or row['pre_sample_index'] != 749+50*frame or tape[frame]['tick'] != frame
                    or tape[frame]['completed'] is not True or tape[frame]['pre_sample_index'] != 749+50*frame
                    or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('ordered original actual command endpoints required')
            policy, depth, fast, now = reader.packet(frame)
            image, auxiliary = observer.packet(directory, frame, policy,
                observer.public_acquisition(acquisitions[frame]), now_ns=now)
            public = (policy, depth, fast, image, auxiliary)
            before = observer.fingerprint(public)
            if before != witness['public_packet_sha256']: raise ValueError('exact verified observer public input required')
            live = controller.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
            serialized = json.loads(json.dumps(live, allow_nan=False))
            try:
                if observer.fingerprint(public) != before: raise ValueError('controller mutated public inputs')
                raw = live['original_visual_evidence']
                if raw['status'] == 'CURRENT_VISUAL_POSE':
                    observer.current_dual_camera_pose(raw, policy, image, auxiliary, identity=(0, 0, 0), now_ns=now)
                if live['terminal'] is None:
                    current_measured_floor_pose(live['evidence'], identity=(0, 0, 0), now_ns=now)
                check = compare(row['decision'], serialized, tape[frame]['requested_command'], witness['candidate'],
                                frame=frame, boundary=BOUNDARY)
            except Exception as error:
                append(dict(tick=frame, decision=serialized, comparison_failure=repr(error)))
                raise
            append(dict(tick=frame, decision=serialized, comparison=check, public_input_sha256=before,
                public_input_arrays_unchanged=True, original_requested_command=tape[frame]['requested_command']))
            frames += 1
            forecasts += int(check['original_forecast_compared'])
            if (OUTPUT/observer.NAME).stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError('bounded controller receipt output exceeded')
            if frame % 50 == 0 or check['stop']: print('CHAINED_CONTROLLER_FRAME', frame, flush=True)
            if check['stop']: break
    if frames != BOUNDARY+1 or forecasts != BOUNDARY-3 or not check['stop']:
        raise ValueError('complete fixed prefix and every original forecast required')
    if state_digest(model.state_dict()) != MODEL_SHA or any(p.grad is not None for p in model.parameters()):
        raise ValueError('assigned model state or gradients changed')
    return dict(frames=frames, exact_original_decisions=BOUNDARY, original_forecasts_compared=forecasts,
        boundary_comparison=check, boundary_terminal=serialized['terminal'], boundary_failure=serialized['failure'],
        boundary_selected_action=serialized['selected_action'], boundary_requested_command=serialized['requested_command'],
        model_state_sha256=MODEL_SHA, model_state_unchanged=True, following_recorded_observations_consumed=False,
        actual_original_commands_before_intervention_exact=True, new_command_executed=False,
        original_native_failure_preserved=True, native_execution=False, navigation_qualified=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observer-verification-sha256', required=True)
    parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k, v in env.items()) or cv2.ocl.useOpenCL():
        raise ValueError('assertions and fixed CPU environment required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive full controller prefix; no retry')
    sources, proof = prepare(args.observer_verification_sha256)
    hw = resources()
    if args.source_preflight_only:
        print('CHAINED_CONTROLLER_SOURCE_PREFLIGHT', len(sources), json.dumps(hw), flush=True)
        return
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_inputs(sources, proof, args.observer_verification_sha256)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, observer_verification_sha256=args.observer_verification_sha256,
        observer_artifact_sha256=proof['observer_artifact_sha256'], input_artifact_sha256=proof['worker_artifact_sha256'],
        hardware=hw, protocol=PROTOCOL, model_state_sha256=MODEL_SHA, case=list(native.CASE), boundary_frame=BOUNDARY,
        owner_pid=os.getpid(), boot_id=observer.BOOT, environment=env,
        opencv_threads=1, blas_threads=1, implementation_class='ChainedAnchorResidualController',
        completed_worker_model_admission_reused=True, assigned_snapshot_and_coefficients_reverified_on_load=True,
        training_data_replayed=False, full_original_training_admission_reexecuted=False,
        native_execution=False, model_training=False, maximum_output_bytes=MAX_OUTPUT_BYTES))
    print('CHAINED_CONTROLLER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        report = replay()
        write_json(OUTPUT/'report.json', report)
        verify_inputs(sources, proof, args.observer_verification_sha256)
        ids = {n: digest(OUTPUT/n) for n in ('launch.json', observer.NAME, 'report.json')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='CHAINED_ANCHOR_CONTROLLER_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-start,
            native_execution=False, goal_achieved=False))
        print('CHAINED_CONTROLLER_COMPLETE', digest(OUTPUT/'result.json'), report, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CHAINED_ANCHOR_CONTROLLER_PREFIX_FAILURE',
            reason=repr(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()

"""Exclusive paired raw observer history; no hypothetical navigation outcome."""
from contextlib import closing
from itertools import islice
import json
import os
from pathlib import Path
import time
import cv2
import psutil

from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from scripts import extended_budget_anchored_maze_development as pipeline
from scripts import diagnose_go2_extended_budget_floor_boundary_v1 as diagnosis
from scripts.novel_maze_auxiliary_rgb_packet_development import public_acquisition
from scripts.navigation_artifact_root_development import BASE, artifact_path, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE = 'scripts/replay_go2_measured_plane_observer_history_v1.py'
PROTOCOL = 'docs/go2_measured_plane_observer_history_v1_2026-09-11.md'
TEST = 'lewm/tests/test_measured_plane_observer_history_development.py'
TESTS = (TEST, 'lewm/tests/test_measured_plane_rigid_fit_development.py',
    'lewm/tests/test_measured_plane_dual_camera_pose_development.py',
    'lewm/tests/test_measured_plane_residual_controller_development.py')
OUTPUT = BASE/'go2_measured_plane_observer_history_v1_attempt_001'
CPU_COMPLETION = 'docs/go2_single_pass_body_projected_completion_verification_2026-09-11.json'
CPU_SHA = 'f9833329610096f9d5776ccc208d0de33988d8ebd4e6cdff4895004aa5190f1e'
CPU_ROOT = BASE/'go2_single_pass_body_projected_late_history_v1_attempt_001'
FRAMES = 3838
MAX_OUTPUT_BYTES = 512*1024**2
ENV = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compare_original(recorded, observed, floor, floor_error, frame):
    if recorded['tick'] != frame or canonical(recorded['decision']['original_visual_evidence']) != canonical(observed):
        raise ValueError('complete original visual evidence must reproduce at frame '+str(frame))
    if frame < FRAMES-1:
        if floor_error is not None or canonical(recorded['decision']['evidence']) != canonical(floor):
            raise ValueError('complete original floor evidence must reproduce at frame '+str(frame))
    elif floor is not None or floor_error != diagnosis.FAILURE or recorded['decision']['failure'] != floor_error:
        raise ValueError('exact original terminal floor rejection required')


def floor_observe(registration, policy, depth, auxiliary, visual, now):
    if visual['terminal_failure'] is not None:
        return None, 'visual observer terminal'
    try:
        return registration.observe(policy, depth, auxiliary, visual, now_ns=now), None
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
        return None, str(error)


def admit_cpu():
    verify({CPU_COMPLETION: CPU_SHA})
    receipt = json.loads((ROOT/CPU_COMPLETION).read_text())
    verify_artifacts(CPU_ROOT, receipt['artifact_sha256'] | {'result.json': receipt['result_sha256']})
    launch = read_json(CPU_ROOT, 'launch.json')
    if (launch['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or owner_live(receipt['original_owner']) or receipt['original_owner_ended'] is not True):
        raise ValueError('completed original CPU replay owner must remain ended on recorded boot')
    return receipt


def resources():
    row = hardware()
    if row['memory_available_bytes'] < 40*1024**3 or row['artifact_free_bytes'] < 40*1024**3+MAX_OUTPUT_BYTES:
        raise ValueError('40 GiB RAM and reserved disk plus diagnostic output required')
    return row


def bind_inputs():
    verify_artifacts(diagnosis.NATIVE, diagnosis.FIXED)
    names = ['result.json', 'policy_observations.json', 'policy_histories.npz',
        'depth_observations.json', 'fast_gyro_histories.npz', 'auxiliary_camera_audit.json']
    names += [f'{kind}_{f:04d}.{suffix}' for f in range(FRAMES)
        for kind,suffix in [('rgb','png'), ('depth','npz'), ('auxiliary_rgb','png'), ('auxiliary_depth','npz')]]
    ids = diagnosis.FIXED | {diagnosis.CASE+'/'+n: digest(artifact_path(diagnosis.NATIVE, diagnosis.CASE+'/'+n)) for n in names}
    verify_artifacts(diagnosis.NATIVE, ids)
    return ids


def replay():
    directory = diagnosis.NATIVE/diagnosis.CASE
    reader = pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    if len(reader.frames) != 3848 or len(acquisitions) != 3848:
        raise ValueError('complete original extended-budget acquisition population required')
    baseline, candidate = DualCameraVisualMotion(), MeasuredPlaneVisualMotion()
    old_floor, new_floor = MeasuredFloorTransportRegistration(), MeasuredFloorTransportRegistration()
    count = refined = missing = 0
    stop = None
    with pipeline.writer(OUTPUT) as append, closing(pipeline.read_rows(directory)) as rows, (OUTPUT/'progress.jsonl').open('x') as progress:
        for frame, recorded in enumerate(islice(rows, FRAMES)):
            policy, depth, fast, now = reader.packet(frame)
            image, auxiliary = pipeline.rgb_packet(directory, frame, policy,
                public_acquisition(acquisitions[frame]), now_ns=now)
            public = (policy, depth, fast, image, auxiliary)
            before = fingerprint(public)
            old = baseline.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
            old_registered, old_error = floor_observe(old_floor, policy, depth, auxiliary, old, now)
            compare_original(recorded, old, old_registered, old_error, frame)
            if fingerprint(public) != before: raise ValueError('original mutated public packets')
            live = candidate.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
            if live['terminal_failure'] is None:
                current_dual_camera_pose(live, policy, image, auxiliary, identity=(0,0,0), now_ns=now)
            registered, error = floor_observe(new_floor, policy, depth, auxiliary, live, now)
            if fingerprint(public) != before: raise ValueError('candidate mutated public packets')
            pair = live.get('measured_plane_selected_pair')
            refined += int(live['terminal_failure'] is None and pair is not None and pair['applied'])
            missing += int(live['terminal_failure'] is None and pair is not None and not pair['applied'])
            stop = ('CANDIDATE_VISUAL_FAILURE' if live['terminal_failure'] is not None else
                'CANDIDATE_FLOOR_FAILURE' if error is not None else 'FIXED_HISTORY_END' if frame == FRAMES-1 else None)
            check = dict(frame=frame, original_visual_exact=True,
                original_floor_exact_or_terminal_reproduced=True, candidate_visual_failure=live['terminal_failure'],
                candidate_floor_failure=error, stop_reason=stop, raw_packet_sha256=before,
                original_requested_command=recorded['decision']['requested_command'], command_selected=False)
            append(dict(tick=frame, original=old, candidate=live, original_floor=old_registered,
                original_floor_error=old_error, candidate_floor=registered, comparison=check))
            count += 1
            if (OUTPUT/'context_decisions.jsonl.gz').stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError('diagnostic output allowance exceeded')
            if frame % 50 == 0 or stop:
                progress.write(json.dumps(check)+'\n'); progress.flush()
                print('MEASURED_PLANE_OBSERVER_FRAME', frame, stop, flush=True)
            if stop: break
    if not count or stop is None: raise ValueError('declared complete or negative stopping boundary required')
    return dict(frames=count, planned_frames=FRAMES, stop_reason=stop,
        complete_planned_history=count == FRAMES, refined_selected_pairs=refined,
        missing_floor_original_selected_pairs=missing, final_comparison=check,
        final_candidate_visual=live, final_candidate_floor=registered,
        full_original_evidence_reproduced_for_consumed_prefix=True,
        actual_public_fast_gyro_history_replayed=True, independent_histories_from_frame_zero=True,
        original_global_floor_gate_unchanged=True, candidate_failure_preserved=stop != 'FIXED_HISTORY_END',
        fixed_executed_trajectory_diagnostic=True, candidate_commands_selected=False,
        hypothetical_navigation_outcome_inferred=False, model_loaded=False, mapper_replayed=False,
        controller_replayed=False, native_completion_admitted=False, native_execution=False,
        navigation_recovered=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)


def main():
    if not __debug__ or any(os.environ.get(k) != v for k,v in ENV.items()) or cv2.ocl.useOpenCL():
        raise ValueError('assertions and fixed CPU environment required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive observer diagnostic; no retry')
    cpu = admit_cpu()
    hw = resources()
    verify_artifacts(diagnosis.NATIVE, diagnosis.FIXED)
    launch = read_json(diagnosis.NATIVE, 'launch.json')
    inherited = dict(cpu['source_sha256'])
    for name, sha in launch['source_sha256'].items():
        if name in inherited and inherited[name] != sha: raise ValueError('inherited source binding conflict')
        inherited[name] = sha
    sources = discover_sources((SOURCE, PROTOCOL, CPU_COMPLETION, *TESTS), inherited)
    verify(sources)
    inputs = bind_inputs()
    admit_cpu(); resources(); verify(sources)
    cv2.setNumThreads(1)
    create_output(OUTPUT)
    process = psutil.Process()
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_artifact_sha256=inputs,
        native_input_root=str(diagnosis.NATIVE), protocol=PROTOCOL, environment=ENV, hardware=hw,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        maximum_frames=FRAMES, maximum_output_bytes=MAX_OUTPUT_BYTES,
        cpu_completion_sha256=CPU_SHA, native_completion_admitted=False,
        model_loaded=False, native_execution=False, automatic_retry=False))
    print('MEASURED_PLANE_OBSERVER_LAUNCHED', digest(OUTPUT/'launch.json'), len(sources), len(inputs), flush=True)
    start = time.perf_counter()
    try:
        report = replay()
        verify(sources); verify_artifacts(diagnosis.NATIVE, inputs); admit_cpu()
        write_json(OUTPUT/'report.json', report)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json','progress.jsonl','context_decisions.jsonl.gz','report.json')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_OBSERVER_HISTORY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-start,
            native_completion_admitted=False, native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('MEASURED_PLANE_OBSERVER_COMPLETE', digest(OUTPUT/'result.json'), report['stop_reason'], flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_MEASURED_PLANE_OBSERVER_HISTORY_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()

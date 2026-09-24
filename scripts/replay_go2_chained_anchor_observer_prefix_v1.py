"""Fresh complete observer history, stopping at the first changed evidence."""
import argparse
from contextlib import closing
from itertools import islice
import json
import os
from pathlib import Path
import time

import cv2

from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.chained_anchor_visual_motion_development import ChainedAnchorVisualMotion
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import run_go2_no_rgb_jepa_direct_flow_maze02_pilot_v1 as native
from scripts import verify_go2_chained_first_bridge_pose_candidate_v1 as probe
from scripts.replay_go2_no_rgb_jepa_direct_flow_observer_prefix_v1 import canonical, resources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT

SOURCE = 'scripts/replay_go2_chained_anchor_observer_prefix_v1.py'
TEST = 'lewm/tests/test_chained_anchor_observer_prefix_development.py'
OBSERVER_TEST = 'lewm/tests/test_chained_anchor_dual_camera_pose_development.py'
PROTOCOL = 'docs/go2_chained_anchor_observer_prefix_v1_2026-09-11.md'
OUTPUT = BASE/'go2_chained_anchor_observer_prefix_v1_attempt_001'
PROBE_SHA = 'd1f581b346815994e35160270f48ee4cfc63f6bc185155b6c7cb0fd586004a03'
NATIVE_LAUNCH_SHA = '5b7a287f2ae7fd2fe1ca2745f5e7ccb980453b45d95894a7b4ab546f5aa55efe'
WORKER_SHA = 'd79f413371e4ee030156b60270cfc7077c50640b416bdca29a6d18aacc0fef4c'
WORKER_OWNER = dict(pid=2830305, created=1789122283.36, command=[
    '/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python',
    '-B', '-c', 'from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=9, pipe_handle=13)',
    '--multiprocessing-fork'])
CPU_ROOT = BASE/'go2_visibility_batched_footprint_late_history_v1_attempt_001'
CPU_LAUNCH_SHA = '6945b551f09de3153378df50fb4e5b018ae695595c5788ccd194123571b7f02a'
CPU_OWNER = dict(pid=2834244, created=1789123877.95, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/replay_go2_visibility_batched_footprint_late_history_v1.py'])
MAX_FRAMES = 864
MAX_OUTPUT_BYTES = 256*1024**2
WAIT_SECONDS = 48*3600


def compare(recorded, original, candidate, *, frame):
    if type(frame) is not int or not 0 <= frame < MAX_FRAMES:
        raise ValueError('bounded observer frame required')
    now = 1_500_000_000+frame*100_000_000
    if any(e['decision_ns'] != now for e in (recorded, original, candidate)):
        raise ValueError('exact uninterrupted observer clock required')
    if canonical(recorded) != canonical(original):
        raise ValueError('complete original visual evidence did not reproduce')
    normalized = dict(candidate)
    fallback = normalized.pop('chained_anchor_fallback', None)
    exact = canonical(normalized) == canonical(original)
    if not exact and fallback is None:
        raise ValueError('unexplained observer divergence before chained fallback')
    terminal = any(e['status'] == 'VISUAL_TERMINAL_FAILURE' for e in (original, candidate))
    return dict(frame=frame, complete_original_visual_evidence_exact=True,
        candidate_original_fields_exact=exact, fallback_attempted=fallback is not None,
        original_status=original['status'], candidate_status=candidate['status'],
        stop=not exact or terminal or frame == MAX_FRAMES-1,
        stop_reason='FIRST_CHANGED_OBSERVER_EVIDENCE' if not exact else
            'OBSERVER_TERMINAL' if terminal else 'FIXED_PREFIX_LIMIT' if frame == MAX_FRAMES-1 else None)


def prepared_sources():
    verify({str(probe.OUTPUT): PROBE_SHA})
    prior = json.loads(probe.OUTPUT.read_text())
    if prior['status'] != 'CHAINED_FIRST_BRIDGE_RECORDED_POSE_CANDIDATE_CHECKED' or prior['original_measurement_checks_pass'] is not True:
        raise ValueError('complete positive pair and recorded-pose candidate checks required')
    verify_artifacts(native.OUTPUT, {'launch.json': NATIVE_LAUNCH_SHA})
    verify_artifacts(CPU_ROOT, {'launch.json': CPU_LAUNCH_SHA})
    inherited = merge_sources(prior['source_sha256'], read_json(native.OUTPUT, 'launch.json')['source_sha256'])
    inherited = merge_sources(inherited, read_json(CPU_ROOT, 'launch.json')['source_sha256'])
    return discover_sources((SOURCE, TEST, OBSERVER_TEST, PROTOCOL, str(probe.OUTPUT)), inherited)


def admit_worker(sources):
    verify(sources)
    if owner_live(WORKER_OWNER): raise ValueError('original raw worker must be ended')
    name = native.CASE[0]
    terminal = name+'_worker_terminal.json'
    verify_artifacts(native.OUTPUT, {'launch.json': NATIVE_LAUNCH_SHA, terminal: WORKER_SHA})
    launch = read_json(native.OUTPUT, 'launch.json')
    record = read_json(native.OUTPUT, terminal)
    bindings = record['artifact_sha256'] | {'launch.json': NATIVE_LAUNCH_SHA,
        terminal: WORKER_SHA, name+'_worker.log': record['worker_log_sha256']}
    verify_artifacts(native.OUTPUT, bindings)
    native.require_worker(record, read_json(native.OUTPUT, name+'_audit.json'), launch['input_admission']['prefix_report'])
    if record['collection']['decisions'] != 874 or record['verified_round_trip'] is not False:
        raise ValueError('same completed negative tracking worker required')
    return bindings


def wait_for_cpu(sources, event, *, sleep=time.sleep, clock=time.monotonic):
    start = clock()
    while owner_live(CPU_OWNER):
        if clock()-start >= WAIT_SECONDS: raise ValueError('CPU wait expired; no replacement attempt')
        event('WAITING_FOR_VISIBILITY_REPLAY', owner=CPU_OWNER)
        sleep(30)
    verify_artifacts(CPU_ROOT, {'launch.json': CPU_LAUNCH_SHA})
    # CPU ordering is operational, not a scientific dependency on speedup.
    terminal = [n for n in ('result.json', 'failure.json') if (CPU_ROOT/n).is_file()]
    if len(terminal) != 1: raise ValueError('ended CPU owner requires one preserved terminal record')
    name = terminal[0]
    result = read_json(CPU_ROOT, name)
    bindings = {'launch.json': CPU_LAUNCH_SHA, name: digest(CPU_ROOT/name)}
    if name == 'result.json':
        bindings.update(result['artifact_sha256'])
        if bindings['launch.json'] != CPU_LAUNCH_SHA:
            raise ValueError('original CPU launch binding required')
    verify_artifacts(CPU_ROOT, bindings)
    verify(sources)
    return dict(original_cpu_owner_ended=True, terminal_name=name, terminal_status=result['status'],
                artifact_sha256=bindings, no_replay_restarted=True)


def replay(directory):
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    baseline, candidate = DirectFlowDualCameraVisualMotion(), ChainedAnchorVisualMotion()
    frames = exact = 0
    with writer(OUTPUT) as append, closing(read_rows(directory)) as rows:
        for frame, row in enumerate(islice(rows, MAX_FRAMES)):
            if row['tick'] != frame or row['observation_index'] != frame:
                raise ValueError('complete original sequential observation identity required')
            policy, depth, fast, now = reader.packet(frame)
            image, auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
            public = (policy, depth, fast, image, auxiliary)
            before = fingerprint(public)
            old = baseline.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
            if fingerprint(public) != before: raise ValueError('original observer mutated public inputs')
            live = candidate.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=auxiliary, now_ns=now)
            if fingerprint(public) != before: raise ValueError('candidate observer mutated public inputs')
            for evidence in (old, live):
                if evidence['status'] == 'CURRENT_VISUAL_POSE':
                    current_dual_camera_pose(evidence, policy, image, auxiliary, identity=(0, 0, 0), now_ns=now)
            check = compare(row['decision']['original_visual_evidence'], old, live, frame=frame)
            append(dict(tick=frame, original=old, candidate=live, comparison=check,
                public_packet_sha256=before, public_inputs_unchanged=True,
                original_requested_command=row['decision']['requested_command'], command_selected=False))
            frames += 1
            exact += int(check['candidate_original_fields_exact'])
            if (OUTPUT/NAME).stat().st_size > MAX_OUTPUT_BYTES: raise ValueError('bounded observer output exceeded')
            if frame % 50 == 0 or check['stop']: print('CHAINED_OBSERVER_FRAME', frame, flush=True)
            if check['stop']: break
    if not frames or not check['stop']: raise ValueError('complete prefix to a declared boundary required')
    return dict(frames=frames, candidate_exact_original_frames=exact, boundary=check,
        boundary_fallback=live.get('chained_anchor_fallback'), candidate_current_pose_at_boundary=live['current_pose'],
        original_terminal_failure=old['terminal_failure'], candidate_terminal_failure=live['terminal_failure'],
        image_history_frames=len(candidate.model._image_history), complete_original_visual_evidence_reproduced=True,
        public_inputs_unchanged=True, full_observer_history_from_frame_zero=True, actual_fast_gyro_history_replayed=True,
        following_recorded_observations_consumed=False, command_selected=False, floor_registration_replayed=False,
        mapping_replayed=False, model_loaded=False, native_execution=False, navigation_qualified=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k, v in env.items()) or cv2.ocl.useOpenCL():
        raise ValueError('assertions and fixed CPU environment required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive observer prefix; no retry')
    sources = prepared_sources()
    verify(sources)
    hw = resources()
    if args.source_preflight_only:
        print('CHAINED_OBSERVER_SOURCE_PREFLIGHT', len(sources), json.dumps(hw), flush=True)
        return
    inputs = admit_worker(sources)
    cv2.setNumThreads(1)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_artifact_sha256=inputs,
        worker_terminal_sha256=WORKER_SHA, native_launch_sha256=NATIVE_LAUNCH_SHA, probe_sha256=PROBE_SHA,
        hardware=hw, protocol=PROTOCOL, cpu_owner=CPU_OWNER, cpu_launch_sha256=CPU_LAUNCH_SHA,
        boot_id=BOOT, owner_pid=os.getpid(), environment=env, maximum_wait_s=WAIT_SECONDS,
        maximum_frames=MAX_FRAMES, maximum_output_bytes=MAX_OUTPUT_BYTES,
        original_class='DirectFlowDualCameraVisualMotion', candidate_class='ChainedAnchorVisualMotion',
        actual_recorded_fast_gyro_used=True, full_controller_replay=False, model_loaded=False,
        native_execution=False, command_selected=False, opencv_threads=1,
        stop_at_first_changed_observer_evidence=True, automatic_retry=False))
    print('CHAINED_OBSERVER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **details):
                events.write(json.dumps(dict(status=status, elapsed_s=time.perf_counter()-started, **details))+'\n')
                events.flush()
                print(status, flush=True)
            completion = wait_for_cpu(sources, event)
            write_json(OUTPUT/'cpu_completion.json', completion)
            resources()
            verify(sources)
            report = replay(native.OUTPUT/native.CASE[0])
        if admit_worker(sources) != inputs: raise ValueError('original worker input binding changed')
        verify_artifacts(CPU_ROOT, completion['artifact_sha256'])
        write_json(OUTPUT/'report.json', report)
        ids = {n: digest(OUTPUT/n) for n in ('launch.json', 'events.jsonl', 'cpu_completion.json', NAME, 'report.json')}
        verify(sources)
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='CHAINED_ANCHOR_OBSERVER_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-started,
            native_execution=False, goal_achieved=False))
        print('CHAINED_OBSERVER_COMPLETE', digest(OUTPUT/'result.json'), report['boundary'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CHAINED_ANCHOR_OBSERVER_PREFIX_FAILURE',
            reason=repr(error), automatic_retry=False, original_work_retained=True))
        raise


if __name__ == '__main__':
    main()

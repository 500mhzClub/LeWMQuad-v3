"""Replay complete contact-worker visual history to the first changed evidence."""
import argparse
from contextlib import closing
from itertools import islice
import json
import os
from pathlib import Path
import time
from types import FunctionType

import cv2

from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from scripts import replay_go2_no_rgb_jepa_direct_flow_observer_prefix_v1 as original
from scripts import probe_go2_contact_anchored_direct_flow_pair_v1 as probe
from scripts import sustained_hold_reorientation_native_inputs_development as previous
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT

SOURCE = 'scripts/replay_go2_contact_anchored_direct_flow_observer_prefix_v1.py'
TEST = 'lewm/tests/test_contact_anchored_direct_flow_observer_prefix_development.py'
PROTOCOL = 'docs/go2_contact_anchored_direct_flow_observer_prefix_v1_2026-09-11.md'
OUTPUT = BASE/'go2_contact_anchored_direct_flow_observer_prefix_v1_attempt_001'
PROBE_SHA = '568bd6f7b13428ccd96291eb9945ec33dd72591a79549a7a4b0f841fc388848b'
MAX_FRAMES = 562
MAX_OUTPUT_BYTES = original.MAX_OUTPUT_BYTES
WAIT_SECONDS = 48*3600
canonical = original.canonical
compare = FunctionType(original.compare.__code__, original.compare.__globals__ | dict(MAX_FRAMES=MAX_FRAMES), 'compare')


def prepared_sources():
    verify({str(probe.OUTPUT.relative_to(ROOT)):PROBE_SHA})
    p = json.loads(probe.OUTPUT.read_text())
    if p['status'] != 'CONTACT_ANCHORED_DIRECT_FLOW_PAIR_PROBE_COMPLETE':
        raise ValueError('exact completed contact-pair probe required')
    verify_artifacts(previous.replay.OUTPUT, {'launch.json':previous.prefix.LAUNCH_SHA})
    sources = merge_sources(p['source_sha256'], read_json(previous.replay.OUTPUT, 'launch.json')['source_sha256'])
    return discover_sources((SOURCE, TEST, PROTOCOL, str(probe.OUTPUT.relative_to(ROOT))), sources)


def admit_worker(sources):
    verify(sources); verify({str(probe.OUTPUT.relative_to(ROOT)):PROBE_SHA,
        str(probe.diagnosis.OUTPUT.relative_to(ROOT)):probe.DIAGNOSIS_SHA})
    d = json.loads(probe.diagnosis.OUTPUT.read_text())
    if (d['status'] != 'CONTACT_ANCHORED_WORKER_RAW_TRACKING_FAILURE_RECONSTRUCTED'
            or d['worker_terminal_sha256'] != probe.diagnosis.WORKER_SHA
            or d['terminal_observation_frame'] != MAX_FRAMES-1):
        raise ValueError('same diagnosed completed worker and fixed terminal frame required')
    probe.diagnosis.worker_ended()
    verify_artifacts(probe.diagnosis.native.OUTPUT, d['artifact_sha256'])
    return d['artifact_sha256']


def wait_for_cpu(sources, event, *, sleep=time.sleep, clock=time.monotonic):
    start = clock()
    while True:
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
            raise ValueError('original CPU owner boot required')
        if not owner_live(previous.RAW_OWNER): break
        if clock()-start >= WAIT_SECONDS:
            raise ValueError('original CPU wait expired; no restart or replacement')
        event('WAITING_FOR_ORIGINAL_SUSTAINED_RAW_REPLAY', owner=previous.RAW_OWNER)
        sleep(30)
    root = previous.replay.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original CPU replay failure must be retained')
    sha = digest(root/'result.json'); result = read_json(root, 'result.json')
    bindings = result['artifact_sha256'] | {'result.json':sha}
    verify_artifacts(root, bindings); verify_artifacts(root, {'launch.json':previous.prefix.LAUNCH_SHA})
    launch = read_json(root, 'launch.json')
    if (result['status'] != 'SUSTAINED_HOLD_REORIENTATION_RAW_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n,h in result['source_sha256'].items())
            or bindings['launch.json'] != previous.prefix.LAUNCH_SHA):
        raise ValueError('exact completed original CPU replay required')
    previous.prefix.boundary(result['report']); verify(sources)
    return dict(original_cpu_result_sha256=sha, original_cpu_artifact_sha256=bindings,
        original_cpu_owner_ended=True, no_replay_restarted=True)


def replay(directory):
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    baseline = DualCameraVisualMotion(identity=(0,0,0))
    candidate = DirectFlowDualCameraVisualMotion(identity=(0,0,0))
    frames = exact = 0
    with writer(OUTPUT) as append, closing(read_rows(directory)) as original_rows:
        for frame, row in enumerate(islice(original_rows, MAX_FRAMES)):
            if row['tick'] != frame or row['observation_index'] != frame:
                raise ValueError('complete original sequential observation identity required')
            policy, depth, fast, now = reader.packet(frame)
            image, aux = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
            public = (policy, depth, fast, image, aux); before = fingerprint(public)
            old = baseline.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=aux, now_ns=now)
            if fingerprint(public) != before: raise ValueError('original observer mutated public inputs')
            live = candidate.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=aux, now_ns=now)
            if fingerprint(public) != before: raise ValueError('candidate observer mutated public inputs')
            for evidence in (old, live):
                if evidence['status'] == 'CURRENT_VISUAL_POSE':
                    current_dual_camera_pose(evidence, policy, image, aux, identity=(0,0,0), now_ns=now)
            check = compare(row['decision']['original_visual_evidence'], old, live, frame=frame)
            append(dict(tick=frame, original=old, candidate=live, comparison=check,
                public_packet_sha256=before, public_inputs_unchanged=True,
                original_requested_command=row['decision']['requested_command'], command_selected=False))
            frames += 1; exact += int(check['candidate_original_fields_exact'])
            if (OUTPUT/NAME).stat().st_size > MAX_OUTPUT_BYTES: raise ValueError('bounded observer output exceeded')
            if frame % 50 == 0 or check['stop']: print('CONTACT_FLOW_OBSERVER_FRAME', frame, flush=True)
            if check['stop']: break
    if not frames or not check['stop']: raise ValueError('complete prefix to a declared boundary required')
    return dict(frames=frames, candidate_exact_original_frames=exact, boundary=check,
        boundary_fallback=live.get('direct_corner_flow_fallback'), candidate_current_pose_at_boundary=live['current_pose'],
        original_terminal_failure=old['terminal_failure'], candidate_terminal_failure=live['terminal_failure'],
        complete_original_visual_evidence_reproduced=True, public_inputs_unchanged=True,
        full_observer_history_from_frame_zero=True, actual_fast_gyro_history_replayed=True,
        following_recorded_observations_consumed=False, command_selected=False,
        floor_registration_replayed=False, mapping_replayed=False, model_loaded=False,
        native_execution=False, navigation_qualified=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true'); args = parser.parse_args()
    env = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')
    if not __debug__ or any(os.environ.get(k) != v for k,v in env.items()) or cv2.ocl.useOpenCL():
        raise ValueError('assertions and fixed CPU environment required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive contact observer prefix; no retry')
    sources = prepared_sources(); verify(sources); hw = original.resources()
    if args.source_preflight_only:
        print('CONTACT_FLOW_OBSERVER_SOURCE_PREFLIGHT', len(sources), json.dumps(hw), flush=True); return
    inputs = admit_worker(sources); cv2.setNumThreads(1); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_artifact_sha256=inputs,
        probe_sha256=PROBE_SHA, diagnosis_sha256=probe.DIAGNOSIS_SHA, hardware=hw, protocol=PROTOCOL,
        previous_cpu_owner=previous.RAW_OWNER, previous_cpu_launch_sha256=previous.prefix.LAUNCH_SHA,
        boot_id=BOOT, owner_pid=os.getpid(), environment=env, maximum_wait_s=WAIT_SECONDS,
        maximum_frames=MAX_FRAMES, maximum_output_bytes=MAX_OUTPUT_BYTES,
        original_class='DualCameraVisualMotion', candidate_class='DirectFlowDualCameraVisualMotion',
        actual_recorded_fast_gyro_used=True, full_controller_replay=False, floor_registration_replayed=False,
        model_loaded=False, native_execution=False, command_selected=False, opencv_threads=1,
        stop_at_first_changed_observer_evidence=True, automatic_retry=False))
    print('CONTACT_FLOW_OBSERVER_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True); started = time.perf_counter()
    try:
        with (OUTPUT/'events.jsonl').open('x') as events:
            def event(status, **kw):
                events.write(json.dumps(dict(status=status, elapsed_s=time.perf_counter()-started, **kw))+'\n'); events.flush()
                print(status, kw, flush=True)
            completion = wait_for_cpu(sources, event)
            write_json(OUTPUT/'cpu_completion.json', completion)
            original.resources(); verify(sources)
            report = replay(probe.diagnosis.native.OUTPUT/probe.diagnosis.native.CASE[0])
        if admit_worker(sources) != inputs: raise ValueError('original worker admission changed')
        verify_artifacts(previous.replay.OUTPUT, completion['original_cpu_artifact_sha256'])
        write_json(OUTPUT/'report.json', report)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json', 'events.jsonl', 'cpu_completion.json', NAME, 'report.json')}
        verify(sources); verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='CONTACT_ANCHORED_DIRECT_FLOW_OBSERVER_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-started,
            native_execution=False, goal_achieved=False))
        print('CONTACT_FLOW_OBSERVER_COMPLETE', digest(OUTPUT/'result.json'), report['boundary'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CONTACT_FLOW_OBSERVER_PREFIX_FAILURE',
            reason=repr(error), automatic_retry=False, original_work_retained=True)); raise


if __name__ == '__main__': main()

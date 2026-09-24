"""Recorded sensor-only observer prefix, ending at its first changed evidence."""
import json
import time
from contextlib import closing
from itertools import islice

import cv2

from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts import probe_go2_no_rgb_jepa_direct_flow_pairs_v1 as probe
from scripts import diagnose_go2_no_rgb_jepa_maze02_matches_v1 as diagnosis
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.maze_decision_stream_development import read_rows, writer, NAME
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE = 'scripts/replay_go2_no_rgb_jepa_direct_flow_observer_prefix_v1.py'
TEST = 'lewm/tests/test_no_rgb_jepa_direct_flow_observer_prefix_development.py'
PROTOCOL = 'docs/go2_no_rgb_jepa_direct_flow_observer_prefix_v1_2026-09-10.md'
OUTPUT = BASE/'go2_no_rgb_jepa_direct_flow_observer_prefix_v1_attempt_001'
PROBE_SHA = '4db83292904deee9bd35e10079e6a9dc424c0e8ec78567bf1829fbeb9621977b'
MAX_FRAMES = 860
MAX_OUTPUT_BYTES = 256*1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compare(recorded, original, candidate, *, frame):
    """No changed observation may be followed by a claimed causal replay future."""
    if type(frame) is not int or not 0 <= frame < MAX_FRAMES:
        raise ValueError('bounded observer frame required')
    now = 1_500_000_000+frame*100_000_000
    for evidence in (recorded, original, candidate):
        if evidence['decision_ns'] != now:
            raise ValueError('exact uninterrupted observer clock required')
    if canonical(recorded) != canonical(original):
        raise ValueError('complete original visual evidence did not reproduce')
    normalized = dict(candidate)
    fallback = normalized.pop('direct_corner_flow_fallback', None)
    exact = canonical(normalized) == canonical(original)
    if not exact and fallback is None:
        raise ValueError('unexplained observer divergence before fallback')
    terminal = any(e['status'] == 'VISUAL_TERMINAL_FAILURE' for e in (original, candidate))
    return dict(frame=frame, complete_original_visual_evidence_exact=True,
        candidate_original_fields_exact=exact, fallback_attempted=fallback is not None,
        original_status=original['status'], candidate_status=candidate['status'],
        stop=not exact or terminal or frame == MAX_FRAMES-1,
        stop_reason='FIRST_CHANGED_OBSERVER_EVIDENCE' if not exact else
            'OBSERVER_TERMINAL' if terminal else 'FIXED_PREFIX_LIMIT' if frame == MAX_FRAMES-1 else None)


def resources():
    row = hardware()
    if row['memory_available_bytes'] < 40*1024**3 or row['artifact_free_bytes'] < 40*1024**3+MAX_OUTPUT_BYTES:
        raise ValueError('8GiB observer plus 32GiB concurrent native RAM and reserved output space required')
    return row


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive observer prefix required')
    verify_artifacts(probe.OUTPUT, {'result.json': PROBE_SHA})
    predecessor = read_json(probe.OUTPUT, 'result.json')
    assert predecessor['status'] == 'NO_RGB_JEPA_DIRECT_FLOW_PAIR_PROBE_V1_COMPLETE'
    probe_ids = {'result.json': PROBE_SHA} | predecessor['artifact_sha256']
    verify_artifacts(probe.OUTPUT, probe_ids); verify(predecessor['source_sha256'])
    inputs = read_json(probe.OUTPUT, 'launch.json')['input_artifact_sha256']
    verify_artifacts(diagnosis.INPUT, inputs)
    sources = discover_sources((SOURCE, TEST, PROTOCOL), predecessor['source_sha256']); verify(sources)
    hw = resources(); cv2.setNumThreads(1)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_artifact_sha256=inputs,
        predecessor_result_sha256=PROBE_SHA, predecessor_artifact_sha256=probe_ids, hardware=hw,
        maximum_frames=MAX_FRAMES, maximum_output_bytes=MAX_OUTPUT_BYTES, protocol=PROTOCOL,
        original_class='DualCameraVisualMotion', candidate_class='DirectFlowDualCameraVisualMotion',
        actual_recorded_fast_gyro_used=True, model_loaded=False, native_execution=False,
        full_controller_replay=False, floor_registration_replayed=False, command_selected=False,
        opencv_threads=1, stop_at_first_changed_observer_evidence=True))
    print('NO_RGB_JEPA_DIRECT_FLOW_OBSERVER_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        directory = diagnosis.INPUT/diagnosis.CASE
        reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
        baseline = DualCameraVisualMotion(identity=(0,0,0))
        candidate = DirectFlowDualCameraVisualMotion(identity=(0,0,0))
        frames = exact = 0
        with writer(OUTPUT) as append, closing(read_rows(directory)) as original_rows:
            for frame, row in enumerate(islice(original_rows, MAX_FRAMES)):
                assert row['tick'] == frame and row['observation_index'] == frame
                policy, depth, fast, now = reader.packet(frame)
                image, aux = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
                public = (policy, depth, fast, image, aux)
                before = fingerprint(public)
                original = baseline.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=aux, now_ns=now)
                if fingerprint(public) != before: raise ValueError('original observer mutated public inputs')
                live = candidate.observe(policy, depth, fast, auxiliary_rgb=image, auxiliary_depth=aux, now_ns=now)
                if fingerprint(public) != before: raise ValueError('candidate observer mutated public inputs')
                for evidence in (original, live):
                    if evidence['status'] == 'CURRENT_VISUAL_POSE':
                        current_dual_camera_pose(evidence, policy, image, aux, identity=(0,0,0), now_ns=now)
                check = compare(row['decision']['original_visual_evidence'], original, live, frame=frame)
                append(dict(tick=frame, original=original, candidate=live, comparison=check,
                    public_packet_sha256=before, public_inputs_unchanged=True,
                    original_requested_command=row['decision']['requested_command'], command_selected=False))
                frames += 1; exact += int(check['candidate_original_fields_exact'])
                if (OUTPUT/NAME).stat().st_size > MAX_OUTPUT_BYTES:
                    raise ValueError('bounded observer output exceeded')
                if frame % 50 == 0: print('DIRECT_FLOW_OBSERVER_PREFIX_FRAME', frame, flush=True)
                if check['stop']: break
        if not frames or not check['stop']: raise ValueError('complete prefix to a declared boundary required')
        report = dict(frames=frames, candidate_exact_original_frames=exact, boundary=check,
            boundary_fallback=live.get('direct_corner_flow_fallback'),
            candidate_current_pose_at_boundary=live['current_pose'],
            original_terminal_failure=original['terminal_failure'], candidate_terminal_failure=live['terminal_failure'],
            complete_original_visual_evidence_reproduced=True, public_inputs_unchanged=True,
            full_observer_history_from_frame_zero=True, actual_fast_gyro_history_replayed=True,
            following_recorded_observations_consumed=False, command_selected=False,
            floor_registration_replayed=False, mapping_replayed=False, model_loaded=False,
            native_execution=False, navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'report.json', report)
        verify(sources); verify_artifacts(diagnosis.INPUT, inputs); verify_artifacts(probe.OUTPUT, probe_ids)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json', NAME, 'report.json')}
        verify_artifacts(OUTPUT, ids)
        write_json(OUTPUT/'result.json', dict(status='NO_RGB_JEPA_DIRECT_FLOW_OBSERVER_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report, wall_s=time.perf_counter()-started,
            native_execution=False, goal_achieved=False))
        print('DIRECT_FLOW_OBSERVER_PREFIX_COMPLETE', digest(OUTPUT/'result.json'), check, flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_DIRECT_FLOW_OBSERVER_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()

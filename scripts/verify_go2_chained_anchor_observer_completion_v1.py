"""Reconstruct completed observer comparisons and every consumed public packet."""
from contextlib import closing
from itertools import islice
import json
from pathlib import Path

from scripts import replay_go2_chained_anchor_observer_prefix_v1 as run
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/verify_go2_chained_anchor_observer_completion_v1.py'
TEST = 'lewm/tests/test_chained_anchor_observer_completion_development.py'
OUTPUT = Path('docs/go2_chained_anchor_observer_completion_2026-09-11.json')
RESULT_SHA = '5cbce75a34aa610d7cf227b772a757509fb213cfb1a0980d7f9ffb6cb7ac2c9e'
LAUNCH_SHA = 'c1ee58ed93cd71dae49198747f8c3be281cd24850051e1379e29c1b1a11edf18'
OWNER = dict(pid=2838943, created=1789126094.17, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', run.SOURCE])


def check_row(recorded, observed, *, frame, public_sha):
    if (recorded['tick'] != frame or recorded['observation_index'] != frame or observed['tick'] != frame
            or observed['public_packet_sha256'] != public_sha or observed['public_inputs_unchanged'] is not True
            or observed['original_requested_command'] != recorded['decision']['requested_command']
            or observed['command_selected'] is not False):
        raise ValueError('actual ordered public packets and original command witness required')
    check = run.compare(recorded['decision']['original_visual_evidence'], observed['original'], observed['candidate'], frame=frame)
    if run.canonical(check) != run.canonical(observed['comparison']):
        raise ValueError('complete saved comparison must reconstruct')
    if check['stop'] is not (frame == 853):
        raise ValueError('exact first changed evidence boundary required')
    return check


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completed observer verification required')
    if run.owner_live(OWNER): raise ValueError('observer owner must be ended')
    verify_artifacts(run.OUTPUT, {'result.json': RESULT_SHA, 'launch.json': LAUNCH_SHA})
    result = read_json(run.OUTPUT, 'result.json')
    launch = read_json(run.OUTPUT, 'launch.json')
    if (result['status'] != 'CHAINED_ANCHOR_OBSERVER_PREFIX_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA):
        raise ValueError('complete exact observer result and launch required')
    sources = discover_sources((SOURCE, TEST), result['source_sha256'])
    verify(sources)
    ids = result['artifact_sha256'] | {'result.json': RESULT_SHA}
    verify_artifacts(run.OUTPUT, ids)
    inputs = run.admit_worker(sources)
    if inputs != launch['input_artifact_sha256']: raise ValueError('same completed worker artifact bindings required')
    directory = run.native.OUTPUT/run.native.CASE[0]
    reader = run.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    count = exact = 0
    with closing(run.read_rows(directory)) as recorded_rows, closing(run.read_rows(run.OUTPUT)) as observed_rows:
        for frame, (recorded, observed) in enumerate(zip(islice(recorded_rows, 854), observed_rows, strict=True)):
            policy, depth, fast, now = reader.packet(frame)
            image, auxiliary = run.packet(directory, frame, policy, run.public_acquisition(acquisitions[frame]), now_ns=now)
            sha = run.fingerprint((policy, depth, fast, image, auxiliary))
            check = check_row(recorded, observed, frame=frame, public_sha=sha)
            for evidence in (observed['original'], observed['candidate']):
                if evidence['status'] == 'CURRENT_VISUAL_POSE':
                    run.current_dual_camera_pose(evidence, policy, image, auxiliary, identity=(0, 0, 0), now_ns=now)
            count += 1
            exact += int(check['candidate_original_fields_exact'])
    if (count, exact) != (854, 853): raise ValueError('complete actual observer prefix required')
    old, live = observed['original'], observed['candidate']
    reconstructed = dict(frames=count, candidate_exact_original_frames=exact, boundary=check,
        boundary_fallback=live.get('chained_anchor_fallback'), candidate_current_pose_at_boundary=live['current_pose'],
        original_terminal_failure=old['terminal_failure'], candidate_terminal_failure=live['terminal_failure'],
        image_history_frames=33, complete_original_visual_evidence_reproduced=True, public_inputs_unchanged=True,
        full_observer_history_from_frame_zero=True, actual_fast_gyro_history_replayed=True,
        following_recorded_observations_consumed=False, command_selected=False, floor_registration_replayed=False,
        mapping_replayed=False, model_loaded=False, native_execution=False, navigation_qualified=False, goal_achieved=False)
    if run.canonical(reconstructed) != run.canonical(result['report']):
        raise ValueError('entire completed observer report must reconstruct')
    fallback = reconstructed['boundary_fallback']
    if (fallback['accepted'] is not True or fallback['selected_reference'] != 850
            or fallback['selected_camera'] != 'auxiliary' or fallback['selected_continuity_status'] != 'ANCHOR_MEASUREMENT'
            or fallback['original_qualified_measurements_checked'] < 1
            or reconstructed['candidate_current_pose_at_boundary']['promoted_keyframe'] is not True):
        raise ValueError('actual accepted and promoted retained anchor required')
    verify(sources)
    verify_artifacts(run.OUTPUT, ids)
    verify_artifacts(run.native.OUTPUT, inputs)
    if run.owner_live(OWNER): raise ValueError('completed observer owner identity changed')
    write_json(OUTPUT, dict(status='CHAINED_ANCHOR_OBSERVER_COMPLETION_VERIFIED', source_sha256=sources,
        result_sha256=RESULT_SHA, launch_sha256=LAUNCH_SHA, owner=OWNER, owner_ended=True,
        observer_artifact_sha256=ids, worker_artifact_sha256=inputs, report=reconstructed,
        public_packets_reconstructed=count, original_visual_rows_reconstructed=count,
        observer_reexecuted=False, controller_replayed=False, native_execution=False, goal_achieved=False))
    print('CHAINED_OBSERVER_COMPLETION_VERIFIED', digest(OUTPUT), count, exact, flush=True)


if __name__ == '__main__':
    main()

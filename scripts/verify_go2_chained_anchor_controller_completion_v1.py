"""Authenticate the completed controller replay without rerunning its policy."""
from contextlib import closing
from itertools import islice
import json
from pathlib import Path

from scripts import replay_go2_chained_anchor_controller_prefix_v1 as run
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/verify_go2_chained_anchor_controller_completion_v1.py'
TEST = 'lewm/tests/test_chained_anchor_controller_completion_development.py'
OUTPUT = Path('docs/go2_chained_anchor_controller_completion_verification_2026-09-11.json')
EXECUTION = Path('docs/go2_chained_anchor_controller_execution_2026-09-11.json')
EXECUTION_SHA = '073bae2e585dc350e218f7d6b25bada3232b51ea0eca71be16e9202cd0a2bad3'
RESULT_SHA = 'd68e5e48916ff84d4034c95a2af5357695d277048519a15cf5dbb6a554703231'
LAUNCH_SHA = '8eba2f8dfea706109f8cec4fcf55206f36fa3b9f588c0bb2c344b492e95269cd'
OBSERVER_SHA = '3ba5a9cdf6bf2a1a4dbd46a95481872fd02aed1acbd7bea0a017612dc515575b'
OWNER = dict(pid=2840884, created=1789126982.74, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B', run.SOURCE,
    '--observer-verification-sha256', OBSERVER_SHA], boot_id=run.observer.BOOT)


def equal(left, right):
    return run.observer.canonical(left) == run.observer.canonical(right)


def check_row(original, saved, visual, command, *, frame, public_sha):
    if (type(frame) is not int or not 0 <= frame <= run.BOUNDARY
            or any(type(row.get('tick')) is not int or row['tick'] != frame
                   for row in (original, saved, visual, command))
            or original['observation_index'] != frame
            or original['pre_sample_index'] != 749+50*frame
            or command['pre_sample_index'] != 749+50*frame
            or command['post_sample_index'] != 799+50*frame
            or command['completed'] is not True
            or saved['public_input_arrays_unchanged'] is not True
            or saved['public_input_sha256'] != public_sha
            or visual['public_packet_sha256'] != public_sha
            or not equal(saved['original_requested_command'], command['requested_command'])):
        raise ValueError('ordered actual public packets and completed original commands required')
    check = run.compare(original['decision'], saved['decision'], command['requested_command'],
                        visual['candidate'], frame=frame, boundary=run.BOUNDARY)
    if not equal(check, saved['comparison']) or check['stop'] is not (frame == run.BOUNDARY):
        raise ValueError('entire saved comparison and exact intervention boundary must reconstruct')
    return check


def reconstruct_report(count, exact, forecasts, saved):
    if (count, exact, forecasts) != (854, 853, 850):
        raise ValueError('complete fixed observation, decision and forecast populations required')
    decision = saved['decision']
    return dict(frames=count, exact_original_decisions=exact, original_forecasts_compared=forecasts,
        boundary_comparison=saved['comparison'], boundary_terminal=decision['terminal'],
        boundary_failure=decision['failure'], boundary_selected_action=decision['selected_action'],
        boundary_requested_command=decision['requested_command'], model_state_sha256=run.MODEL_SHA,
        model_state_unchanged=True, following_recorded_observations_consumed=False,
        actual_original_commands_before_intervention_exact=True, new_command_executed=False,
        original_native_failure_preserved=True, native_execution=False, navigation_qualified=False, goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive completion verification required')
    if run.observer.owner_live(OWNER): raise ValueError('controller replay owner must be ended')
    if (run.OUTPUT/'failure.json').exists(): raise ValueError('failed replay cannot be admitted')
    verify({str(EXECUTION): EXECUTION_SHA})
    execution = json.loads(EXECUTION.read_text())
    if execution['owner'] != OWNER or execution['launch_sha256'] != LAUNCH_SHA:
        raise ValueError('original live execution identity must match')
    verify_artifacts(run.OUTPUT, {'result.json': RESULT_SHA, 'launch.json': LAUNCH_SHA})
    result, launch = (read_json(run.OUTPUT, n) for n in ('result.json', 'launch.json'))
    if (result['status'] != 'CHAINED_ANCHOR_CONTROLLER_PREFIX_V1_COMPLETE'
            or result['native_execution'] is not False or result['goal_achieved'] is not False
            or result['source_sha256'] != launch['source_sha256']
            or set(result['artifact_sha256']) != {'launch.json', run.observer.NAME, 'report.json'}
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or launch['owner_pid'] != OWNER['pid'] or launch['boot_id'] != OWNER['boot_id']
            or launch['boundary_frame'] != run.BOUNDARY or launch['model_state_sha256'] != run.MODEL_SHA
            or launch['case'] != list(run.native.CASE)
            or launch['implementation_class'] != 'ChainedAnchorResidualController'
            or launch['observer_verification_sha256'] != OBSERVER_SHA):
        raise ValueError('exact completed controller, model, source and launch required')
    sources = discover_sources((SOURCE, TEST, str(EXECUTION)), result['source_sha256'])
    verify(sources)
    ids = result['artifact_sha256'] | {'result.json': RESULT_SHA}
    verify_artifacts(run.OUTPUT, ids)
    if not equal(read_json(run.OUTPUT, 'report.json'), result['report']):
        raise ValueError('bound report differs from result')
    _, proof = run.prepare(OBSERVER_SHA)
    if (launch['observer_artifact_sha256'] != proof['observer_artifact_sha256']
            or launch['input_artifact_sha256'] != proof['worker_artifact_sha256']):
        raise ValueError('same authenticated observer and original worker required')
    run.verify_inputs(sources, proof, OBSERVER_SHA)
    directory = run.native.OUTPUT/run.native.CASE[0]
    reader = run.observer.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    tape = read_json(directory, 'command_tape.json')
    count = exact = forecasts = 0
    with closing(run.observer.read_rows(directory)) as originals, \
            closing(run.observer.read_rows(run.OUTPUT)) as saved_rows, \
            closing(run.observer.read_rows(run.observer.OUTPUT)) as visuals:
        for frame, (old, saved, visual) in enumerate(zip(islice(originals, 854), saved_rows, visuals, strict=True)):
            p, d, fast, now = reader.packet(frame)
            image, aux = run.observer.packet(directory, frame, p,
                run.observer.public_acquisition(acquisitions[frame]), now_ns=now)
            sha = run.observer.fingerprint((p, d, fast, image, aux))
            check = check_row(old, saved, visual, tape[frame], frame=frame, public_sha=sha)
            raw = saved['decision']['original_visual_evidence']
            if raw['status'] == 'CURRENT_VISUAL_POSE':
                run.completed.check_serialized_pose(raw, p, image, aux, identity=(0, 0, 0), now_ns=now)
            count += 1
            exact += int(check['complete_original_decision_exact'])
            forecasts += int(check['original_forecast_compared'])
    report = reconstruct_report(count, exact, forecasts, saved)
    if not equal(report, result['report']): raise ValueError('whole completed report must reconstruct')
    verify(sources)
    verify_artifacts(run.OUTPUT, ids)
    verify_artifacts(run.native.OUTPUT, proof['worker_artifact_sha256'])
    if run.observer.owner_live(OWNER): raise ValueError('completed owner identity changed')
    write_json(OUTPUT, dict(status='CHAINED_ANCHOR_CONTROLLER_COMPLETION_VERIFIED', source_sha256=sources,
        result_sha256=RESULT_SHA, launch_sha256=LAUNCH_SHA, owner=OWNER, owner_ended=True,
        controller_artifact_sha256=ids, observer_verification_sha256=OBSERVER_SHA,
        worker_artifact_sha256=proof['worker_artifact_sha256'], report=report,
        public_packets_reconstructed=count, complete_saved_comparisons_reconstructed=count,
        policy_reexecuted=False, floor_registration_reexecuted=False, model_state_rechecked_in_original_replay=True,
        native_execution=False, navigation_qualified=False, goal_achieved=False))
    print('CHAINED_CONTROLLER_COMPLETION_VERIFIED', digest(OUTPUT), count, exact, forecasts, flush=True)


if __name__ == '__main__':
    main()

"""Locate the first changed selection in fixed saved development decisions.

No raw observation reconstruction, model inference or candidate outcome replay.
"""
from copy import deepcopy
import hashlib
import json
from lewm.hold_reorientation_development import HoldReorientation
from scripts import diagnose_go2_adapter_hold_prefix_v1 as diagnosis
from scripts.navigation_artifact_root_development import BASE, create_output, verify_artifacts, validate_root
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_hold_reorientation_saved_prefix_v1_attempt_001'
INPUT_SHA = '1a9b9627a9496b40aaf5e20aacad00172fbc892c827bda3e4d830df5b4843884'
SOURCES = ('scripts/check_go2_hold_reorientation_saved_prefix_v1.py',
    'lewm/hold_reorientation_controller_development.py',
    'lewm/tests/test_hold_reorientation_development.py',
    'docs/go2_hold_reorientation_v1_2026-09-10.md',
    'docs/go2_hold_reorientation_saved_prefix_v1_2026-09-10.md')


def identity(value):
    return hashlib.sha256(diagnosis.canonical(value)).hexdigest()


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive boundary check required')
    verify_artifacts(diagnosis.OUTPUT, {'result.json': INPUT_SHA})
    admitted = json.loads((diagnosis.OUTPUT/'result.json').read_text())
    if admitted['status'] != 'ADAPTER_HOLD_PREFIX_DIAGNOSIS_V1_COMPLETE':
        raise ValueError('fixed completed observed-prefix diagnosis required')
    verify_artifacts(diagnosis.OUTPUT, admitted['artifact_sha256'])
    sources = discover_sources(SOURCES, admitted['source_sha256']); verify(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('bounded CPU diagnostic resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_result_sha256=INPUT_SHA,
        input_root=str(diagnosis.OUTPUT), output_root=str(OUTPUT), hardware=resources,
        model_inference=False, native_execution=False, stop_at_first_changed_command=True))
    print('HOLD_REORIENTATION_SAVED_PREFIX_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        state = HoldReorientation(); goal = None; prefix = hashlib.sha256(); count = 0; boundary = None
        with (OUTPUT/'comparison.jsonl').open('x') as output:
            for row in read_rows(diagnosis.OUTPUT):
                frame = row['tick']; d = row['decision']; old = d['new_selection']
                if frame != count or d['terminal'] is not None:
                    raise ValueError('uninterrupted nonterminal original prefix required')
                mission = d['mission_receipt']
                current_goal = None if mission is None else mission['active_goal_initial_body_xy_m']
                if current_goal != goal: state.reset_goal(); goal = deepcopy(current_goal)
                if old is None or mission['hold_required']:
                    new = old
                else:
                    new = state.reconsider(old, frame=frame, now_ns=1_500_000_000+frame*100_000_000)
                changed = new != old
                prefix.update(diagnosis.canonical(row)); count += 1
                output.write(json.dumps(dict(frame=frame, original_row_sha256=identity(row),
                    original_selection_sha256=identity(old), candidate_selection_sha256=identity(new),
                    changed=changed, hold_count_after=state.holds), allow_nan=False)+'\n')
                if not changed: continue
                normalized = deepcopy(new); intervention = normalized.pop('hold_reorientation')
                for key in ('action', 'action_index', 'requested_command'): normalized[key] = old[key]
                if (normalized != old or old['requested_command'] != d['requested_command']
                        or new['requested_command'] == d['requested_command']):
                    raise ValueError('only declared first changed action and intervention receipt allowed')
                boundary = dict(frame=frame, original_selection=old, candidate_selection=new,
                    original_requested_command=d['requested_command'],
                    candidate_requested_command=new['requested_command'], intervention=intervention,
                    observed_goal_distance_m=d['observed_goal_distance_m'],
                    unchanged_before_boundary=True, candidate_next_observation_consumed=False,
                    original_native_command_audited=False, candidate_command_executed=False,
                    full_controller_reconstructed=False, model_inference_performed=False)
                write_json(OUTPUT/'first_boundary.json', boundary)
                break
        if boundary is None: raise ValueError('no changed command in fixed original snapshot')
        verify(sources); verify_artifacts(diagnosis.OUTPUT, admitted['artifact_sha256'])
        verify_artifacts(diagnosis.OUTPUT, {'result.json': INPUT_SHA})
        bindings = {n: digest(OUTPUT/n) for n in ('launch.json', 'comparison.jsonl', 'first_boundary.json')}
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='HOLD_REORIENTATION_SAVED_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, input_result_sha256=INPUT_SHA,
            frames=count, first_changed_command_frame=boundary['frame'],
            canonical_consumed_original_prefix_sha256=prefix.hexdigest(),
            original_requested_command=boundary['original_requested_command'],
            candidate_requested_command=boundary['candidate_requested_command'],
            unchanged_before_boundary=True, full_controller_reconstructed=False,
            raw_sensor_prefix_replay_pending=True, native_execution=False,
            model_inference_performed=False, candidate_next_observation_consumed=False,
            navigation_qualified=False, goal_achieved=False))
        print('HOLD_REORIENTATION_SAVED_PREFIX_COMPLETE', digest(OUTPUT/'result.json'),
            'first_changed_command_frame', boundary['frame'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_HOLD_REORIENTATION_SAVED_PREFIX_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()

"""Stop at the first changed saved supervised decision; no candidate rollout."""
import argparse
from copy import deepcopy
import hashlib
import json
from lewm.commitment_contact_anchored_controller_development import ordinary_commitment_contact
from scripts import run_go2_all_phase_adapter_maze02_matched_native_v1 as original
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_commitment_contact_anchored_saved_prefix_v1_attempt_001'
CASE = 'all_phase_full_supervised_rollout_residual_maze_02'
MODEL = 'seed_2026091001_full_supervised_rollout'
MODEL_SHA = '755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5'
LAUNCH_SHA = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
STREAM_SHA = 'eb5a044d40a14f3cfd442ab33eb1d4a8d6c7913555a6b61231bd3027ee50c66c'
SOURCE = 'scripts/check_go2_commitment_contact_anchored_saved_prefix_v1.py'
PROTOCOL = 'docs/go2_commitment_contact_anchored_v1_2026-09-10.md'
TEST = 'lewm/tests/test_commitment_contact_anchored_saved_prefix_development.py'
READOUT = 'docs/go2_all_phase_adapter_full_supervised_maze02_provisional_readout_2026-09-10.json'
READOUT_SHA = '162c5f43fade5297cf3f6fbbac3e5fe111cb3f3c04d068ed05db8de703589155'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def check_rows(rows, tape, append):
    count = 0; prefix = hashlib.sha256()
    for row in rows:
        frame = row['tick']; decision = row['decision']
        if (type(frame) is not int or frame != count or frame >= 3004 or decision['tick'] != frame
                or decision['terminal'] is not None
                or decision['controller'] != 'residual_anchored_continuation_controller_v1'
                or decision['model_condition'] != 'supervised_rollout'
                or decision['input_variant'] != 'full' or decision['memory_variant'] != 'persistent'
                or frame >= len(tape) or tape[frame]['tick'] != frame or tape[frame]['completed'] is not True
                or tape[frame]['pre_sample_index'] != 749+50*frame
                or tape[frame]['post_sample_index'] != 799+50*frame
                or decision['requested_command'] != tape[frame]['requested_command']):
            raise ValueError('complete ordered original supervised command prefix required')
        old = decision['new_selection']; candidate = ordinary_commitment_contact(old)
        new_command = decision['requested_command'] if candidate is None else candidate['requested_command']
        changed = new_command != decision['requested_command']
        prefix.update(canonical(row)); count += 1
        append(dict(frame=frame, original_row_sha256=identity(row),
            original_selection_sha256=identity(old), candidate_selection_sha256=identity(candidate),
            original_requested_command=decision['requested_command'], candidate_requested_command=new_command,
            changed=changed, expected_selection_only=True, model_inference_performed=False))
        if changed:
            if not old or old['action'] is None or candidate['action'] is None:
                raise ValueError('first changed ordinary feasible command required')
            return dict(frame=frame, consumed_frames=count, canonical_consumed_original_prefix_sha256=prefix.hexdigest(),
                original_selection=deepcopy(old), expected_candidate_selection=deepcopy(candidate),
                original_requested_command=decision['requested_command'], candidate_requested_command=new_command,
                unchanged_commands_before_boundary=True, candidate_next_observation_consumed=False,
                candidate_commands_executed=False, model_inference_performed=False,
                full_controller_reconstructed=False)
    raise ValueError('no changed command in complete original saved prefix')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive saved boundary check required')
    verify_artifacts(original.OUTPUT, {'launch.json': LAUNCH_SHA})
    launch = read_json(original.OUTPUT, 'launch.json')
    if launch['assigned_model_states'][MODEL] != MODEL_SHA: raise ValueError('fixed assigned supervised model required')
    verify({READOUT: READOUT_SHA}); readout = json.loads((ROOT/READOUT).read_text())
    if (readout['case'] != CASE or readout['launch_sha256'] != LAUNCH_SHA
            or readout['source_sha256'] != launch['source_sha256']
            or readout['raw_sensor_command_visibility_audit_complete'] is not False):
        raise ValueError('exact provisional original collection identity required')
    seeds = (SOURCE, PROTOCOL, TEST, READOUT,
        'lewm/tests/test_commitment_contact_anchored_development.py',
        'lewm/commitment_contact_anchored_prefix_development.py',
        'docs/go2_supervised_commitment_contact_maze01_result_2026-09-10.md')
    sources = discover_sources(seeds, launch['source_sha256']); verify(sources)
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 41*1024**3:
        raise ValueError('8GiB RAM and 41GiB artifact headroom required')
    if args.source_preflight_only:
        print('COMMITMENT_CONTACT_ANCHORED_SAVED_SOURCE_PREFLIGHT_PASS', len(sources), flush=True)
        return
    inputs = readout['artifact_sha256'] | {CASE+'/context_decisions.jsonl.gz': STREAM_SHA}
    verify_artifacts(original.OUTPUT, inputs); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, original_launch_sha256=LAUNCH_SHA,
        original_artifact_sha256=inputs, provisional_readout_sha256=READOUT_SHA, model_state_sha256=MODEL_SHA,
        hardware=resources, stop_at_first_changed_command=True, native_execution=False,
        model_inference=False, original_raw_audit_pending=True, original_full_input_admission_performed=False))
    print('COMMITMENT_CONTACT_ANCHORED_SAVED_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        tape = read_json(original.OUTPUT/CASE, 'command_tape.json')
        with (OUTPUT/'comparison.jsonl').open('x') as stream:
            def append(row):
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            boundary = check_rows(read_rows(original.OUTPUT/CASE), tape, append)
        write_json(OUTPUT/'first_boundary.json', boundary)
        verify(sources); verify_artifacts(original.OUTPUT, inputs)
        verify_artifacts(original.OUTPUT, {'launch.json': LAUNCH_SHA}); verify({READOUT: READOUT_SHA})
        artifacts = {name:digest(OUTPUT/name) for name in ('launch.json', 'comparison.jsonl', 'first_boundary.json')}
        verify_artifacts(OUTPUT, artifacts)
        write_json(OUTPUT/'result.json', dict(status='COMMITMENT_CONTACT_ANCHORED_SAVED_PREFIX_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=artifacts,
            first_changed_command_frame=boundary['frame'], consumed_frames=boundary['consumed_frames'],
            canonical_consumed_original_prefix_sha256=boundary['canonical_consumed_original_prefix_sha256'],
            original_requested_command=boundary['original_requested_command'],
            candidate_requested_command=boundary['candidate_requested_command'],
            raw_sensor_prefix_replay_pending=True, original_raw_audit_pending=True,
            full_controller_reconstructed=False, native_execution=False, model_inference_performed=False,
            candidate_next_observation_consumed=False, goal_achieved=False))
        print('COMMITMENT_CONTACT_ANCHORED_SAVED_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMMITMENT_CONTACT_ANCHORED_SAVED_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()

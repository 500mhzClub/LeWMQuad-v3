"""Independent recorded decisions, model identity and paired timing verification.

The original replay owns inference, public-array preservation and hidden-state
execution. This checker reconstructs expected decision hashes from original
records and authenticates the reported state checks against the prior replay.
"""
import argparse
from itertools import islice
import json
import math
from pathlib import Path

from scripts import replay_go2_reused_floor_mesh_prefix_v1 as replay
from scripts.navigation_artifact_root_development import verify_artifacts, artifact_path
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows, NAME

SOURCE = 'scripts/verify_go2_reused_floor_mesh_prefix_v1.py'
TEST = 'lewm/tests/test_reused_floor_mesh_verification_development.py'
PROTOCOL = 'docs/go2_reused_floor_mesh_verification_v1_2026-09-10.md'
OUTPUT = ROOT/'docs/go2_reused_floor_mesh_prefix_verification_2026-09-10.json'
PREPARATION = Path('docs/go2_reused_floor_mesh_execution_preparation_2026-09-10.json')
PREPARATION_SHA = 'd3cbb8a236ce5c136bf97a823b73302c4c9e9f67593eaa9ea74768a285869979'


def prepared_sources():
    if digest(PREPARATION) != PREPARATION_SHA:
        raise ValueError('exact original replay source preparation required')
    original = json.loads(PREPARATION.read_text())['prepared_source_sha256']
    verify(original)
    sources = discover_sources((SOURCE, TEST, PROTOCOL), original)
    verify(sources)
    return original, sources


def require_report(result, launch, reference):
    if (result['status'] != 'REUSED_FLOOR_MESH_PREFIX_V1_COMPLETE'
            or result['native_execution'] is not False or result['goal_achieved'] is not False
            or result['isolated_incremental_mesh_speedup_claimed'] is not False
            or result['optimized_profile_result_sha256'] != replay.PROFILE_RESULT_SHA
            or result['source_sha256'] != launch['source_sha256']
            or type(result['wall_s']) not in (int,float)
            or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0):
        raise ValueError('complete combined-optimization replay and scope required')
    expected_launch = dict(optimized_profile_result_sha256=replay.PROFILE_RESULT_SHA,
        optimized_profile_launch_sha256=replay.PROFILE_LAUNCH_SHA,
        paired_frozen_footprint_result_sha256=replay.optimized_profile.COMPLETED_SHA,
        frames=405, state_frames=[3,12,395,404], model_state_sha256=replay.profile.reference.MODEL_SHA,
        native_execution=False, model_training=False, profiling_enabled=False,
        imported_module_globals_mutated=False, invocation_frozen_footprint_receipts=True,
        observation_local_floor_mesh_reuse=True, normalized_state_type_paths=[],
        comparison='original controller versus combined frozen-footprint and mesh-reuse controller',
        isolated_incremental_mesh_speedup_claimed=False)
    if any(type(launch.get(k)) is not type(v) or launch[k] != v for k,v in expected_launch.items()):
        raise ValueError('exact prospective launch and comparison required')
    expected_report = dict(frames=405, raw_model_forecast_comparisons=402,
        model_state_sha256=replay.profile.reference.MODEL_SHA, model_state_unchanged=True,
        complete_original_decisions_reconstructed=True, complete_normalized_candidate_decisions_exact=True,
        public_input_arrays_unchanged=True, alternating_execution_order=True, profiling_enabled=False,
        controller_observe_only_timed=True, sensor_acquisition_timed=False, isolated_benchmark=False,
        no_observation_405_consumed=True, native_execution=False, real_time_qualified=False,
        navigation_qualified=False)
    report = result['report']
    if any(type(report.get(k)) is not type(v) or report[k] != v for k,v in expected_report.items()):
        raise ValueError('complete original paired replay report required')
    states = report['observed_state_checks']
    if (states != reference['report']['observed_state_checks']
            or [s['frame'] for s in states] != list(replay.STATE_FRAMES)
            or any(s['complete_retained_observed_state_equal'] is not True for s in states)):
        raise ValueError('all four reported retained-state hashes must match completed frozen-footprint replay')


def check_rows(rows, saved, profile_rows, frozen_rows, tape, expected_timings):
    if any(len(p) != 405 for p in (rows,profile_rows,frozen_rows)) or len(tape) < 405:
        raise ValueError('complete fixed recorded populations and command tape required')
    forecasts = count = 0
    for frame,(row,original,profiled,frozen) in enumerate(zip(rows,saved,profile_rows,frozen_rows,strict=True)):
        decision = original['decision']; command = tape[frame]
        if (row['frame'] != frame or original['tick'] != frame or profiled['frame'] != frame
                or frozen['frame'] != frame or command['tick'] != frame
                or command['completed'] is not True or command['pre_sample_index'] != 749+50*frame
                or command['post_sample_index'] != 799+50*frame
                or command['requested_command'] != decision['requested_command']
                or decision['terminal'] is not None):
            raise ValueError('original completed command endpoints and ordered decisions required')
        identity = replay.profile.reference.saved.identity
        original_sha = identity(decision)
        expected = decision | {'controller':replay.CONTROLLER, replay.FLAG:True, replay.frozen.FLAG:True}
        if (row['original_decision_sha256'] != original_sha
                or profiled['original_decision_sha256'] != original_sha
                or frozen['original_decision_sha256'] != original_sha
                or row['candidate_decision_sha256'] != identity(expected)
                or row['public_input_sha256'] != profiled['public_input_sha256']
                or row['public_input_sha256'] != frozen['public_input_sha256']
                or row['complete_original_decision_reconstructed'] is not True
                or row['candidate_normalized_decision_exact'] is not True
                or row['public_input_arrays_unchanged'] is not True):
            raise ValueError('exact complete original/candidate decisions and public input hashes required')
        selection = decision['new_selection']
        forecasts += int(bool(selection and 'prediction' in selection)); count += 1
    timings = replay.preceding.timing_summary(rows)
    if count != 405 or forecasts != 402 or timings != expected_timings:
        raise ValueError('all forecasts and independently recomputed timing windows required')
    return dict(original_saved_decisions_reconstructed=count,
        candidate_expected_decision_hashes_checked=count, forecast_count=forecasts,
        original_command_endpoints_checked=count, public_input_hashes_match_completed_references=True,
        timing_windows_recomputed=True, timing_windows=timings)


def rows_at(root):
    raw = artifact_path(root,'comparison.jsonl').read_bytes()
    if not raw.endswith(b'\n'):
        raise ValueError('complete newline-terminated comparison stream required')
    return [json.loads(line) for line in raw.splitlines()]


def verified_result(result_sha, launch_sha, original_sources):
    root = replay.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('failed replay remains a failure')
    verify_artifacts(root, {'result.json':result_sha,'launch.json':launch_sha})
    result = read_json(root,'result.json'); launch = read_json(root,'launch.json')
    if (result['source_sha256'] != original_sources
            or set(result['artifact_sha256']) != {'launch.json','comparison.jsonl'}
            or result['artifact_sha256']['launch.json'] != launch_sha):
        raise ValueError('exact prepared sources and two completed output artifacts required')
    verify(original_sources); verify_artifacts(root,result['artifact_sha256'])
    profile_launch = replay.prepared_inputs()
    replay.completed_profile(replay.PROFILE_RESULT_SHA,profile_launch)
    reference,_ = replay.optimized_profile.completed_inputs()
    replay.preceding.profile_inputs()
    require_report(result,launch,reference)
    original_root = replay.profile.reference.original.OUTPUT
    case = replay.profile.reference.CASE[0]
    names = [case+'/'+n for n in (NAME,'command_tape.json')]
    original_ids = {n:launch['input_admission']['original_artifact_sha256'][n] for n in names}
    verify_artifacts(original_root,original_ids)
    stream = read_rows(original_root/case)
    try:
        checked = check_rows(rows_at(root),islice(stream,405),rows_at(replay.profile.OUTPUT),
            rows_at(replay.frozen.OUTPUT),read_json(original_root/case,'command_tape.json'),
            result['report']['timing_windows'])
    finally:
        stream.close()
    verify(original_sources); verify_artifacts(original_root,original_ids)
    verify_artifacts(root,result['artifact_sha256'] | {'result.json':result_sha})
    return dict(checked,result_sha256=result_sha,launch_sha256=launch_sha,
        artifact_sha256=result['artifact_sha256'],source_count=len(original_sources),
        original_decision_stream_and_command_tape_sha256=original_ids,
        all_four_reported_retained_state_hashes_match_completed_reference=True,
        model_inference_rerun=False,hidden_state_reconstructed=False,raw_sensor_packets_reloaded=False,
        isolated_incremental_mesh_speedup_claimed=False,native_execution=False,
        scope='Independent recorded decisions, hashes and timings; original replay owns neural and hidden-state execution.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-preflight-only',action='store_true')
    parser.add_argument('--result-sha256'); parser.add_argument('--launch-sha256')
    args = parser.parse_args()
    if not args.source_preflight_only and (args.result_sha256 is None or args.launch_sha256 is None):
        parser.error('exact result and launch SHA-256 values required')
    original,sources = prepared_sources()
    if args.source_preflight_only:
        print('REUSED_FLOOR_MESH_VERIFICATION_SOURCE_PREFLIGHT_PASS',len(sources),flush=True)
        return
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive independent result verification document required')
    checked = verified_result(args.result_sha256,args.launch_sha256,original)
    verify(sources); write_json(OUTPUT,checked | {'verification_source_sha256':sources})
    print('REUSED_FLOOR_MESH_RECORDED_EVIDENCE_VERIFIED',digest(OUTPUT),flush=True)


if __name__ == '__main__':
    main()

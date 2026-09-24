"""Independent recorded-evidence check; neural/state execution belongs to replay."""
import argparse
from itertools import islice
import json
import math
from scripts import replay_go2_frozen_footprint_anchored_prefix_v1 as replay
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows, NAME

SOURCE = 'scripts/verify_go2_frozen_footprint_anchored_prefix_v1.py'
TEST = 'lewm/tests/test_frozen_footprint_anchored_verification_development.py'
PROTOCOL = 'docs/go2_frozen_footprint_anchored_verification_v1_2026-09-10.md'
OUTPUT = ROOT/'docs/go2_frozen_footprint_anchored_prefix_verification_2026-09-10.json'


def require_report(result, launch, reference):
    if (result['status'] != 'FROZEN_FOOTPRINT_ANCHORED_PREFIX_V1_COMPLETE'
            or result['native_execution'] is not False or result['goal_achieved'] is not False
            or result['preceding_result_sha256'] != replay.PRECEDING_SHA
            or result['source_sha256'] != launch['source_sha256']
            or type(result['wall_s']) not in (float, int)
            or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0):
        raise ValueError('complete exact non-native paired replay required')
    expected_launch = dict(preceding_result_sha256=replay.PRECEDING_SHA,
        profile_result_sha256=replay.preceding.PROFILE_SHA, frames=405, state_frames=[3, 12, 395, 404],
        model_state_sha256=replay.profile.reference.MODEL_SHA, native_execution=False,
        model_training=False, profiling_enabled=False, imported_module_globals_mutated=False,
        receipt_copy_optimization_enabled=False, batched_patch_queries_enabled=False,
        invocation_local_surface_receipt_sharing=False, invocation_frozen_footprint_receipts=True, normalized_state_type_paths=[])
    if any(type(launch.get(k)) is not type(v) or launch[k] != v for k, v in expected_launch.items()):
        raise ValueError('exact frozen-footprint-only prospective launch required')
    expected = dict(frames=405, raw_model_forecast_comparisons=402,
        model_state_sha256=replay.profile.reference.MODEL_SHA, model_state_unchanged=True,
        complete_original_decisions_reconstructed=True, complete_normalized_candidate_decisions_exact=True,
        public_input_arrays_unchanged=True, alternating_execution_order=True, profiling_enabled=False,
        controller_observe_only_timed=True, sensor_acquisition_timed=False, isolated_benchmark=False,
        no_observation_405_consumed=True, native_execution=False, real_time_qualified=False,
        navigation_qualified=False)
    report = result['report']
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError('complete original replay report and scope required')
    states = report['observed_state_checks']
    if (states != reference['report']['observed_state_checks']
            or [r['frame'] for r in states] != list(replay.STATE_FRAMES)
            or any(r['complete_retained_observed_state_equal'] is not True for r in states)):
        raise ValueError('all four reported complete-state hashes must match the completed reference')


def check_rows(rows, saved, profile_rows, copy_rows, tape, expected_timings):
    if any(len(population) != 405 for population in (rows, saved, profile_rows, copy_rows)) or len(tape) < 405:
        raise ValueError('four complete fixed prefix populations and original tape required')
    forecasts = 0
    for frame, (row, original, profiled, copied) in enumerate(zip(rows, saved, profile_rows, copy_rows, strict=True)):
        decision = original['decision']; command = tape[frame]
        if (row['frame'] != frame or original['tick'] != frame or profiled['frame'] != frame
                or copied['frame'] != frame or command['tick'] != frame
                or command['completed'] is not True or command['pre_sample_index'] != 749+50*frame
                or command['post_sample_index'] != 799+50*frame
                or command['requested_command'] != decision['requested_command']
                or decision['terminal'] is not None):
            raise ValueError('ordered original observations and completed command endpoints required')
        expected = decision | {'controller': replay.CONTROLLER, replay.FLAG: True}
        original_sha = replay.profile.reference.saved.identity(decision)
        if (row['original_decision_sha256'] != original_sha
                or profiled['original_decision_sha256'] != original_sha
                or copied['original_decision_sha256'] != original_sha
                or row['candidate_decision_sha256'] != replay.profile.reference.saved.identity(expected)
                or row['public_input_sha256'] != profiled['public_input_sha256']
                or row['public_input_sha256'] != copied['public_input_sha256']
                or row['complete_original_decision_reconstructed'] is not True
                or row['candidate_normalized_decision_exact'] is not True
                or row['public_input_arrays_unchanged'] is not True):
            raise ValueError('complete saved decisions, metadata-only candidate and public input hashes required')
        selection = decision['new_selection']
        forecasts += int(bool(selection and 'prediction' in selection))
    timings = replay.preceding.timing_summary(rows)
    if forecasts != 402 or timings != expected_timings:
        raise ValueError('all original forecasts and independently recomputed timing windows required')
    return dict(original_saved_decisions_reconstructed=405, candidate_expected_decision_hashes_checked=405,
        forecast_count=forecasts, original_command_endpoints_checked=405,
        public_input_hashes_match_completed_profile_and_copy_replay=True,
        timing_windows_recomputed=True, timing_windows=timings)


def verified_result(result_sha, launch_sha):
    root = replay.OUTPUT
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('failed original replay must remain a failure')
    verify_artifacts(root, {'result.json': result_sha, 'launch.json': launch_sha})
    result = read_json(root, 'result.json'); launch = read_json(root, 'launch.json')
    if result['artifact_sha256'].get('launch.json') != launch_sha or set(result['artifact_sha256']) != {'launch.json', 'comparison.jsonl'}:
        raise ValueError('exact two completed replay outputs required')
    verify_artifacts(root, result['artifact_sha256']); verify(result['source_sha256'])
    replay.preceding_inputs()
    reference = read_json(replay.preceding.OUTPUT, 'result.json')
    require_report(result, launch, reference)
    def rows_at(directory):
        return [json.loads(line) for line in (directory/'comparison.jsonl').read_text().splitlines()]
    original_root = replay.profile.reference.original.OUTPUT/replay.profile.reference.CASE[0]
    original_bindings = launch['input_admission']['original_artifact_sha256']
    names = [replay.profile.reference.CASE[0]+'/'+n for n in (NAME, 'command_tape.json')]
    original_ids = {name: original_bindings[name] for name in names}
    verify_artifacts(replay.profile.reference.original.OUTPUT, original_ids)
    saved = list(islice(read_rows(original_root), replay.FRAMES))
    checked = check_rows(rows_at(root), saved, rows_at(replay.profile.OUTPUT), rows_at(replay.preceding.OUTPUT),
        read_json(original_root, 'command_tape.json'), result['report']['timing_windows'])
    verify_artifacts(root, result['artifact_sha256'] | {'result.json': result_sha})
    verify_artifacts(replay.profile.reference.original.OUTPUT, original_ids)
    return result, dict(checked, result_sha256=result_sha, launch_sha256=launch_sha,
        artifact_sha256=result['artifact_sha256'], source_count=len(result['source_sha256']),
        preceding_result_sha256=replay.PRECEDING_SHA,
        all_four_reported_retained_state_hashes_match_completed_copy_replay=True,
        original_decision_stream_and_command_tape_sha256=original_ids,
        model_inference_rerun=False, hidden_state_reconstructed_in_this_check=False,
        raw_sensor_packets_reloaded_in_this_check=False, native_execution=False,
        report_scope='independent recorded decision/hash/timing verification; original replay owns neural and hidden-state execution')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--launch-sha256', required=True)
    args = parser.parse_args(); launch_sha = args.launch_sha256
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive completed-result verification record required')
    verify_artifacts(replay.OUTPUT, {'result.json': args.result_sha256, 'launch.json': launch_sha})
    original = read_json(replay.OUTPUT, 'result.json')
    sources = discover_sources((SOURCE, TEST, PROTOCOL), original['source_sha256']); verify(sources)
    result, checked = verified_result(args.result_sha256, launch_sha)
    if result['source_sha256'] != original['source_sha256']:
        raise ValueError('original result source binding changed during verification')
    verify(sources)
    write_json(OUTPUT, dict(checked, verification_source_sha256=sources))
    print('FROZEN_FOOTPRINT_ANCHORED_RECORDED_EVIDENCE_VERIFIED', digest(OUTPUT), flush=True)


if __name__ == '__main__': main()

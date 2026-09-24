"""Verify the completed fixed profile and reconstruct its saved timing summaries."""
import argparse
import re
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pstats

from scripts import profile_go2_receipt_copied_footprint_late_history_v1 as run
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/verify_go2_receipt_copied_profile_completion_v1.py'
EXECUTION = 'docs/go2_receipt_copied_footprint_profile_execution_2026-09-11.json'
EXECUTION_SHA = 'a02bc5b851edf5e4e63a4d72274b2de57c011f491e620e8df54f69d5f924a35e'
TEST = 'lewm/tests/test_receipt_copied_profile_completion_development.py'
OUTPUT = ROOT/'docs/go2_receipt_copied_profile_completion_verification_2026-09-11.json'


def reconstruct_report(rows, report, summaries):
    if (len(rows) != 1428 or any(type(row['frame']) is not int or row['frame'] != i
            for i,row in enumerate(rows)) or set(summaries) != set(run.original.WINDOWS)):
        raise ValueError('complete ordered history and all three profiler summaries required')
    expected = dict(frames=1428, raw_model_forecast_comparisons=1425,
        model_state_sha256=run.original.reference.MODEL_SHA, last_replayed_observation=1427,
        controller='ReceiptCopiedFootprintController')
    for key in ('retained_state_identity_established', 'sensor_acquisition_profiled',
                'profiler_overhead_removed', 'isolated_benchmark', 'speedup_established',
                'native_execution', 'policy_changed', 'real_time_qualified',
                'navigation_qualified', 'imported_module_globals_mutated'):
        expected[key] = False
    for key in ('complete_original_decisions_reconstructed', 'model_state_unchanged',
                'controller_observe_only_profiled', 'no_observation_1428_consumed',
                'invocation_frozen_footprint_receipts', 'normalization_outside_profiled_region',
                'complete_normalized_candidate_decisions_exact',
                'receipt_copied_footprint_controller_profiled',
                'all_profiled_decisions_equal_completed_copied_replay'):
        expected[key] = True
    actual = {k:v for k,v in report.items() if k not in ('windows', 'state_size_snapshots')}
    if fingerprint(actual) != fingerprint(expected):
        raise ValueError('complete fixed report scope required')
    if set(report['windows']) != set(run.original.WINDOWS):
        raise ValueError('all three fixed windows required')
    for row in rows:
        window = next((n for n, (a,b) in run.original.WINDOWS.items() if a <= row['frame'] <= b), None)
        elapsed = row['controller_wall_s']
        if (row['profiled_window'] != window or type(elapsed) not in (float, int)
                or not math.isfinite(elapsed) or elapsed <= 0):
            raise ValueError('finite positive timing and original window membership required')
    windows = {}
    for name, (first,last) in run.original.WINDOWS.items():
        summary = summaries[name]
        window = report['windows'][name]
        observations = window['observations']
        if [r['frame'] for r in observations] != list(range(first,last+1)):
            raise ValueError('exact ten-frame population required')
        for observation in observations:
            if observation['controller_wall_s_with_profiling'] != rows[observation['frame']]['controller_wall_s']:
                raise ValueError('window timing must match complete comparison stream')
            if name == 'repeated_hold' and observation['action'] != 'hold':
                raise ValueError('original repeated hold actions required')
        rebuilt = dict(observations=observations,
            total_exclusive_profiled_s=summary['total_exclusive_profiled_s'],
            top_functions_by_cumulative_time=summary['functions'][:15],
            top_modules_by_exclusive_time=summary['modules_by_exclusive_time'][:15])
        if fingerprint(rebuilt) != fingerprint(window):
            raise ValueError('complete report timing summaries must reconstruct')
        windows[name] = rebuilt
    return windows


def verify_completed(result_sha):
    if type(result_sha) is not str or re.fullmatch('[0-9a-f]{64}',result_sha) is None:
        raise ValueError('explicit completed profile result SHA-256 required')
    verify({EXECUTION: EXECUTION_SHA})
    execution = json.loads((ROOT/EXECUTION).read_text())
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != execution['boot_id']:
        raise ValueError('original execution boot required')
    if owner_live(execution['owner']):
        raise ValueError('original profile owner still live')
    if (run.OUTPUT/'failure.json').exists() or (run.OUTPUT/'failure.json').is_symlink():
        raise ValueError('original failure must be preserved, not accepted as completion')
    sources = discover_sources((SOURCE, TEST, EXECUTION), run.prepared_sources())
    verify(sources)
    fixed = {'result.json': result_sha, 'launch.json': execution['launch_sha256']}
    verify_artifacts(run.OUTPUT, fixed)
    result = read_json(run.OUTPUT, 'result.json')
    launch = read_json(run.OUTPUT, 'launch.json')
    if (result['status'] != 'RECEIPT_COPIED_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE'
            or result['source_sha256'] != execution['source_sha256']
            or launch['source_sha256'] != execution['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())
            or result['native_execution'] is not False or result['goal_achieved'] is not False
            or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0):
        raise ValueError('original completed profile and unchanged source/scope required')
    names = {'launch.json', 'comparison.jsonl'} | {
        n+s for n in run.original.WINDOWS for s in ('.json', '.prof')}
    if set(result['artifact_sha256']) != names:
        raise ValueError('exact eight-output population required')
    verify_artifacts(run.OUTPUT, result['artifact_sha256'])
    preceding = run.admit_completed(sources)
    admission = run.profile.bound_profile_inputs(preceding[-1], sources)
    if (admission != launch['input_admission'] or admission != preceding[2]['input_admission']
            or result['sensing_scope'] != admission['sensing_scope']):
        raise ValueError('unchanged actual raw/model inputs and failed sensing scope required')
    rows = [json.loads(line) for line in (run.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    run.compare_profile_rows(rows, preceding[3])
    summaries = {}
    for name in run.original.WINDOWS:
        summary = run.original.profile_summary(pstats.Stats(str(run.OUTPUT/(name+'.prof'))).stats)
        if fingerprint(summary) != fingerprint(read_json(run.OUTPUT, name+'.json')):
            raise ValueError('saved profiler data must exactly reconstruct full JSON summary')
        summaries[name] = summary
    windows = reconstruct_report(rows, result['report'], summaries)
    verify(sources)
    verify_artifacts(run.OUTPUT, result['artifact_sha256'] | fixed)
    if owner_live(execution['owner']):
        raise ValueError('original owner unexpectedly live')
    return dict(status='RECEIPT_COPIED_PROFILE_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        source_count=len(sources), original_source_count=len(result['source_sha256']),
        result_sha256=result_sha, artifact_sha256=result['artifact_sha256'],
        original_owner=execution['owner'], original_owner_ended=True,
        complete_rows=1428, raw_model_forecasts=1425,
        completed_receipt_copy_reference_reauthenticated=True, actual_raw_and_model_bindings_rehashed=True,
        all_three_pstats_summaries_exact=True, windows=windows, sensing_scope=result['sensing_scope'],
        state_size_snapshots_independently_reconstructed=False,
        raw_sensor_or_model_replay_reexecuted=False, full_training_ancestry_reexecuted=False,
        profiler_overhead_removed=False, isolated_benchmark=False,
        cumulative_times_must_not_be_summed=True, native_execution=False,
        real_time_qualified=False, navigation_qualified=False, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args(); result = verify_completed(args.result_sha256)
    write_json(OUTPUT, result)
    print('RECEIPT_COPIED_PROFILE_COMPLETION_VERIFIED', digest(OUTPUT), result['source_count'], flush=True)


if __name__ == '__main__':
    main()

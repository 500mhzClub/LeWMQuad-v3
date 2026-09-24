"""Verify every paired timing row, predecessor identity and complete final report."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from scripts import replay_go2_tiled_density_progressive_floor_late_history_v1 as run
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE = 'scripts/verify_go2_tiled_density_progressive_floor_controller_completion_v1.py'
TEST = 'lewm/tests/test_tiled_density_progressive_floor_controller_completion_development.py'
EXECUTION = 'docs/go2_tiled_density_progressive_floor_controller_replay_execution_2026-09-11.json'
EXECUTION_SHA = '0d449415fdd73ce8160e73a243ec285612dcf4dc1d3be37e240480f95a7bb762'
OUTPUT = ROOT/'docs/go2_tiled_density_progressive_floor_controller_completion_verification_2026-09-11.json'


def check_rows(rows, prior_rows):
    if len(rows) != len(prior_rows) or len(rows) != 1428:
        raise ValueError('complete paired and prior 1428-frame histories required')
    for i, (row, prior) in enumerate(zip(rows, prior_rows, strict=True)):
        if (type(row['frame']) is not int or row['frame'] != i
                or row['public_input_sha256'] != prior['public_input_sha256']
                or row['original_decision_sha256'] != prior['original_decision_sha256']
                or row['baseline_decision_sha256'] != prior['candidate_decision_sha256']
                or type(row['candidate_decision_sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}', row['candidate_decision_sha256']) is None
                or any(row[k] is not True for k in ('complete_original_decision_reconstructed',
                    'candidate_normalized_decision_exact', 'public_input_arrays_unchanged'))):
            raise ValueError('every original input, prior candidate and present baseline must match')
    return run.original.timing_summary(rows)


def expected_report(prior_report, timing):
    expected = deepcopy(prior_report)
    for key in ('incremental_progressive_patch_batching_comparison', 'both_controllers_use_density_routed_floor_registration_and_mapping'):
        if expected.pop(key) is not True: raise ValueError('original progressive comparison scope required')
    expected.update(baseline='ProgressiveBatchedFloorController', candidate='TiledDensityProgressiveFloorController',
        timing_windows=timing, normalized_state_type_paths=run.STATE_TYPE_PATHS,
        incremental_reuse_comparison=False, incremental_tiled_dense_floor_comparison=True,
        persistent_memory_type_unchanged=True, both_controllers_use_progressive_retained_floor_patch_batching=True,
        imported_module_globals_mutated=False)
    return expected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result-sha256',required=True)
    args=parser.parse_args();result_sha=args.result_sha256
    if re.fullmatch('[0-9a-f]{64}',result_sha) is None:raise ValueError('exact completed result SHA256 required')
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive timing completion verification required')
    verify({EXECUTION: EXECUTION_SHA})
    execution = json.loads((ROOT/EXECUTION).read_text())
    if (Path('/proc/sys/kernel/random/boot_id').read_text().strip() != execution['boot_id']
            or run.owner_live(execution['owner'])):
        raise ValueError('original replay owner must be ended on its recorded boot')
    if (run.OUTPUT/'failure.json').exists() or (run.OUTPUT/'failure.json').is_symlink():
        raise ValueError('original failed replay must not be accepted as complete')
    sources = discover_sources((SOURCE, TEST, EXECUTION), run.prepared_sources())
    verify(sources)
    fixed = {'result.json': result_sha, 'launch.json': execution['launch_sha256']}
    verify_artifacts(run.OUTPUT, fixed)
    result = read_json(run.OUTPUT, 'result.json')
    launch = read_json(run.OUTPUT, 'launch.json')
    if (result['status'] != 'TILED_DENSITY_PROGRESSIVE_FLOOR_LATE_HISTORY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or result['source_sha256'] != execution['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())
            or set(result['artifact_sha256']) != {'launch.json', 'comparison.jsonl'}
            or result['artifact_sha256']['launch.json'] != fixed['launch.json']
            or result['native_execution'] is not False or result['goal_achieved'] is not False
            or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0):
        raise ValueError('exact completed paired replay, source and diagnostic scope required')
    verify_artifacts(run.OUTPUT, result['artifact_sha256'])
    _, prior, prior_launch, prior_rows, raw_launch = run.admit_completed(sources)
    admission = run.profile.profile.bound_profile_inputs(raw_launch, sources)
    if admission != launch['input_admission'] or admission != prior_launch['input_admission']:
        raise ValueError('same actual original raw sensor and model bindings required')
    expected_launch = dict(previous_result_sha256=run.PREVIOUS_RESULT_SHA,
        previous_launch_sha256=run.PREVIOUS_LAUNCH_SHA,
        tiled_density_recorded_probe_sha256=run.MICROBENCHMARK_SHA,frames=1428,
        state_frames=list(run.original.STATE_FRAMES), normalized_state_type_paths=run.STATE_TYPE_PATHS,
        baseline='ProgressiveBatchedFloorController', candidate='TiledDensityProgressiveFloorController',
        native_execution=False, model_training=False,
        environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled'))
    if fingerprint({k: launch[k] for k in expected_launch}) != fingerprint(expected_launch):
        raise ValueError('exact paired controllers, windows, state paths and environment required')
    rows = [json.loads(line) for line in (run.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    timing = check_rows(rows, prior_rows)
    expected = expected_report(prior['report'], timing)
    if fingerprint(result['report']) != fingerprint(expected) or result['sensing_scope'] != prior['sensing_scope']:
        raise ValueError('complete report, seven state witnesses, all timings and negative sensing scope must reconstruct')
    verify(sources)
    verify_artifacts(run.OUTPUT, result['artifact_sha256'] | fixed)
    if run.owner_live(execution['owner']): raise ValueError('original replay owner unexpectedly live')
    all_times = timing['all_navigation']
    reduction = 100*(1-all_times['candidate_total_s']/all_times['baseline_total_s'])
    write_json(OUTPUT, dict(status='TILED_DENSITY_PROGRESSIVE_FLOOR_COMPLETION_VERIFIED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        original_source_count=len(result['source_sha256']), result_sha256=result_sha,
        artifact_sha256=result['artifact_sha256'], original_owner=execution['owner'], original_owner_ended=True,
        rows=1428, raw_model_forecasts=1425, observed_state_checks=result['report']['observed_state_checks'],
        timing_windows=timing, total_navigation_time_reduction_percent=reduction,
        preceding_progressive_completion_reauthenticated=True, actual_raw_and_model_bindings_rehashed=True,
        complete_report_and_timing_population_reconstructed=True, sensing_scope=result['sensing_scope'],
        raw_sensor_model_replay_reexecuted=False, full_training_ancestry_reexecuted=False,
        native_execution=False, real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
    print('TILED_DENSITY_PROGRESSIVE_FLOOR_COMPLETION_VERIFIED', digest(OUTPUT), len(sources), reduction, flush=True)


if __name__ == '__main__':
    main()

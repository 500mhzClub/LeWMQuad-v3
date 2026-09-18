"""Authenticate the completed packed replay and reconstruct all paired timings."""
from copy import deepcopy
import math
import re

from scripts import replay_go2_packed_fused_scoped_late_history_v1 as run
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live

LAUNCH_SHA = 'c23a894e770903e05cbe55e91e82b21c30dda29326309ebd3fcb0596cb1c6b88'
FUSED_SHA = '021b52bbc269bfd4f92a493f7adf90f9b8c714b8c4dac673e9fd1f3ccb97a2ba'
OWNER = dict(pid=2786620, created=1789093441.1, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
    'scripts/replay_go2_packed_fused_scoped_late_history_v1.py', '--fused-result-sha256', FUSED_SHA])


def require_result(result, launch, rows, preceding, prior_rows, sources):
    if (result['status'] != 'PACKED_FUSED_SCOPED_LATE_HISTORY_REPLAY_V1_COMPLETE'
            or result['source_sha256'] != launch['source_sha256']
            or any(sources.get(n) != h for n, h in result['source_sha256'].items())
            or set(result['artifact_sha256']) != {'launch.json', 'comparison.jsonl'}
            or result['artifact_sha256']['launch.json'] != LAUNCH_SHA
            or result['native_execution'] is not False or result['goal_achieved'] is not False
            or type(result['wall_s']) not in (float, int) or not math.isfinite(result['wall_s']) or result['wall_s'] <= 0
            or result['sensing_scope'] != preceding['sensing_scope']):
        raise ValueError('complete original packed replay and unchanged scientific scope required')
    expected_launch = dict(fused_result_sha256=FUSED_SHA, fused_launch_sha256=run.PREVIOUS_LAUNCH,
        frames=1428, state_frames=list(run.previous.profile.paired.previous.STATE_FRAMES),
        normalized_state_type_paths=run.STATE_TYPE_PATHS, baseline='FusedScopedBatchedController',
        candidate='PackedFusedScopedController', native_execution=False, model_training=False,
        environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled'))
    if any(fingerprint(launch[k]) != fingerprint(v) for k, v in expected_launch.items()):
        raise ValueError('exact paired implementation, environment and state normalization required')
    if len(rows) != 1428 or len(prior_rows) != 1428:
        raise ValueError('both complete ordered observation populations required')
    for i, (row, prior) in enumerate(zip(rows, prior_rows, strict=True)):
        if (type(row['frame']) is not int or row['frame'] != i
                or row['public_input_sha256'] != prior['public_input_sha256']
                or row['original_decision_sha256'] != prior['original_decision_sha256']
                or row['baseline_decision_sha256'] != prior['candidate_decision_sha256']
                or type(row['candidate_decision_sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}', row['candidate_decision_sha256']) is None):
            raise ValueError('every original input, fused baseline and packed decision must be bound')
        for key in ('complete_original_decision_reconstructed', 'candidate_normalized_decision_exact', 'public_input_arrays_unchanged'):
            if row[key] is not True: raise ValueError('every complete paired comparison must pass: '+key)
    timing = run.previous.profile.paired.previous.timing_summary(rows)
    expected = deepcopy(preceding['report'])
    for key in ('incremental_receipt_construction_comparison', 'both_controllers_use_scoped_reuse_and_batched_patches'):
        expected.pop(key)
    expected.update(baseline='FusedScopedBatchedController', candidate='PackedFusedScopedController',
        normalized_state_type_paths=run.STATE_TYPE_PATHS, incremental_packed_insertion_comparison=True,
        both_controllers_use_fused_scoped_batched_receipts=True, timing_windows=timing)
    if fingerprint(result['report']) != fingerprint(expected):
        raise ValueError('complete report, seven states, model, flags and timing populations must reconstruct')
    return timing


def completed(result_sha, sources):
    if owner_live(OWNER): raise ValueError('original packed replay owner is still live')
    if type(result_sha) is not str or re.fullmatch('[0-9a-f]{64}', result_sha) is None:
        raise ValueError('exact original completed packed result SHA-256 required')
    if (run.OUTPUT/'failure.json').exists() or (run.OUTPUT/'failure.json').is_symlink():
        raise ValueError('original packed failure must not be replaced by a completion')
    verify(sources); verify_artifacts(run.OUTPUT, {'result.json':result_sha, 'launch.json':LAUNCH_SHA})
    result = read_json(run.OUTPUT, 'result.json'); launch = read_json(run.OUTPUT, 'launch.json')
    verify(result['source_sha256']); verify_artifacts(run.OUTPUT, result['artifact_sha256'])
    preceding, fused_launch, prior_rows, _ = run.completed_previous(FUSED_SHA, sources)
    if launch['input_admission'] != fused_launch['input_admission']:
        raise ValueError('unchanged original raw/model admission required')
    rows = [run.json.loads(line) for line in (run.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    timing = require_result(result, launch, rows, preceding, prior_rows, sources)
    verify(sources); verify_artifacts(run.OUTPUT, result['artifact_sha256'] | {'result.json':result_sha})
    if owner_live(OWNER): raise ValueError('original packed owner identity unexpectedly live')
    return dict(result_sha256=result_sha, launch_sha256=LAUNCH_SHA, source_count=len(result['source_sha256']),
        rows=len(rows), raw_model_forecasts=result['report']['raw_model_forecast_comparisons'],
        observed_state_checks=result['report']['observed_state_checks'], timing_windows=timing,
        original_fused_completion_reauthenticated=True, all_source_and_output_bindings_verified=True,
        complete_rows_and_timings_reconstructed=True, original_owner_ended=True,
        full_raw_sensor_model_replay_reexecuted=False, training_ancestry_reexecuted=False,
        original_failed_visibility_and_round_trip_retained=True,
        native_execution=False, real_time_qualified=False, navigation_qualified=False)

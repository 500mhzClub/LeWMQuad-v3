"""Synthetic tamper tests for packed/fused result and full timing evidence."""
from copy import deepcopy

import pytest

from scripts import verify_go2_packed_fused_scoped_completion_v1 as check


def fixture():
    sources = {'synthetic.py':'a'*64}; rows = []; prior = []
    for i in range(1428):
        r = dict(frame=i, execution_order=[0, 1] if i % 2 == 0 else [1, 0],
            baseline_controller_s=.2, candidate_controller_s=.4,
            public_input_sha256='b'*64, original_decision_sha256='c'*64,
            baseline_decision_sha256='d'*64, candidate_decision_sha256='e'*64,
            complete_original_decision_reconstructed=True, candidate_normalized_decision_exact=True,
            public_input_arrays_unchanged=True)
        rows.append(r); prior.append(dict(public_input_sha256='b'*64,
            original_decision_sha256='c'*64, candidate_decision_sha256='d'*64))
    launch = dict(source_sha256=sources, fused_result_sha256=check.FUSED_SHA,
        fused_launch_sha256=check.run.PREVIOUS_LAUNCH, frames=1428,
        state_frames=list(check.run.previous.profile.paired.previous.STATE_FRAMES),
        normalized_state_type_paths=check.run.STATE_TYPE_PATHS, baseline='FusedScopedBatchedController',
        candidate='PackedFusedScopedController', native_execution=False, model_training=False,
        environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled'))
    preceding = dict(sensing_scope={'original_visibility_failed':True}, report=dict(
        frames=1428, raw_model_forecast_comparisons=1425, model_state_sha256='f'*64,
        observed_state_checks=[dict(frame=i, state_sha256='a'*64, retained_observed_state_equal=True)
            for i in launch['state_frames']], model_state_unchanged=True,
        incremental_receipt_construction_comparison=True, both_controllers_use_scoped_reuse_and_batched_patches=True,
        baseline='ScopedBatchedFootprintController', candidate='FusedScopedBatchedController',
        timing_windows={}, normalized_state_type_paths=[]))
    report = deepcopy(preceding['report'])
    report.pop('incremental_receipt_construction_comparison'); report.pop('both_controllers_use_scoped_reuse_and_batched_patches')
    report.update(baseline=launch['baseline'], candidate=launch['candidate'],
        normalized_state_type_paths=launch['normalized_state_type_paths'], incremental_packed_insertion_comparison=True,
        both_controllers_use_fused_scoped_batched_receipts=True,
        timing_windows=check.run.previous.profile.paired.previous.timing_summary(rows))
    result = dict(status='PACKED_FUSED_SCOPED_LATE_HISTORY_REPLAY_V1_COMPLETE', source_sha256=sources,
        artifact_sha256={'launch.json':check.LAUNCH_SHA, 'comparison.jsonl':'b'*64}, report=report,
        native_execution=False, goal_achieved=False, wall_s=100., sensing_scope=deepcopy(preceding['sensing_scope']))
    return result, launch, rows, preceding, prior, sources


def test_complete_result_retains_slowdown_and_failed_scientific_scope():
    values = fixture(); timing = check.require_result(*values)
    all_rows = timing['all_navigation']
    assert all_rows['baseline_median_s'] == .2 and all_rows['candidate_median_s'] == .4
    assert all_rows['baseline_total_s'] == pytest.approx(1425*.2)
    assert all_rows['candidate_total_s'] == pytest.approx(1425*.4)
    assert all_rows['total_ratio'] == .5 and all_rows['candidate_over_100ms'] == 1425


@pytest.mark.parametrize('fault', ['missing_row', 'reordered', 'bool_frame', 'reference_packet', 'reference_baseline',
    'candidate_hash', 'flag', 'state', 'state_missing', 'model', 'summary', 'timing_negative', 'timing_nan',
    'execution_order', 'sensing', 'native', 'launch', 'environment', 'extra_normalization', 'extra_report_flag', 'source'])
def test_incomplete_tampered_or_relabelled_evidence_is_rejected(fault):
    r, launch, rows, preceding, prior, sources = fixture()
    if fault == 'missing_row': rows.pop()
    elif fault == 'reordered': rows[-1]['frame'] -= 1
    elif fault == 'bool_frame': rows[1]['frame'] = True
    elif fault == 'reference_packet': rows[-1]['public_input_sha256'] = '0'*64
    elif fault == 'reference_baseline': rows[-1]['baseline_decision_sha256'] = '0'*64
    elif fault == 'candidate_hash': rows[-1]['candidate_decision_sha256'] = 'unbound'
    elif fault == 'flag': rows[-1]['candidate_normalized_decision_exact'] = 1
    elif fault == 'state': r['report']['observed_state_checks'][-1]['state_sha256'] = '0'*64
    elif fault == 'state_missing': r['report']['observed_state_checks'].pop()
    elif fault == 'model': r['report']['model_state_sha256'] = '0'*64
    elif fault == 'summary': r['report']['timing_windows']['all_navigation']['candidate_total_s'] -= 1
    elif fault == 'timing_negative': rows[-1]['candidate_controller_s'] = -1
    elif fault == 'timing_nan': rows[-1]['candidate_controller_s'] = float('nan')
    elif fault == 'execution_order': rows[-1]['execution_order'] = [0, 1]
    elif fault == 'sensing': r['sensing_scope']['original_visibility_failed'] = False
    elif fault == 'native': r['native_execution'] = True
    elif fault == 'launch': r['artifact_sha256']['launch.json'] = '0'*64
    elif fault == 'environment': launch['environment']['OMP_NUM_THREADS'] = '2'
    elif fault == 'extra_normalization': launch['normalized_state_type_paths'] = list(check.run.STATE_TYPE_PATHS)+['memory.extra']
    elif fault == 'extra_report_flag': r['report']['real_time_qualified'] = True
    else: sources = {'synthetic.py':'0'*64}
    with pytest.raises(ValueError): check.require_result(r, launch, rows, preceding, prior, sources)


def test_live_owner_is_rejected_before_any_file_admission(monkeypatch):
    monkeypatch.setattr(check, 'owner_live', lambda owner:True)
    monkeypatch.setattr(check, 'verify_artifacts', lambda *a:pytest.fail('must wait for original owner'))
    with pytest.raises(ValueError, match='still live'): check.completed('a'*64, {})

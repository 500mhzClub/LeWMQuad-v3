"""Report scope must change only in declared implementation labels and timings."""
from copy import deepcopy
import pytest
from scripts import run_go2_deferred_memo_single_pass_late_history_v1 as run


def prior():
    return dict(incremental_single_pass_bounds_comparison=True,
        both_controllers_use_original_body_projection_and_receipt_handling=True,
        observed_state_checks=[dict(frame=1427,state_sha256='complete-state')],
        model_state_unchanged=True,complete_original_decisions_reconstructed=True,
        complete_normalized_candidate_decisions_exact=True,native_execution=False,navigation_qualified=False,
        real_time_qualified=False,timing_windows={'original':'timing'})


def test_report_preserves_all_unrelated_evidence_and_input_ownership():
    original=prior();before=deepcopy(original);timing={'new':'timing'}
    actual=run.expected_report(original,timing)
    assert original==before and actual['timing_windows']==timing
    assert actual['observed_state_checks']==before['observed_state_checks']
    assert actual['observed_state_checks'] is not original['observed_state_checks']
    assert actual['model_state_unchanged'] and not actual['native_execution']
    assert actual['only_two_pure_footprint_copiers_changed']


@pytest.mark.parametrize('key',['incremental_single_pass_bounds_comparison',
    'both_controllers_use_original_body_projection_and_receipt_handling'])
def test_different_predecessor_scope_cannot_be_relabeled(key):
    original=prior();original[key]=False
    with pytest.raises(ValueError): run.expected_report(original,{})


@pytest.mark.parametrize('fault',['state','model','scope','timing'])
def test_changed_report_evidence_is_rejected(monkeypatch,fault):
    original=prior();timing={'new':'timing'}
    monkeypatch.setattr(run,'check_rows',lambda rows,prior_rows:timing)
    report=run.expected_report(original,timing)
    if fault=='state': report['observed_state_checks'][0]['state_sha256']='changed'
    elif fault=='model': report['model_state_unchanged']=False
    elif fault=='scope': report['navigation_qualified']=True
    else: report['timing_windows']={'different':'time'}
    with pytest.raises(ValueError): run.validate_report(report,[],({},dict(report=original),[]))

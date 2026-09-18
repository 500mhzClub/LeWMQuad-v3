"""Full synthetic prefix reconstruction and terminal audit acceptance gates."""
from copy import deepcopy

import pytest

from scripts import measured_plane_chained_native_result_development as result
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS
from lewm.tests.test_measured_plane_chained_native_prefix_development import setup, fixture_report

CASE = ('synthetic_chained_case', 2, 'DISTINCTIVE', 'direct', 'no_rgb_direct')
STATUS = 'SYNTHETIC_CHAINED_COLLECTED_AND_RAW_AUDITED'


def collection(count=5):
    return dict(navigation_ticks=4000,
        status='RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED',
        decisions=count, command_ticks=count, completed_ticks=count,
        physics_samples=750+50*count)


def worker_fixture(tmp_path, monkeypatch, *, intervention=4, terminal_only=False):
    prior, current, _, report, documents, _ = setup(tmp_path, monkeypatch,
        intervention=intervention, terminal_only=terminal_only)
    audit = dict(raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True,
        raw_model_command_replay_pass=True, model_state_unchanged=True,
        verified_round_trip=False, native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={})
    record = dict(status=STATUS, case=CASE[0], layout_index=CASE[1], variant=CASE[2],
        condition=CASE[3], model_name=CASE[4],
        model_state_sha256=result.prefix.replay.inputs.job.MODEL_SHA, model_state_unchanged=True,
        measured_plane_constrained_estimator=True, collection=collection(intervention+1),
        **{k:deepcopy(audit[k]) for k in OUTCOME_KEYS})
    record['prefix_comparison'] = result.prefix_result(record['collection'], report, prior=prior, current=current)
    return record, audit, report, dict(case=CASE, worker_status=STATUS, prior=prior, current=current), documents


@pytest.mark.parametrize('intervention', [3, 17, 122])
def test_actual_dynamic_boundary_accepts_complete_negative(tmp_path, monkeypatch, intervention):
    record, audit, report, args, _ = worker_fixture(tmp_path, monkeypatch, intervention=intervention)
    actual = result.require_worker(record, audit, report, **args)
    assert actual['common_prefix_frames'] == intervention+1
    assert actual['first_changed_decision_frame'] == intervention
    assert actual['physical_prefix_samples'] == 750+50*intervention
    assert actual['actual_paired_execution_compared'] is True
    assert record['verified_round_trip'] is False


def test_terminal_only_intervention_remains_distinct_from_command_change(tmp_path, monkeypatch):
    record, audit, report, args, _ = worker_fixture(tmp_path, monkeypatch, terminal_only=True)
    actual = result.require_worker(record, audit, report, **args)
    assert actual['intervention_terminal_changed'] is True
    assert actual['intervention_command_changed'] is False


@pytest.mark.parametrize('fault', ['model', 'case', 'budget', 'population', 'raw', 'outcome',
    'saved_prefix', 'saved_hash', 'saved_flag_type', 'actual_decision', 'failure'])
def test_invalid_full_audit_or_fabricated_prefix_cannot_pass(tmp_path, monkeypatch, fault):
    record, audit, report, args, documents = worker_fixture(tmp_path, monkeypatch)
    if fault == 'model': record['model_state_sha256'] = '0'*64
    elif fault == 'case': record['condition'] = 'jepa'
    elif fault == 'budget': record['collection']['navigation_ticks'] = 5000
    elif fault == 'population': record['collection']['decisions'] = 4015
    elif fault == 'raw': audit['raw_model_command_replay_pass'] = False
    elif fault == 'outcome': record['verified_round_trip'] = True
    elif fault == 'saved_prefix': record['prefix_comparison']['first_changed_decision_frame'] += 1
    elif fault == 'saved_hash': record['prefix_comparison']['raw_physics_prefix_sha256'] = '0'*64
    elif fault == 'saved_flag_type': record['prefix_comparison']['actual_paired_execution_compared'] = 1
    elif fault == 'actual_decision': documents[args['current']][3]['decision']['evidence']['floor'] = 'changed'
    else: record['failure'] = 'preserved execution failure'
    with pytest.raises(ValueError): result.require_worker(record, audit, report, **args)


@pytest.mark.parametrize('physical,visibility,hard_failure', [
    (False,True,False), (True,False,False), (True,True,True), (True,True,False)])
def test_success_requires_physical_and_strict_sensing_evidence(tmp_path, monkeypatch, physical, visibility, hard_failure):
    record, audit, report, args, _ = worker_fixture(tmp_path, monkeypatch)
    success = physical and visibility and not hard_failure
    audit.update(verified_round_trip=success, strict_physical_visibility_pass=visibility,
        hard_measurement_failed_frames=[4] if hard_failure else [],
        native_evaluation=dict(native_round_trip_candidate_pass=physical))
    for k in OUTCOME_KEYS: record[k] = deepcopy(audit[k])
    result.require_worker(record, audit, report, **args)
    audit['verified_round_trip'] = record['verified_round_trip'] = not success
    with pytest.raises(ValueError, match='joint physical'):
        result.require_worker(record, audit, report, **args)


def test_early_negative_preserves_raw_audit_without_reading_missing_prefix(tmp_path, monkeypatch):
    record, audit, report, args, _ = worker_fixture(tmp_path, monkeypatch)
    record['collection'] = collection(3)
    monkeypatch.setattr(result.prefix, 'compare', lambda *a, **kw: pytest.fail('early missing physical interval read'))
    record['prefix_comparison'] = result.prefix_result(record['collection'], report,
        prior=args['prior'], current=args['current'])
    actual = result.require_worker(record, audit, report, **args)
    assert not actual['actual_paired_execution_compared'] and actual['full_raw_audit_retained']
    assert actual['collection_availability']['unavailable_reason'] == 'INTERVENTION_OBSERVATION_NOT_REACHED'
    audit['native_evaluation']['native_round_trip_candidate_pass'] = True
    audit['verified_round_trip'] = record['verified_round_trip'] = True
    record['native_evaluation'] = deepcopy(audit['native_evaluation'])
    with pytest.raises(ValueError, match='early negative'):
        result.require_worker(record, audit, report, **args)


def test_available_counts_cannot_skip_raw_reconstruction(monkeypatch):
    def reject(*a, **kw): raise ValueError('actual raw prefix rejected')
    monkeypatch.setattr(result.prefix, 'compare', reject)
    with pytest.raises(ValueError, match='actual raw prefix'):
        result.prefix_result(collection(), fixture_report(), prior=None, current=None)

"""Accept outcomes only with raw audits, resource receipts and actual prefixes."""
from copy import deepcopy
import json

import pytest

from scripts import extended_return_budget_native_result_development as result
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS
from lewm.tests.test_extended_return_budget_native_prefix_development import setup

CASE = ('synthetic_long', 2, 'DISTINCTIVE', 'direct', 'no_rgb_direct')
STATUS = 'SYNTHETIC_EXTENDED_COLLECTED_AND_RAW_AUDITED'


def fixture(tmp_path, monkeypatch):
    prior, current, report, ids, checks, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(result.inputs.native, 'OUTPUT', prior.parent)
    monkeypatch.setattr(result.inputs.native, 'CASE', (prior.name, *CASE[1:]))
    admission = dict(prefix_report=report, original_native_artifact_sha256=ids[0], prefix_artifact_sha256=ids[2])
    collection = dict(navigation_ticks=8000,
        status='RESIDUAL_ANCHORED_CONTINUATION_MAZE_TERMINAL_AUDIT_REQUIRED',
        decisions=5, command_ticks=5, completed_ticks=5, physics_samples=1000)
    audit = dict(raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True,
        raw_model_command_replay_pass=True, model_state_unchanged=True,
        verified_round_trip=False, native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={})
    resources = dict(synthetic_collection_complete=True, synthetic_audit_complete=True)
    monkeypatch.setattr(result.resource_audit, 'check', lambda *args: deepcopy(resources))
    record = dict(status=STATUS, case=CASE[0], layout_index=CASE[1], variant=CASE[2],
        condition=CASE[3], model_name=CASE[4], model_state_sha256=result.inputs.replay.pair.MODEL_SHA,
        model_state_unchanged=True, measured_plane_constrained_estimator=True,
        extended_return_budget_enabled=True, single_pass_timing_change_adopted=True,
        sampled_resource_guards_enabled=True, collection=collection, artifact_sha256=ids[1],
        resource_audit=resources, **{key:deepcopy(audit[key]) for key in OUTCOME_KEYS})
    args = dict(output=current.parent, case=CASE, worker_status=STATUS)
    record['prefix_comparison'] = result.prefix_result(collection, admission,
        output=current.parent, case=CASE, current_bindings=ids[1])
    persist(record, audit, args)
    return record, audit, admission, args, checks


def persist(record, audit, args):
    for suffix, data in (('_audit.json', audit), ('_resource_audit.json', record['resource_audit']),
            ('_prefix_comparison.json', record['prefix_comparison'])):
        (args['output']/(CASE[0]+suffix)).write_text(json.dumps(data))


def test_parent_reconstructs_actual_prefix_again_and_keeps_negative_outcome(tmp_path, monkeypatch):
    record, audit, admission, args, checks = fixture(tmp_path, monkeypatch)
    before = len(checks)
    receipt = result.require_worker(record, audit, admission, **args)
    assert len(checks) == before+6
    assert receipt['actual_paired_execution_compared'] and receipt['physical_and_public_prefix_exact']
    assert not record['verified_round_trip']


def test_worker_can_supply_its_fresh_receipt_without_recursive_reconstruction(tmp_path, monkeypatch):
    record, audit, admission, args, checks = fixture(tmp_path, monkeypatch)
    before = len(checks)
    result.require_worker(record, audit, admission, **args, prefix_receipt=record['prefix_comparison'])
    assert len(checks) == before


@pytest.mark.parametrize('fault', ['model', 'case', 'budget', 'population', 'raw', 'outcome',
    'prefix', 'prefix_flag_type', 'resources', 'closed_audit', 'closed_prefix', 'failure'])
def test_invalid_worker_or_changed_closed_evidence_is_rejected(tmp_path, monkeypatch, fault):
    record, audit, admission, args, _ = fixture(tmp_path, monkeypatch)
    if fault == 'model': record['model_state_sha256'] = '0'*64
    elif fault == 'case': record['condition'] = 'jepa'
    elif fault == 'budget': record['collection']['navigation_ticks'] = 4000
    elif fault == 'population': record['collection']['decisions'] = 8015
    elif fault == 'raw': audit['raw_model_command_replay_pass'] = False
    elif fault == 'outcome': record['verified_round_trip'] = True
    elif fault == 'prefix': record['prefix_comparison']['physical_prefix_samples'] += 1
    elif fault == 'prefix_flag_type': record['prefix_comparison']['actual_paired_execution_compared'] = 1
    elif fault == 'resources': record['resource_audit'] = {'fabricated': True}
    elif fault in ('closed_audit', 'closed_prefix'):
        suffix = '_audit.json' if fault == 'closed_audit' else '_prefix_comparison.json'
        (args['output']/(CASE[0]+suffix)).write_text('{}')
    else: record['failure'] = 'original failure'
    with pytest.raises(ValueError): result.require_worker(record, audit, admission, **args)


@pytest.mark.parametrize('physical,visibility,hard_failure', [
    (False, True, False), (True, False, False), (True, True, True), (True, True, False)])
def test_joint_success_criteria_are_not_weakened(tmp_path, monkeypatch, physical, visibility, hard_failure):
    record, audit, admission, args, _ = fixture(tmp_path, monkeypatch)
    success = physical and visibility and not hard_failure
    audit.update(verified_round_trip=success, strict_physical_visibility_pass=visibility,
        hard_measurement_failed_frames=[4] if hard_failure else [],
        native_evaluation=dict(native_round_trip_candidate_pass=physical))
    record.update({key:deepcopy(audit[key]) for key in OUTCOME_KEYS}); persist(record, audit, args)
    result.require_worker(record, audit, admission, **args)
    audit['verified_round_trip'] = record['verified_round_trip'] = not success
    with pytest.raises(ValueError, match='joint physical'):
        result.require_worker(record, audit, admission, **args)


def test_early_failure_cannot_claim_unexecuted_prefix_or_success(tmp_path, monkeypatch):
    record, audit, admission, args, checks = fixture(tmp_path, monkeypatch)
    record['collection'].update(decisions=3, command_ticks=3, completed_ticks=3, physics_samples=900)
    before = len(checks)
    receipt = result.prefix_result(record['collection'], admission, output=args['output'], case=CASE,
        current_bindings=record['artifact_sha256'])
    record['prefix_comparison'] = receipt; persist(record, audit, args)
    result.require_worker(record, audit, admission, **args)
    assert len(checks) == before and not receipt['actual_paired_execution_compared']
    assert receipt['full_raw_audit_retained'] and not receipt['unexecuted_outcomes_inferred']
    audit['native_evaluation']['native_round_trip_candidate_pass'] = True
    audit['verified_round_trip'] = True
    record.update({key:deepcopy(audit[key]) for key in OUTCOME_KEYS}); persist(record, audit, args)
    with pytest.raises(ValueError, match='early negative'):
        result.require_worker(record, audit, admission, **args)

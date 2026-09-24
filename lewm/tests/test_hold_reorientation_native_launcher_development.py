from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import run_go2_hold_reorientation_maze02_pilot_v1 as native
from scripts import hold_reorientation_native_inputs_development as inputs


def fixture():
    prefix = dict(frames=406, first_changed_command_frame=405, raw_model_forecast_comparisons=403,
        original_requested_command=[0., 0., 0.], candidate_requested_command=[0., 0., .45])
    audit = dict(layout_index=2, hold_reorientation_enabled=True,
        raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True, raw_model_command_replay_pass=True,
        model_state_unchanged=True, native_evaluation=dict(native_round_trip_candidate_pass=False),
        strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={},
        verified_round_trip=False)
    receipt = dict(common_prefix_frames=406, first_intervention_frame=405, physical_prefix_samples=21000,
        raw_model_forecast_comparisons=403, physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        all_compared_raw_model_forecasts_exact=True, candidate_intervention_command_completed=True,
        original_intervention_command=[0., 0., 0.], candidate_intervention_command=[0., 0., .45],
        following_physical_outcomes_compared=False, unexecuted_outcomes_inferred=False)
    record = dict(status=native.WORKER_STATUS, case=native.CASE[0], layout_index=2,
        model_name=native.CASE[4], condition='jepa', variant='full',
        model_state_sha256=inputs.replay.MODEL_SHA, model_state_unchanged=True,
        collection=dict(status='HOLD_REORIENTATION_MAZE02_TERMINAL_AUDIT_REQUIRED', hold_reorientation_enabled=True),
        prefix_comparison=receipt, **{k:deepcopy(audit[k]) for k in native.OUTCOME_KEYS})
    return record, audit, prefix


def test_scientific_failure_with_full_raw_and_physical_evidence_is_retained():
    native.require_worker(*fixture())


@pytest.mark.parametrize('fault', ['model', 'variant', 'condition', 'collection', 'raw', 'intervention',
    'frames', 'physics', 'forecasts', 'command', 'counterfactual', 'false_success'])
def test_incomplete_changed_or_overclaimed_native_result_is_rejected(fault):
    record, audit, prefix = fixture()
    if fault == 'model': record['model_state_sha256'] = 'other'
    if fault == 'variant': record['variant'] = 'no_rgb'
    if fault == 'condition': record['condition'] = 'supervised_rollout'
    if fault == 'collection': record['collection']['hold_reorientation_enabled'] = False
    if fault == 'raw': audit['raw_model_command_replay_pass'] = False
    if fault == 'intervention': record['prefix_comparison']['candidate_intervention_command_completed'] = False
    if fault == 'frames': record['prefix_comparison']['common_prefix_frames'] = 405
    if fault == 'physics': record['prefix_comparison']['physical_prefix_samples'] = 20999
    if fault == 'forecasts': record['prefix_comparison']['raw_model_forecast_comparisons'] = 402
    if fault == 'command': record['prefix_comparison']['candidate_intervention_command'] = [0., 0., -.45]
    if fault == 'counterfactual': record['prefix_comparison']['following_physical_outcomes_compared'] = True
    if fault == 'false_success': record['verified_round_trip'] = audit['verified_round_trip'] = True
    with pytest.raises(ValueError): native.require_worker(record, audit, prefix)


def test_incompatible_original_wrapper_rejected_before_execution(monkeypatch):
    monkeypatch.setattr(native, 'load_assigned', lambda *a: (SimpleNamespace(training=False), 'jepa', 'full'))
    with pytest.raises(ValueError, match='assigned'):
        native.assigned_model(dict(input_admission=dict(correction_admission={})))


def test_both_exact_prior_waiters_required_without_retries():
    raw = dict(status='HOLD_REORIENTATION_RAW_PREFIX_WAIT_V1_COMPLETE', native_execution=False, automatic_retry=False)
    frontier = dict(status='REACHED_FRONTIER_MAZE03_NATIVE_WAIT_COMPLETE', automatic_retry=False)
    inputs.require_waits(raw, frontier)
    for key in ('status', 'native_execution', 'automatic_retry'):
        bad = deepcopy(raw); bad[key] = 'incomplete' if key == 'status' else True
        with pytest.raises(ValueError): inputs.require_waits(bad, frontier)
    for key in ('status', 'automatic_retry'):
        bad = deepcopy(frontier); bad[key] = 'incomplete' if key == 'status' else True
        with pytest.raises(ValueError): inputs.require_waits(raw, bad)


def test_source_preflight_never_admits_inputs_or_creates_runtime_output(monkeypatch, tmp_path):
    monkeypatch.setattr(native, 'OUTPUT', tmp_path/'uncreated')
    monkeypatch.setattr(native, 'validate_root', lambda *a, **k: None)
    monkeypatch.setattr(native.inputs, 'prepared_sources', lambda *a: {})
    monkeypatch.setattr(native, 'hardware', lambda: dict(memory_available_bytes=64*1024**3, artifact_free_bytes=100*1024**3))
    monkeypatch.setattr(native.inputs, 'admit', lambda *a: pytest.fail('no input admission'))
    monkeypatch.setattr(native, 'create_output', lambda *a: pytest.fail('no output creation'))
    monkeypatch.setattr('sys.argv', ['native', '--source-preflight-only'])
    native.main()
    assert not native.OUTPUT.exists()


def admission_fixture(monkeypatch, fault=None):
    raw_root, front_root = inputs.raw_wait.OUTPUT, inputs.frontier_wait.OUTPUT
    native_root, prefix_root, batch_root = inputs.frontier_wait.native.OUTPUT, inputs.replay.OUTPUT, inputs.replay.original.OUTPUT
    original = dict(original_artifact_sha256={'worker': 'worker-sha'})
    raw_report = dict(original_worker_terminal_sha256='terminal-sha', raw_replay_result_sha256='prefix-result')
    front_report = dict(native_result_sha256='frontier-result')
    raw = dict(status='HOLD_REORIENTATION_RAW_PREFIX_WAIT_V1_COMPLETE', native_execution=False,
        automatic_retry=False, report=deepcopy(raw_report))
    frontier = dict(status='REACHED_FRONTIER_MAZE03_NATIVE_WAIT_COMPLETE', automatic_retry=False, report=deepcopy(front_report))
    native_result = dict(adapter_batch_result_sha256='batch-result')
    native_launch = dict(input_admission=dict(adapter_batch_result_sha256='batch-result'))
    prefix_launch = dict(input_admission=deepcopy(original))
    if fault == 'raw_report': raw['report']['raw_replay_result_sha256'] = 'changed'
    if fault == 'front_report': frontier['report']['native_result_sha256'] = 'changed'
    if fault == 'original_admission': prefix_launch['input_admission']['changed'] = True
    if fault == 'different_batch': native_launch['input_admission']['adapter_batch_result_sha256'] = 'different'
    batch_ids = {'worker': 'worker-sha' if fault != 'changed_worker' else 'changed'}
    results = {raw_root:(raw, {}, {'result.json':'raw-wait-result'}),
        front_root:(frontier, {}, {'result.json':'front-wait-result'}),
        native_root:(native_result, native_launch, {'result.json':'frontier-result'}),
        prefix_root:({}, prefix_launch, {'result.json':'prefix-result'}),
        batch_root:({}, {'input_admission':{'correction_admission':{'same':'model'}}}, batch_ids)}
    records = {(front_root,'input_completion.json'):{'adapter_batch_result_sha256':'batch-result'},
        (front_root,'native_completion.json'):front_report,
        (raw_root,'input_completion.json'):{'original_worker_terminal_sha256':'terminal-sha'},
        (raw_root,'replay_completion.json'):raw_report}
    full = []; completed_calls = []
    def completed(root, sha, launch_sha, sources):
        completed_calls.append((root,sha,launch_sha)); return deepcopy(results[root])
    monkeypatch.setattr(inputs, 'completed', completed)
    monkeypatch.setattr(inputs, 'read_json', lambda root, name: deepcopy(records[root,name]))
    monkeypatch.setattr(inputs, 'digest', lambda path: 'launch-sha')
    monkeypatch.setattr(inputs.frontier_wait, 'authenticate_completed', lambda *a: deepcopy(front_report))
    monkeypatch.setattr(inputs.raw_wait, 'authenticate_replay', lambda *a: deepcopy(raw_report))
    monkeypatch.setattr(inputs.frontier_wait.native, 'verify_inputs', lambda launch, **kwargs: full.append(kwargs))
    monkeypatch.setattr(inputs.replay, 'admit_worker', lambda *a: deepcopy(original))
    monkeypatch.setattr(inputs, 'admit_prefix', lambda *a: fixture()[2])
    return full, completed_calls


def test_input_admission_links_same_first_worker_to_both_completed_chains(monkeypatch):
    full, calls = admission_fixture(monkeypatch)
    result = inputs.admit('raw-wait-result', 'front-wait-result', {})
    assert full == [{'full':True}]
    assert [r for r,_,_ in calls] == [inputs.frontier_wait.OUTPUT, inputs.raw_wait.OUTPUT,
        inputs.frontier_wait.native.OUTPUT, inputs.replay.OUTPUT, inputs.replay.original.OUTPUT]
    assert result['original_worker_terminal_sha256'] == 'terminal-sha'
    assert result['scheduled_frontier_completed_before_this_experiment'] is True
    assert result['all_six_adapter_cases_completed'] is True
    assert result['correction_admission'] == {'same':'model'}
    assert len(result['completed_artifact_bindings']) == 5


@pytest.mark.parametrize('fault', ['raw_report', 'front_report', 'original_admission', 'different_batch', 'changed_worker'])
def test_mismatched_completion_chain_cannot_be_substituted(monkeypatch, fault):
    admission_fixture(monkeypatch, fault)
    with pytest.raises(ValueError): inputs.admit('raw-wait-result', 'front-wait-result', {})

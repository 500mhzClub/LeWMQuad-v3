"""Native composition, completed-input admission and source-only isolation."""
from copy import deepcopy
import pickle

import pytest

from scripts import run_go2_measured_plane_chained_maze02_v1 as native
from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorController


def isolated_start(monkeypatch, tmp_path):
    output = tmp_path/'attempt'
    monkeypatch.setattr(native, 'OUTPUT', output)
    monkeypatch.setattr(native.run, 'validate_root', lambda *a, **kw: None)
    monkeypatch.setattr(native.inputs, 'prepared_sources', lambda seeds: {'source': 'sha'})
    monkeypatch.setattr(native, 'resources', lambda: {'synthetic_resources': True})
    monkeypatch.setattr(native, 'assigned_model', lambda: pytest.fail('model before completed input admission'))
    monkeypatch.setattr(native.run, 'create_output', lambda *a: pytest.fail('output before completed input admission'))
    monkeypatch.setattr(native, 'wait_for_idle', lambda: pytest.fail('scene serialization before input admission'))
    return output


def test_source_preflight_does_not_admit_inputs_load_model_or_create_output(monkeypatch, tmp_path):
    output = isolated_start(monkeypatch, tmp_path)
    monkeypatch.setattr(native.inputs, 'admit', lambda *a: pytest.fail('runtime admission during source preflight'))
    native.main(source_only=True)
    assert not output.exists()


def test_runtime_requires_actual_completed_waiter_identity(monkeypatch, tmp_path):
    output = isolated_start(monkeypatch, tmp_path)
    monkeypatch.setattr(native.inputs, 'admit', lambda *a: pytest.fail('missing identity passed to admission'))
    with pytest.raises(ValueError, match='completed chained waiter'):
        native.main()
    assert not output.exists()


@pytest.mark.parametrize('preflight', [False, True])
def test_unfinished_or_failed_replay_cannot_create_native_attempt(monkeypatch, tmp_path, preflight):
    output = isolated_start(monkeypatch, tmp_path)
    calls = []
    def reject(sha, sources):
        calls.append((sha, sources))
        raise ValueError('original queue owners must finish and end first')
    monkeypatch.setattr(native.inputs, 'admit', reject)
    with pytest.raises(ValueError, match='owners must finish'):
        native.main('0'*64, preflight=preflight)
    assert calls == [('0'*64, {'source': 'sha'})] and not output.exists()


@pytest.mark.parametrize('existing', ['directory', 'dangling_symlink'])
def test_exclusive_attempt_cannot_be_reused(monkeypatch, tmp_path, existing):
    output = isolated_start(monkeypatch, tmp_path)
    if existing == 'directory': output.mkdir()
    else: output.symlink_to(tmp_path/'absent')
    monkeypatch.setattr(native.inputs, 'prepared_sources', lambda *a: pytest.fail('read inputs for existing attempt'))
    with pytest.raises(ValueError, match='exclusive'):
        native.main(source_only=True)


def test_actual_worker_and_pipeline_bind_candidate_without_mutating_original():
    assert native._worker.__code__ is native.original.worker.__code__
    allowed = {'OUTPUT', 'PROTOCOL', 'CASE', 'WORKER_STATUS', 'inputs', 'pipeline',
        'verify_inputs', 'prefix_result', 'require_worker', 'assigned_model'}
    for key, value in native.original.worker.__globals__.items():
        if key not in allowed: assert native._worker.__globals__[key] is value
    assert native._worker.__globals__['pipeline'] is native.pipeline
    assert native._worker.__globals__['inputs'].job is native.prefix.replay.inputs.job
    assert native._worker.__globals__['verify_inputs'] is native.verify_inputs
    assert native._worker.__globals__['require_worker'] is native.require_worker
    assert native._worker.__globals__['prefix_result'] is native.prefix_result
    for function in (native.pipeline.collect, native.pipeline.audit):
        assert function.__globals__['ResidualAnchoredContinuationController'] is MeasuredPlaneChainedAnchorController
    assert native.original.worker.__globals__['pipeline'] is not native.pipeline
    assert pickle.loads(pickle.dumps(native.worker)) is native.worker


def test_actual_definition_retains_budget_model_and_only_registered_tracking_candidate():
    definition = native.definition()
    assert definition['implementation_class'] == 'MeasuredPlaneChainedAnchorController'
    assert definition['planned_case'] == list(native.CASE)
    assert definition['output_root'] == str(native.OUTPUT)
    assert definition['navigation_ticks'] == 4000
    assert definition['model_state_sha256'] == native.prefix.replay.inputs.job.MODEL_SHA
    assert definition['original_bridge_allowance_unchanged'] is True
    assert definition['single_pass_timing_change_adopted'] is False
    assert definition['actual_completed_replay_boundary_required'] is True
    assert definition['native_scene_workers'] == definition['maximum_tasks_per_process'] == 1
    assert definition['physics_paused_during_compute'] is True


def test_worker_acceptance_uses_saved_completed_dynamic_report(monkeypatch):
    report = {'synthetic_dynamic_report': True}; record = {}; audit = {}; calls = []
    monkeypatch.setattr(native.run, 'read_json', lambda root, name: {'input_admission': {'prefix_report': report}})
    def check(saved, raw, actual, **kwargs):
        calls.append((saved, raw, actual, kwargs)); return 'reconstructed'
    monkeypatch.setattr(native.results, 'require_worker', check)
    assert native.require_worker(record, audit) == 'reconstructed'
    assert calls == [(record, audit, report, dict(case=native.CASE, worker_status=native.WORKER_STATUS,
        prior=native.learned.OUTPUT/native.original.CASE[0], current=native.OUTPUT/native.CASE[0]))]


@pytest.mark.parametrize('fault', [None, 'definition', 'matched_scene', 'admission'])
def test_launch_revalidates_definition_matching_and_completed_receipt(monkeypatch, fault):
    definition = native.definition()
    baseline = {k:deepcopy(definition.get(k, 'synthetic-'+k)) for k in native.MATCHED_KEYS}
    admission = dict(chained_wait_result_sha256='a'*64, prefix_report={'synthetic': True})
    launch = baseline | definition | dict(source_sha256={}, input_admission=deepcopy(admission))
    monkeypatch.setattr(native.original, 'require_environment', lambda value: None)
    monkeypatch.setattr(native.original.old, 'verify_ordered_launch', lambda value: None)
    monkeypatch.setattr(native.prefix.replay.inputs, 'learned_launch', lambda: baseline)
    monkeypatch.setattr(native.inputs, 'admit', lambda sha, sources: admission)
    if fault == 'definition': launch['original_bridge_allowance_unchanged'] = False
    elif fault == 'matched_scene': launch['native_scene_sha256'] = 'changed'
    elif fault == 'admission': launch['input_admission']['prefix_report'] = {'changed': True}
    if fault:
        with pytest.raises(ValueError): native.verify_inputs(launch)
    else: native.verify_inputs(launch)

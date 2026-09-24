"""Full causal paired-loop checks for the receipt-only successor."""
import json
import sys
from itertools import islice
from types import SimpleNamespace

import pytest
from scripts import replay_go2_fused_scoped_batched_late_history_v1 as replay
from lewm import fused_scoped_batched_controller_development as fused
from lewm import scoped_batched_footprint_controller_development as combined
from lewm.tests.test_scoped_batched_footprint_late_history_replay_development import combined_fixture


def fixture(monkeypatch, tmp_path, fault=None):
    rows, report, tape, calls, models = combined_fixture(monkeypatch, tmp_path, fault)
    paired = replay.profile.paired
    previous = paired.previous
    fake = paired.ScopedBatchedFootprintController
    class Baseline(fake):
        index = 0
    class Candidate(fake):
        index = 1
        def observe(self, *args, **kwargs):
            return super().observe(*args, **kwargs) | {'controller': fused.CONTROLLER, fused.FLAG: True}
    monkeypatch.setattr(replay, 'ScopedBatchedFootprintController', Baseline)
    monkeypatch.setattr(replay, 'FusedScopedBatchedController', Candidate)
    monkeypatch.setattr(replay, 'OUTPUT', tmp_path)
    for i, row in enumerate(islice(previous.profile.read_rows(None), previous.FRAMES)):
        decision = row['decision'] | {previous.FLAG: True, 'controller': combined.CONTROLLER,
            combined.FLAG: True, 'batched_retained_floor_queries_enabled': True}
        rows[i]['candidate_decision_sha256'] = previous.profile.reference.saved.identity(decision)
    return rows, report, tape, calls, models


def test_exact_original_loop_uses_actual_new_pair_without_global_mutation():
    original = replay.profile.paired.previous.replay
    before = dict(original.__globals__)
    function = replay.isolated_replay()
    assert function.__code__ is original.__code__
    assert function.__globals__['FrozenFootprintAnchoredController'] is replay.ScopedBatchedFootprintController
    assert function.__globals__['ScopedFootprintAnchoredController'] is replay.FusedScopedBatchedController
    assert function.__globals__['profile'] is not replay.profile.paired.previous.profile
    assert all(original.__globals__[k] is v for k,v in before.items())


def test_complete_history_and_predecessor_states(monkeypatch, tmp_path):
    rows, prior, _, calls, models = fixture(monkeypatch, tmp_path)
    result = replay.replay(rows, prior)
    assert calls == [(i,j) for i in range(1428) for j in replay.profile.paired.previous.execution_order(i)]
    assert models[0] is not models[1]
    assert result['observed_state_checks'] == prior['observed_state_checks']
    assert result['frames'] == 1428 and result['raw_model_forecast_comparisons'] == 1425
    assert result['baseline'] == 'ScopedBatchedFootprintController'
    assert result['candidate'] == 'FusedScopedBatchedController'
    assert result['normalized_state_type_paths'] == replay.profile.paired.STATE_TYPE_PATHS
    assert not result['native_execution'] and not result['real_time_qualified']


@pytest.mark.parametrize('fault', ['receipt', 'input', 'metadata', 'terminal', 'state', 'gradient',
                                  'weight', 'shared_model', 'shared_storage'])
def test_original_corruption_and_shared_model_rejections(monkeypatch, tmp_path, fault):
    rows, prior, _, _, _ = fixture(monkeypatch, tmp_path, fault)
    with pytest.raises(ValueError): replay.replay(rows, prior)


@pytest.mark.parametrize('fault', ['endpoint', 'input_hash', 'original_hash', 'baseline_hash', 'prior_state'])
def test_reference_bindings_cannot_change(monkeypatch, tmp_path, fault):
    rows, prior, tape, _, _ = fixture(monkeypatch, tmp_path)
    if fault == 'endpoint': tape[1001]['post_sample_index'] += 1
    elif fault == 'prior_state': prior['observed_state_checks'][-1]['state_sha256'] = 'f'*64
    else:
        key = dict(input_hash='public_input_sha256', original_hash='original_decision_sha256',
                   baseline_hash='candidate_decision_sha256')[fault]
        rows[1001][key] = 'wrong'
    with pytest.raises(ValueError): replay.replay(rows, prior)


@pytest.mark.parametrize('mode', ['live', 'reused', 'gone', 'boot'])
def test_profile_must_end_with_exact_owner_identity(monkeypatch, mode):
    monkeypatch.setattr(replay.Path, 'read_text', lambda *a,**k:
                        'different' if mode == 'boot' else replay.profile.paired.previous.BOOT)
    def process(pid):
        assert pid == replay.PROFILE_OWNER['pid']
        if mode == 'gone': raise replay.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda: replay.PROFILE_OWNER['created']+(mode == 'reused'),
                               cmdline=lambda: replay.PROFILE_OWNER['command'])
    monkeypatch.setattr(replay.psutil, 'Process', process)
    if mode == 'gone': replay.profile_owner_ended()
    else:
        with pytest.raises(ValueError): replay.profile_owner_ended()


def prepare_main(monkeypatch, tmp_path):
    for k,v in dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled').items():
        monkeypatch.setenv(k,v)
    monkeypatch.setattr(replay, 'OUTPUT', tmp_path/'attempt')
    monkeypatch.setattr(replay, 'validate_root', lambda *a,**k: None)
    monkeypatch.setattr(replay, 'prepared_sources', lambda: {})
    monkeypatch.setattr(replay.profile.original.reference, 'hardware', lambda: dict(
        memory_available_bytes=64*1024**3, artifact_free_bytes=41*1024**3, physical_cpus=4))


def test_preflight_never_reads_raw_data_or_runs_controller(monkeypatch, tmp_path):
    prepare_main(monkeypatch, tmp_path)
    def forbidden(*a,**k): pytest.fail('runtime during preflight')
    for name in ('completed_reference', 'create_output', 'replay'):
        monkeypatch.setattr(replay, name, forbidden)
    monkeypatch.setattr(replay.profile, 'bound_profile_inputs', forbidden)
    monkeypatch.setattr(sys, 'argv', [replay.SOURCE, '--source-preflight-only'])
    replay.main()
    assert not replay.OUTPUT.exists()


def test_changed_final_input_preserves_failure_and_prevents_relaunch(monkeypatch, tmp_path):
    prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(replay, 'profile_owner_ended', lambda: None)
    monkeypatch.setattr(replay, 'create_output', lambda p: p.mkdir())
    monkeypatch.setattr(replay, 'completed_reference', lambda *a: ({'report': {}, 'sensing_scope': {}}, {}, []))
    admissions = iter([{'identity': 'original'}, {'identity': 'changed'}])
    monkeypatch.setattr(replay.profile, 'bound_profile_inputs', lambda *a: next(admissions))
    def run(*a):
        (replay.OUTPUT/'comparison.jsonl').write_text('retained partial evidence\n')
        return {}
    monkeypatch.setattr(replay, 'replay', run)
    monkeypatch.setattr(sys, 'argv', [replay.SOURCE])
    with pytest.raises(ValueError, match='original raw/model inputs'): replay.main()
    assert json.loads((replay.OUTPUT/'failure.json').read_text())['automatic_retry'] is False
    assert (replay.OUTPUT/'comparison.jsonl').read_text() == 'retained partial evidence\n'
    assert not (replay.OUTPUT/'result.json').exists()
    with pytest.raises(ValueError, match='no retry or resume'): replay.main()

"""Full synthetic histories, completed-reference rejection and runtime gating."""
from copy import deepcopy
from itertools import islice
import json
import sys
from types import SimpleNamespace

import pytest
from scripts import replay_go2_packed_fused_scoped_late_history_v1 as replay
from lewm import fused_scoped_batched_controller_development as fused
from lewm import packed_fused_scoped_controller_development as packed
from lewm import scoped_batched_footprint_controller_development as combined
from lewm.tests.test_fused_scoped_batched_replay_development import fixture as fused_fixture
from lewm.tests.test_scoped_batched_footprint_late_history_profile_development import fixture as completed_fixture


def fixture(monkeypatch, tmp_path, fault=None):
    rows, prior, tape, calls, models = fused_fixture(monkeypatch, tmp_path, fault)
    previous = replay.previous
    original = previous.profile.paired.previous
    fake = previous.FusedScopedBatchedController
    class Baseline(fake):
        index = 0
    class Candidate(fake):
        index = 1
        def observe(self, *args, **kwargs):
            return super().observe(*args, **kwargs) | {'controller': packed.CONTROLLER, packed.FLAG: True}
    monkeypatch.setattr(replay, 'FusedScopedBatchedController', Baseline)
    monkeypatch.setattr(replay, 'PackedFusedScopedController', Candidate)
    monkeypatch.setattr(replay, 'OUTPUT', tmp_path)
    # Synthetic memory omits robot indices; the actual ten-path checker is
    # covered by real-observation composition tests, without normalizing data.
    monkeypatch.setattr(replay, 'normalized_state_tree', original.state_tree)
    for i,row in enumerate(islice(original.profile.read_rows(None), original.FRAMES)):
        decision = row['decision'] | {original.FLAG: True, 'controller': fused.CONTROLLER,
            combined.FLAG: True, fused.FLAG: True, 'batched_retained_floor_queries_enabled': True}
        rows[i]['candidate_decision_sha256'] = original.profile.reference.saved.identity(decision)
    return rows, prior, tape, calls, models


def test_original_loop_and_isolated_actual_pair():
    original = replay.previous.profile.paired.previous.replay
    before = dict(original.__globals__)
    clone = replay.isolated_replay()
    assert clone.__code__ is original.__code__
    assert clone.__globals__['FrozenFootprintAnchoredController'] is replay.FusedScopedBatchedController
    assert clone.__globals__['ScopedFootprintAnchoredController'] is replay.PackedFusedScopedController
    assert clone.__globals__['state_tree'] is replay.normalized_state_tree
    assert all(original.__globals__[k] is v for k,v in before.items())


def test_full_history_and_all_predecessor_states(monkeypatch, tmp_path):
    rows, prior, _, calls, models = fixture(monkeypatch, tmp_path)
    result = replay.replay(rows, prior)
    assert calls == [(i,j) for i in range(1428) for j in replay.previous.profile.paired.previous.execution_order(i)]
    assert models[0] is not models[1]
    assert result['frames'] == 1428 and result['raw_model_forecast_comparisons'] == 1425
    assert result['observed_state_checks'] == prior['observed_state_checks']
    assert result['baseline'] == 'FusedScopedBatchedController'
    assert result['candidate'] == 'PackedFusedScopedController'
    assert result['normalized_state_type_paths'] == replay.STATE_TYPE_PATHS
    assert result['incremental_packed_insertion_comparison']
    assert not result['native_execution'] and not result['real_time_qualified']


@pytest.mark.parametrize('fault', ['receipt', 'input', 'metadata', 'terminal', 'state', 'gradient',
                                  'weight', 'shared_model', 'shared_storage'])
def test_all_original_corruption_rejections(monkeypatch, tmp_path, fault):
    rows, prior, _, _, _ = fixture(monkeypatch, tmp_path, fault)
    with pytest.raises(ValueError): replay.replay(rows, prior)


@pytest.mark.parametrize('fault', ['endpoint', 'input_hash', 'original_hash', 'baseline_hash', 'prior_state'])
def test_original_reference_and_state_binding_rejections(monkeypatch, tmp_path, fault):
    rows, prior, tape, _, _ = fixture(monkeypatch, tmp_path)
    if fault == 'endpoint': tape[1001]['post_sample_index'] += 1
    elif fault == 'prior_state': prior['observed_state_checks'][-1]['state_sha256'] = 'f'*64
    else:
        rows[1001][dict(input_hash='public_input_sha256', original_hash='original_decision_sha256',
                        baseline_hash='candidate_decision_sha256')[fault]] = 'wrong'
    with pytest.raises(ValueError): replay.replay(rows, prior)


def completion():
    result, launch, rows, prior, prior_rows, sources = completed_fixture()
    result['status'] = 'FUSED_SCOPED_BATCHED_LATE_HISTORY_REPLAY_V1_COMPLETE'
    result['artifact_sha256']['launch.json'] = replay.PREVIOUS_LAUNCH
    report = result['report']
    report.update(baseline='ScopedBatchedFootprintController', candidate='FusedScopedBatchedController',
        incremental_receipt_construction_comparison=True, both_controllers_use_scoped_reuse_and_batched_patches=True)
    report.pop('incremental_batching_comparison'); report.pop('both_controllers_use_scoped_reuse')
    launch.pop('scoped_result_sha256')
    launch.update(paired_result_sha256=replay.previous.PAIRED_SHA, profile_result_sha256=replay.previous.PROFILE_SHA)
    return result, launch, rows, prior, prior_rows, sources


def test_complete_exact_fused_reference_accepted():
    replay.require_completed(*completion())


@pytest.mark.parametrize('fault', ['status', 'launch', 'profile', 'paired', 'source', 'artifact', 'partial',
    'order', 'timing', 'input', 'original', 'baseline', 'candidate_hash', 'row_flag', 'state', 'scope',
    'policy', 'normalization', 'model', 'forecasts', 'qualification', 'goal', 'profiling', 'receipt_flag'])
def test_incomplete_or_changed_fused_reference_rejected(fault):
    args = completion(); result, launch, rows, prior, prior_rows, sources = args
    if fault == 'status': result['status'] = 'RUNNING'
    elif fault == 'launch': result['artifact_sha256']['launch.json'] = '0'*64
    elif fault == 'profile': launch['profile_result_sha256'] = '0'*64
    elif fault == 'paired': launch['paired_result_sha256'] = '0'*64
    elif fault == 'source': args = (*args[:-1], dict(sources, **{'synthetic.py': '0'*64}))
    elif fault == 'artifact': result['artifact_sha256'].pop('comparison.jsonl')
    elif fault == 'partial': rows.pop()
    elif fault == 'order': rows[-1]['frame'] -= 1
    elif fault == 'timing': rows[-1]['candidate_controller_s'] *= 2
    elif fault == 'input': rows[-1]['public_input_sha256'] = '0'*64
    elif fault == 'original': rows[-1]['original_decision_sha256'] = '0'*64
    elif fault == 'baseline': rows[-1]['baseline_decision_sha256'] = '0'*64
    elif fault == 'candidate_hash': rows[-1]['candidate_decision_sha256'] = 'malformed'
    elif fault == 'row_flag': rows[-1]['candidate_normalized_decision_exact'] = False
    elif fault == 'state': result['report']['observed_state_checks'][-1]['state_sha256'] = '0'*64
    elif fault == 'scope': result['sensing_scope']['synthetic_negative'] = False
    elif fault == 'policy': result['report']['candidate'] = 'different'
    elif fault == 'normalization': result['report']['normalized_state_type_paths'] = []
    elif fault == 'model': result['report']['model_state_sha256'] = '0'*64
    elif fault == 'forecasts': result['report']['raw_model_forecast_comparisons'] -= 1
    elif fault == 'qualification': result['report']['navigation_qualified'] = True
    elif fault == 'goal': result['goal_achieved'] = True
    elif fault == 'profiling': result['report']['profiling_enabled'] = True
    else: result['report']['incremental_receipt_construction_comparison'] = False
    with pytest.raises(ValueError): replay.require_completed(*args)


@pytest.mark.parametrize('mode', ['live', 'reused', 'gone', 'boot'])
def test_exact_original_owner_must_end(monkeypatch, mode):
    monkeypatch.setattr(replay.Path, 'read_text', lambda *a,**k:
        'different' if mode == 'boot' else replay.previous.profile.paired.previous.BOOT)
    def process(pid):
        assert pid == replay.PREVIOUS_OWNER['pid']
        if mode == 'gone': raise replay.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda: replay.PREVIOUS_OWNER['created']+(mode == 'reused'),
                               cmdline=lambda: replay.PREVIOUS_OWNER['command'])
    monkeypatch.setattr(replay.psutil, 'Process', process)
    if mode == 'gone': replay.previous_owner_ended()
    else:
        with pytest.raises(ValueError): replay.previous_owner_ended()


def prepare_main(monkeypatch, tmp_path):
    for k,v in dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled').items(): monkeypatch.setenv(k,v)
    monkeypatch.setattr(replay, 'OUTPUT', tmp_path/'attempt')
    monkeypatch.setattr(replay, 'validate_root', lambda *a,**k: None)
    monkeypatch.setattr(replay, 'prepared_sources', lambda: {})
    monkeypatch.setattr(replay.previous.profile.original.reference, 'hardware', lambda: dict(
        memory_available_bytes=64*1024**3, artifact_free_bytes=41*1024**3, physical_cpus=4))


def test_source_preflight_does_not_admit_or_execute(monkeypatch, tmp_path):
    prepare_main(monkeypatch, tmp_path)
    def forbidden(*a,**k): pytest.fail('runtime during source preflight')
    for name in ('completed_previous', 'create_output', 'replay'): monkeypatch.setattr(replay, name, forbidden)
    monkeypatch.setattr(replay.previous.profile, 'bound_profile_inputs', forbidden)
    monkeypatch.setattr(sys, 'argv', [replay.SOURCE, '--source-preflight-only'])
    replay.main()
    assert not replay.OUTPUT.exists()


def test_final_changed_input_preserves_failure_and_prevents_relaunch(monkeypatch, tmp_path):
    prepare_main(monkeypatch, tmp_path)
    monkeypatch.setattr(replay, 'previous_owner_ended', lambda: None)
    monkeypatch.setattr(replay, 'create_output', lambda p: p.mkdir())
    monkeypatch.setattr(replay, 'completed_previous', lambda *a:
        ({'report': {}, 'sensing_scope': {}}, {'input_admission': {'identity': 'original'}}, [], {}))
    admissions = iter([{'identity': 'original'}, {'identity': 'changed'}])
    monkeypatch.setattr(replay.previous.profile, 'bound_profile_inputs', lambda *a: next(admissions))
    def run(*a):
        (replay.OUTPUT/'comparison.jsonl').write_text('retained partial evidence\n')
        return {}
    monkeypatch.setattr(replay, 'replay', run)
    monkeypatch.setattr(sys, 'argv', [replay.SOURCE, '--fused-result-sha256', 'a'*64])
    with pytest.raises(ValueError, match='original raw/model inputs'): replay.main()
    assert json.loads((replay.OUTPUT/'failure.json').read_text())['automatic_retry'] is False
    assert not (replay.OUTPUT/'result.json').exists()
    assert (replay.OUTPUT/'comparison.jsonl').read_text() == 'retained partial evidence\n'
    with pytest.raises(ValueError, match='no retry or resume'): replay.main()

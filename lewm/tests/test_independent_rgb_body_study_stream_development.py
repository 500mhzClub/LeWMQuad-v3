"""Synthetic policy-only IO sets, role isolation and bounded materialization."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.tests.test_independent_pulse_evaluation_development import fixture
from lewm.tests.test_independent_pulse_context_development import policy
from lewm.tests.test_rgb_body_depth_separation_development import assert_same_tree
from scripts.independent_rgb_body_study_data_development import StudyData, AUDIT
import scripts.independent_rgb_body_study_stream_development as stream


@pytest.fixture(scope='module')
def study_data():
    inv, windows, targets, roles = fixture()
    for w in windows:
        for t in w['targets']:
            t.update(target_ns=w['decision_ns'] + t['offset_ns'] if t['offset_ns'] else None,
                command_prefix_executed=t['future_valid'], observation_available=t['future_valid'],
                future_observation_index=8 + t['offset_ns'] // 100_000_000 if t['future_valid'] else None)
    view = IndependentPulseEvaluation(inv, PulseTimedDataset(windows, targets, roles))
    receipts = {b: {'launch.json': 'a' * 64, AUDIT: 'b' * 64} for b in stream.BATCHES}
    batches = {}
    for b in stream.BATCHES:
        bindings = {}
        for w in windows:
            if w['condition'] in inv.episode_ids(b):
                bindings.update({w['condition'] + '/' + n: 'c' * 64 for n in stream.policy_artifacts(w, include_future=True)})
        batches[b] = dict(output_root=str(stream.output_root(b)), receipt=receipts[b],
            source_and_artifact_bindings_verified=True, artifact_sha256=bindings)
    return StudyData(view, {}, batches, receipts)


@pytest.fixture(scope='module')
def cached_policy():
    return lru_cache(maxsize=16)(policy)


@pytest.fixture
def io(monkeypatch, cached_policy):
    reads, checks = [], []
    monkeypatch.setattr(stream, 'load_route_observation', lambda directory, i: reads.append((directory.name, i)) or deepcopy(cached_policy(i)))
    monkeypatch.setattr(stream, 'verify_artifacts', lambda root, bindings: checks.append((root, deepcopy(bindings))))
    return reads, checks


def test_inference_does_not_even_inspect_future_metadata(study_data):
    w = deepcopy(study_data.evaluation.dataset.windows[0]); w['targets'] = object()
    names = stream.policy_artifacts(w, include_future=False)
    assert names == ['policy_observations.json', 'policy_histories.npz'] + [f'rgb_{i:04d}.png' for i in (5, 6, 7, 8)]


def test_training_adds_only_available_future_rgb_and_no_privileged_artifacts(study_data):
    w = deepcopy(study_data.evaluation.dataset.windows[0]); w['targets'][0]['future_valid'] = False
    names = stream.policy_artifacts(w, include_future=True)
    assert 'rgb_0013.png' not in names and 'rgb_0030.png' in names
    assert len(names) == 10
    assert not any(any(k in n for k in ('depth', 'native', 'physics', 'contact', 'command_tape')) for n in names)


@pytest.mark.parametrize('fault', ['missing_history', 'past_bool', 'past_future', 'future_bool', 'future_range', 'mode'])
def test_invalid_materialization_requests_rejected_before_reads(study_data, fault):
    w = deepcopy(study_data.evaluation.dataset.windows[0]); future = True
    if fault == 'missing_history': w['history_ready'] = False
    elif fault == 'past_bool': w['history_observation_indices'][0] = True
    elif fault == 'past_future': w['history_observation_indices'][0] = 34
    elif fault == 'future_bool': w['targets'][0]['future_observation_index'] = True
    elif fault == 'future_range': w['targets'][0]['future_observation_index'] = 34
    else: future = 1
    with pytest.raises(ValueError): stream.policy_artifacts(w, include_future=future)


def test_policy_reader_refuses_outside_bound_frame_set(monkeypatch):
    monkeypatch.setattr(stream, 'load_route_observation', lambda *a: pytest.fail('must reject before read'))
    reader = stream._PolicyReader(Path('/synthetic'), [5, 6, 7, 8])
    for i in (True, 4, 9, '5'):
        with pytest.raises(ValueError): reader.packet(i)


def test_training_inputs_equal_inference_without_future_or_label_access(study_data, io):
    s = stream.AuditedStudyStream(study_data); reads, checks = io
    train = s.training_batch([0]); train_reads = list(reads); reads.clear(); checks.clear()
    infer = s.inference_batch([0], role='train')
    assert_same_tree(train['inputs'], infer)
    assert [i for _, i in reads] == [5, 6, 7, 8]
    assert len(train_reads) == 9 and max(i for _, i in train_reads) == 30
    assert set(infer) == {'observation_history', 'known_action_blocks', 'known_action_valid'}
    assert len(checks) == 4 and all(not any('depth' in k for k in b) for _, b in checks)
    # Returning one batch cannot mutate the stream's private labels or history.
    train['targets']['motion'].fill_(999.); infer['observation_history']['rgb'].fill_(999.)
    again = s.training_batch([0]); assert not (again['targets']['motion'] == 999.).any()
    assert not (again['inputs']['observation_history']['rgb'] == 999.).any()


def test_repeated_training_draws_are_not_deduplicated(study_data, io):
    s = stream.AuditedStudyStream(study_data); b = s.training_batch([1, 1, 0])
    assert b['inputs']['observation_history']['rgb'].shape[0] == 3
    assert_same_tree({k: v[0] for k, v in b['targets'].items() if isinstance(v, torch.Tensor)},
        {k: v[1] for k, v in b['targets'].items() if isinstance(v, torch.Tensor)})
    assert b['targets']['target_offsets_ns'][0, 4] == 2_500_000_000
    assert b['targets']['target_offsets_ns'][2, 4] == 2_200_000_000


@pytest.mark.parametrize('fault', ['empty', 'oversize', 'bool', 'tuple', 'wrong_role', 'out_of_range'])
def test_bad_batch_latches_before_policy_reads(study_data, io, fault):
    s = stream.AuditedStudyStream(study_data); reads, _ = io
    ids = [0]
    if fault == 'empty': ids = []
    elif fault == 'oversize': ids *= 17
    elif fault == 'bool': ids = [True]
    elif fault == 'tuple': ids = (0,)
    elif fault == 'wrong_role': ids = s.evaluation.arrays('selection')['indices'][:1].tolist()
    else: ids = [len(s.evaluation.dataset)]
    with pytest.raises(ValueError): s.training_batch(ids)
    assert s.failed and not reads
    with pytest.raises(ValueError, match='latched'): s.training_batch([0])


@pytest.mark.parametrize('fault', ['receipt', 'unverified', 'root', 'missing_binding', 'changed_file'])
def test_stream_requires_bound_terminal_metadata_and_consumed_bytes(monkeypatch, study_data, io, fault):
    data = deepcopy(study_data); reads, _ = io
    if fault == 'receipt': data.batches['l00']['receipt'] = {}
    elif fault == 'unverified': data.batches['l00']['source_and_artifact_bindings_verified'] = False
    elif fault == 'root': data.batches['l00']['output_root'] = str(stream.output_root('l01'))
    elif fault == 'missing_binding': data.batches['l00']['artifact_sha256'] = {}
    else:
        def verify(root, bindings):
            if any('/' in k for k in bindings): raise ValueError('synthetic changed policy file')
        monkeypatch.setattr(stream, 'verify_artifacts', verify)
    with pytest.raises(ValueError):
        s = stream.AuditedStudyStream(data); s.training_batch([0])
    assert not reads


def test_post_read_byte_change_is_rejected(monkeypatch, study_data, io):
    counts = 0
    def verify(root, bindings):
        nonlocal counts
        if any('/' in k for k in bindings):
            counts += 1
            if counts == 2: raise ValueError('changed during materialization')
    monkeypatch.setattr(stream, 'verify_artifacts', verify)
    s = stream.AuditedStudyStream(study_data)
    with pytest.raises(ValueError, match='during'): s.inference_batch([0], role='train')
    assert s.failed

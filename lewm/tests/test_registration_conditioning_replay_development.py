"""Synthetic whole-stream ordering, exact replay and evaluator-only native data."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.tests.test_rgbd_correspondence_motion_development import texture, packets
import scripts.navigation_artifact_root_development as authority
import scripts.replay_go2_registration_conditioning_v1 as replay


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, 'BASE', tmp_path)
    monkeypatch.setattr(replay, 'BASE', tmp_path)
    monkeypatch.setattr(replay, 'OUTPUT', tmp_path / 'go2_synthetic_registration_attempt_001')
    monkeypatch.setattr(replay.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(replay, 'COHORTS', {'inner': dict(root=tmp_path / 'input', arms=('original',))})
    monkeypatch.setattr(replay, 'TRIALS', ('synthetic',))
    samples = list(packets([texture()] * 3)); model = MultiReferenceVisualLedMotion(); original = []
    for p, d, f, now in samples:
        original.append(dict(evidence=replay.serial(model.observe(p, d, f, now_ns=now))))
    class Reader:
        def __init__(self, path): self.frames = list(range(len(samples)))
        def packet(self, i): return deepcopy(samples[i])
    monkeypatch.setattr(replay, 'IntentReturnRGBDReplay', Reader)
    read_json = replay.read_json; reads = []
    def read(directory, name):
        if name == 'servo_decisions.json': return deepcopy(original)
        return read_json(directory, name)
    monkeypatch.setattr(replay, 'read_json', read)
    poses = np.zeros((900, 7)); poses[:, 6] = 1.
    clock = np.arange(1, 901) * .002
    for i, sample in enumerate(samples): clock[749 + 50 * i] = sample[-1] / 1e9
    def native(directory, name):
        reads.append(name)
        assert (replay.OUTPUT / 'all_sensor_estimates_complete.json').is_file()
        return dict(base_pose_world=poses.copy(), timestamp_s=clock.copy())
    monkeypatch.setattr(replay, 'read_npz', native)
    monkeypatch.setattr(replay, 'preflight', lambda: {'synthetic': True})
    monkeypatch.setattr(replay, 'verify_launch', lambda launch: None)
    return original, samples, reads


def test_full_stream_is_persisted_before_any_native_coordinates_are_parsed(synthetic):
    original, samples, reads = synthetic
    replay.run()
    result = replay.read_json(replay.OUTPUT, 'result.json')
    assert result['status'] == 'WHOLE_STREAM_REGISTRATION_WITNESS_COMPLETE' and reads == ['physics_trace.npz']
    report = result['reports']['inner_synthetic']['original']
    assert report['frames'] == report['available'] == report['exact_recorded_decisions'] == 3
    assert report['candidate_calls'] == 2
    evaluation = result['evaluations']['inner_synthetic_original']
    # The unchanged float32 image/flow path has small nonzero drift even on
    # repeated synthetic texture. Verify its actual error, not fictitious zero.
    expected_error = max(float(np.linalg.norm(r['evidence']['current_pose']['position_initial_body_m'])) for r in original)
    assert evaluation['maximum_available_position_error_m'] == expected_error
    assert evaluation['candidate_groups']['candidate_qualified']['calls'] == 2
    assert evaluation['instrumented_pair_components']['matching']['calls'] == 2
    assert not result['gates_modified'] and not result['physical_recovery'] and not result['goal_achieved']
    authority.verify_artifacts(replay.OUTPUT, result['output_sha256'])


def test_original_pose_mismatch_keeps_partial_sensor_evidence_and_never_loads_native(synthetic):
    original, samples, reads = synthetic
    original[1]['evidence']['current_pose']['position_initial_body_m'][0] = 999.
    with pytest.raises(ValueError, match='exact original'): replay.run()
    failure = replay.read_json(replay.OUTPUT, 'failure.json')
    assert failure['active'] == 'inner_synthetic' and not reads and not failure['retry_performed']
    assert not (replay.OUTPUT / 'result.json').exists()
    path = replay.OUTPUT / 'inner_synthetic_original_estimates.jsonl'
    assert len(path.read_text().splitlines()) == 1


def test_native_clock_error_cannot_publish_a_completed_evaluation(monkeypatch, synthetic):
    native = replay.read_npz
    def broken(*a):
        raw = native(*a); raw['timestamp_s'][799] += .002; return raw
    monkeypatch.setattr(replay, 'read_npz', broken)
    with pytest.raises(ValueError, match='clock'): replay.run()
    assert (replay.OUTPUT / 'all_sensor_estimates_complete.json').exists()
    assert not (replay.OUTPUT / 'result.json').exists()


def test_evaluator_refuses_native_access_before_bound_sensor_phase(synthetic):
    replay.create_output(replay.OUTPUT); store = replay.Store(replay.OUTPUT)
    with pytest.raises(ValueError, match='sensor phase'):
        replay.evaluate_trial('inner', 'synthetic', 'original', {}, store)
    assert not synthetic[2]


@pytest.mark.parametrize('fault', ['source', 'budget', 'trace'])
def test_other_failures_preserve_evidence_and_do_not_restart(monkeypatch, synthetic, fault):
    if fault == 'source':
        monkeypatch.setattr(replay, 'verify_launch', lambda launch: (_ for _ in ()).throw(ValueError('synthetic source mismatch')))
    elif fault == 'budget': monkeypatch.setattr(replay, 'BUDGET', 20)
    else:
        class Broken:
            def __enter__(self): raise RuntimeError('synthetic trace fault')
            def __exit__(self, *a): return False
        monkeypatch.setattr(replay, 'TracePairs', Broken)
    with pytest.raises((ValueError, RuntimeError)): replay.run()
    assert (replay.OUTPUT / 'failure.json').exists() and not synthetic[2]
    assert not (replay.OUTPUT / 'result.json').exists()
    with pytest.raises(ValueError, match='exclusive'): replay.run()


@pytest.mark.parametrize('name', ['sealed_test.json', '../x.json', 'nested/x.json', 'payload.pt'])
def test_store_requires_narrow_ordinary_explicit_names(synthetic, name):
    replay.create_output(replay.OUTPUT); store = replay.Store(replay.OUTPUT)
    with pytest.raises(ValueError): store.save(name, {})
    assert not store.hashes and store.used == 0


def test_failed_native_parse_cannot_modify_sensor_only_estimates(monkeypatch, synthetic):
    def fail(*args): raise ValueError('synthetic native unavailable')
    monkeypatch.setattr(replay, 'read_npz', fail)
    with pytest.raises(ValueError, match='native unavailable'): replay.run()
    phase = replay.read_json(replay.OUTPUT, 'all_sensor_estimates_complete.json')
    authority.verify_artifacts(replay.OUTPUT, phase['estimates_sha256'])
    assert not phase['native_coordinates_parsed'] and not (replay.OUTPUT / 'result.json').exists()

import numpy as np
import pytest
from scripts import run_go2_augmented_family_switch_goal_probe_v1 as runner


def test_comparison_keeps_existing_execution_and_audit_and_fixed_models():
    from scripts.overlap_retention_goal_episode_development import collect
    from scripts.overlap_retention_goal_audit_development import audit
    assert runner.collect is collect and runner.audit is audit
    assert {(c[1], c[2], c[3], c[4]) for c in runner.CASES} == {
        ('family_episode_039', 'full', 'jepa', 'seed_2026091001_full_jepa'),
        ('family_episode_039', 'full', 'direct', 'seed_2026091001_full_direct')}


def test_prefix_detects_warmup_changes_but_excludes_controlled_future(tmp_path, monkeypatch):
    packets = [{'rgb': np.zeros((2, 2, 3), np.uint8), 'clock': i} for i in range(5)]
    class Reader:
        frames = packets
        def __init__(self, directory): pass
        def packet(self, i):
            if i >= 4: raise AssertionError('controlled future must not be read')
            return packets[i]
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', Reader)
    stamps = np.arange(950, dtype=np.float64)
    states = np.zeros((950, 3), dtype=np.float64)
    def save():
        np.savez(tmp_path/'physics_trace.npz', timestamp_s=stamps, state=states)
    save(); original = runner.causal_prefix(tmp_path)
    states[900:] = 3; save()
    assert runner.causal_prefix(tmp_path) == original
    states[899, 0] = 1; save()
    assert runner.causal_prefix(tmp_path) != original
    states[899, 0] = 0; save()
    packets[3]['rgb'][0, 0, 0] = 1
    assert runner.causal_prefix(tmp_path) != original


def test_incomplete_native_prefix_cannot_pass(tmp_path, monkeypatch):
    class Reader:
        frames = [None]*4
        def __init__(self, directory): pass
        def packet(self, i): raise AssertionError('incomplete native prefix')
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', Reader)
    np.savez(tmp_path/'physics_trace.npz', timestamp_s=np.arange(899))
    with pytest.raises(ValueError, match='native warmup'):
        runner.causal_prefix(tmp_path)

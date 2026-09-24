"""Full replay code identity and metadata-only comparison scope."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import pytest
from scripts import replay_go2_frozen_footprint_anchored_prefix_v1 as candidate
from lewm.frozen_footprint_anchored_controller_development import CONTROLLER, FLAG


def test_original_replay_code_and_state_comparison_remain_exact():
    assert hashlib.sha256(Path('scripts/replay_go2_receipt_copied_anchored_prefix_v1.py').read_bytes()).hexdigest() == (
        '0fdc4ac3493ce8c4744cab4a7270c3ff8b41163fe9575988a2111fbbe5bb1a7d')
    original = candidate.preceding.replay; before = original.__globals__.copy()
    new = candidate.isolated_replay()
    overrides = dict(ReceiptCopiedAnchoredController=candidate.FrozenFootprintAnchoredController,
        normalize_candidate=candidate.normalize_candidate, OUTPUT=candidate.OUTPUT, print=candidate.progress)
    assert new.__code__ is original.__code__
    assert new.__defaults__ is original.__defaults__
    assert new.__kwdefaults__ is original.__kwdefaults__
    assert new.__closure__ is original.__closure__ is None
    assert new.__globals__.keys() == before.keys() | overrides.keys()
    for name, value in new.__globals__.items():
        assert value is (overrides[name] if name in overrides else before[name])
    assert new.__globals__['state_tree'] is candidate.state_tree
    assert all(original.__globals__[name] is value for name, value in before.items())
    assert candidate.FRAMES == 405 and candidate.STATE_FRAMES == (3, 12, 395, 404)


def test_normalization_cannot_hide_command_or_evidence_changes():
    d = dict(controller=CONTROLLER, **{FLAG: True}, requested_command=[0., 0., .45],
        new_selection={'witness': {'controller': CONTROLLER, FLAG: True}, 'utility_m': -.2})
    before = deepcopy(d); expected = deepcopy(d); expected.pop(FLAG)
    expected['controller'] = 'residual_anchored_continuation_controller_v1'
    assert candidate.normalize_candidate(d) == expected and d == before
    for update in ({'controller': 'other'}, {FLAG: False}, {FLAG: 1}):
        with pytest.raises(ValueError): candidate.normalize_candidate(d | update)


def test_incomplete_predecessor_is_rejected(monkeypatch):
    calls = []
    monkeypatch.setattr(candidate, 'verify_artifacts', lambda root, ids: calls.append((root, ids)))
    monkeypatch.setattr(candidate, 'read_json', lambda *args: dict(status='INCOMPLETE'))
    with pytest.raises(ValueError, match='completed exact'):
        candidate.preceding_inputs()
    assert calls == [(candidate.completed_shared.OUTPUT, {'result.json': candidate.PRECEDING_SHA})]


def test_source_preflight_does_not_load_models_or_create_output(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs): raise AssertionError('runtime work during source preflight')
    monkeypatch.setattr(candidate, 'OUTPUT', tmp_path/'uncreated')
    monkeypatch.setattr(candidate, 'validate_root', lambda *args, **kwargs: None)
    monkeypatch.setattr(candidate, 'preceding_inputs', lambda: {'source_sha256': {}})
    monkeypatch.setattr(candidate, 'discover_sources', lambda *args: {})
    monkeypatch.setattr(candidate, 'verify', lambda *args: None)
    monkeypatch.setattr(candidate.profile.reference, 'hardware', lambda: {})
    monkeypatch.setattr(candidate.profile, 'resources_for', lambda *args: None)
    monkeypatch.setattr(candidate.profile.reference, 'admit_worker', forbidden)
    monkeypatch.setattr(candidate, 'create_output', forbidden)
    monkeypatch.setattr(candidate, 'isolated_replay', forbidden)
    monkeypatch.setattr(sys, 'argv', [candidate.SOURCE, '--source-preflight-only'])
    candidate.main()
    assert 'FROZEN_FOOTPRINT_ANCHORED_SOURCE_PREFLIGHT_PASS 0' in capsys.readouterr().out
    assert not candidate.OUTPUT.exists()


def test_progress_only_changes_label(capsys):
    candidate.progress('RECEIPT_COPIED_ANCHORED_RAW_FRAME', 50, flush=True)
    assert capsys.readouterr().out == 'FROZEN_FOOTPRINT_ANCHORED_RAW_FRAME 50\n'

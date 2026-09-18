"""End-to-end synthetic report reconstruction and execution admission guards."""
from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType

import pytest

from scripts import run_go2_single_pass_body_projected_late_history_v1 as run
from lewm.tests import test_body_projected_tiled_controller_completion_development as original
from lewm.tests import test_body_projected_tiled_replay_development as baseline
from lewm.tests import test_single_pass_body_projected_replay_development as candidate


for name in ('test_complete_population_rebuilds_deadline_counts_and_all_windows',
        'test_rewritten_row_or_nonboolean_claim_rejected',
        'test_invalid_timing_or_execution_order_rejected', 'test_missing_population_rejected'):
    function = getattr(original, name)
    clone = FunctionType(function.__code__, function.__globals__ | dict(check=run),
        name, function.__defaults__, function.__closure__)
    clone.__dict__.update(function.__dict__)
    clone.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = clone


def test_full_synthetic_replay_reconstructs_from_complete_predecessor(monkeypatch, tmp_path):
    old_root = tmp_path/'baseline'; old_root.mkdir()
    with monkeypatch.context() as patch:
        rows, prior, *_ = baseline.fixture(patch, old_root)
        old_report = baseline.run.replay(rows, prior)
    new_root = tmp_path/'candidate'; new_root.mkdir()
    rows, _, *_ = candidate.fixture(monkeypatch, new_root)
    report = run.harness.replay(rows, old_report)
    written = [json.loads(line) for line in (new_root/'comparison.jsonl').read_text().splitlines()]
    admission = (None, {'report': old_report}, None, rows)
    timing = run.validate_report(json.loads(json.dumps(report)), written, admission)
    assert timing['all_navigation']['observations'] == 1425
    for key, value in (('navigation_qualified', True), ('model_state_unchanged', False),
            ('original_packed_insertion_unchanged', False), ('observed_state_checks', []),
            ('normalized_state_type_paths', [])):
        changed = deepcopy(report); changed[key] = value
        with pytest.raises(ValueError, match='complete report'):
            run.validate_report(changed, written, admission)


def _capture_probe():
    calls.append((OUTPUT, capture_verification))
    return 'captured'


def test_private_capture_retains_original_code_and_does_not_mutate_globals(monkeypatch):
    original_capture = run.original_admission.capture_verification
    before = original_capture.__globals__.copy()
    calls = []
    monkeypatch.setattr(run.original_admission, 'admit_completed', _capture_probe)
    monkeypatch.setattr(run.original_admission, 'calls', calls, raising=False)
    assert run.private_admission() == 'captured'
    assert calls[0][0] == run.OUTPUT
    assert calls[0][1].__code__ is original_capture.__code__
    assert original_capture.__globals__['OUTPUT'] is before['OUTPUT']
    assert original_capture.__globals__['capture_verification'] is original_capture


def test_live_profiler_owner_keeps_replay_slot_unavailable(monkeypatch):
    monkeypatch.setattr(run, 'snapshot', lambda: {'profile': {'owners': {
        'child': {'state': 'live'}, 'parent': {'state': 'ended'}}}})
    with pytest.raises(ValueError, match='all four'): run.slot_available()


def test_live_original_replay_owner_rejects_completion_before_input_admission(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUTPUT', tmp_path)
    monkeypatch.setattr(run, 'COMPLETION', tmp_path/'completion.json')
    monkeypatch.setattr(run, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(run, 'owner_live', lambda owner: True)
    monkeypatch.setattr(run, 'private_admission', lambda: pytest.fail('must reject live owner first'))
    (tmp_path/'result.json').write_text(json.dumps({'artifact_sha256': {
        'launch.json': 'a'*64, 'comparison.jsonl': 'b'*64}}))
    (tmp_path/'launch.json').write_text(json.dumps({'owner': {},
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
    with pytest.raises(ValueError, match='owner must be ended'): run.verify_completion('c'*64)


def test_preserved_failure_prevents_completion(monkeypatch, tmp_path):
    monkeypatch.setattr(run, 'OUTPUT', tmp_path)
    monkeypatch.setattr(run, 'COMPLETION', tmp_path/'completion.json')
    (tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='failed replay'): run.verify_completion('c'*64)

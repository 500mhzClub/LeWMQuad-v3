import hashlib

import pytest

from scripts import read_go2_independent_tracking_quaternion_diagnostic_v1 as diagnostic


def test_definition_identity_includes_frozen_newline():
    assert diagnostic.definition_identity({'b': 2, 'a': 1}) == hashlib.sha256(b'{"a":1,"b":2}\n').hexdigest()


def fixtures(monkeypatch):
    definition = {'source_sha256': {}}
    identity = diagnostic.definition_identity(definition)
    terminal = dict(status='SCOPED_COMMAND_FAILED', child_handle_terminal=True,
        systemd_run_returncode=1, log_complete=True, log_omitted_bytes=0, definition_sha256=identity)
    rows = {
        'failure.json': dict(status='TERMINAL_TRACKING_COHORT_PHASE_FAILURE',
            stage='complete_stress_native_audit_or_admission', reason="ValueError('unit native quaternions required')"),
        'terminal.json': terminal,
        'launch.json': dict(definition=definition, definition_sha256=identity),
    }
    monkeypatch.setattr(diagnostic, 'DEFINITION', identity)
    monkeypatch.setattr(diagnostic, 'validate_root', lambda *a: None)
    monkeypatch.setattr(diagnostic, 'verify_artifacts', lambda *a: None)
    monkeypatch.setattr(diagnostic, 'source_bindings', lambda *a: None)
    monkeypatch.setattr(diagnostic.base, 'read', lambda root, name: rows[name])
    monkeypatch.setattr(diagnostic.np, 'load', lambda *a, **k: pytest.fail('native arrays read before sensor admission'))
    return rows


def test_all_sensor_gate_precedes_native_access(monkeypatch):
    fixtures(monkeypatch)
    def denied(*args): raise RuntimeError('sensor admission denied')
    monkeypatch.setattr(diagnostic.stress, 'admit_complete_sensor_phase', denied)
    with pytest.raises(RuntimeError, match='sensor admission denied'): diagnostic.diagnose()


def test_incomplete_terminal_rejected_before_sensor_or_native_access(monkeypatch):
    rows = fixtures(monkeypatch)
    rows['terminal.json']['log_complete'] = False
    monkeypatch.setattr(diagnostic.stress, 'admit_complete_sensor_phase',
                        lambda *a: pytest.fail('invalid terminal admitted'))
    with pytest.raises(ValueError, match='retained terminal'): diagnostic.diagnose()

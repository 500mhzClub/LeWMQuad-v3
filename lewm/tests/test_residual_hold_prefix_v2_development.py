"""Output-only replacement preserves the original causal replay and failures."""
import ast
from pathlib import Path
import pytest
from scripts import replay_go2_residual_hold_prefix_v1 as original
from scripts import replay_go2_residual_hold_prefix_v2 as replacement


def test_scientific_replay_is_identical_with_only_larger_output_envelope():
    def function(path, name):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(function(original.__file__, 'replay')) == ast.dump(function(replacement.__file__, 'replay'))
    for name in ('ResidualHoldFeasibilityController', 'compare_step', 'load_assigned', 'INPUT', 'CASE', 'MAX_FRAMES'):
        assert getattr(original, name) == getattr(replacement, name)
    assert original.MAX_OUTPUT_BYTES == 256*1024**2
    assert replacement.MAX_OUTPUT_BYTES == 2*1024**3
    assert original.OUTPUT != replacement.OUTPUT and original.PROTOCOL != replacement.PROTOCOL


@pytest.mark.parametrize('boundary', ['command', 'old_terminal', 'limit', 'mutated_input', 'model_change', 'truncated'])
def test_existing_replay_failure_and_stop_contracts_apply_to_replacement(monkeypatch, tmp_path, boundary):
    from lewm.tests import test_residual_hold_prefix_development as contracts
    monkeypatch.setattr(contracts, 'runner', replacement)
    contracts.test_replay_never_consumes_next_packet_or_decision(monkeypatch, tmp_path, boundary)


@pytest.mark.parametrize('fault', [None, 'hash', 'source', 'failure', 'short', 'extra', 'changed', 'state'])
def test_exact_failed_predecessor_is_required_and_never_replaced(monkeypatch, fault):
    calls = []
    def verify_artifacts(root, bindings):
        assert root == replacement.FAILED and bindings == replacement.FAILED_BINDINGS
        calls.append('artifacts')
        if fault == 'hash': raise ValueError('artifact mismatch')
    def verify_source(launch):
        calls.append('source')
        if fault == 'source': raise ValueError('source mismatch')
    monkeypatch.setattr(replacement, 'verify_artifacts', verify_artifacts)
    monkeypatch.setattr(original, 'verify_inputs', verify_source)
    launch = {'source_sha256':{'synthetic':'a'*64}}
    failure = dict(status='TERMINAL_RESIDUAL_HOLD_PREFIX_FAILURE',
        reason="ValueError('compressed replay output headroom exceeded')")
    if fault == 'failure': failure['reason'] = 'different failure'
    monkeypatch.setattr(replacement, 'read_json', lambda root, name:launch if name == 'launch.json' else failure)
    def rows(root):
        assert root == replacement.FAILED
        for i in range(1184 if fault == 'short' else 1186 if fault == 'extra' else 1185):
            yield dict(tick=i, decision={'tick':i}, comparison=dict(
                requested_command_changed=fault == 'changed' and i == 10,
                hold_reconsideration_changed_action=False, complete_original_selection_preserved=True,
                unchanged_observed_mission_and_residual_state_exact=not (fault == 'state' and i == 10)))
    monkeypatch.setattr(replacement, 'read_rows', rows)
    if fault:
        with pytest.raises(ValueError): replacement.verify_failed_predecessor()
    else:
        assert replacement.verify_failed_predecessor() == launch['source_sha256']
        assert calls == ['artifacts', 'source']

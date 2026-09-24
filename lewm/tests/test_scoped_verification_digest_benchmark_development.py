"""The paired benchmark must execute both verifiers and retain completed stages."""
from copy import deepcopy
import pytest
from scripts import benchmark_go2_scoped_verification_digest_v1 as runner


@pytest.mark.parametrize('fault', [None, 'scoped_failure', 'original_failure', 'input_mutation', 'wrong_return'])
def test_both_stages_are_executed_on_unchanged_context(monkeypatch, fault):
    context = dict(target_bindings={'result.json':'a'*64}, target_launch={'fixture':True})
    before = deepcopy(context); calls = []; saved = []
    def scoped(function, digest, supplied):
        assert function is runner.verify_target and digest is runner.digest and supplied is context
        calls.append('scoped')
        if fault == 'scoped_failure': raise ValueError('scoped failure')
        if fault == 'input_mutation': supplied['changed'] = True
        return ('wrong' if fault == 'wrong_return' else None), {'fixture':True}
    def original(supplied):
        assert supplied is context; calls.append('original')
        if fault == 'original_failure': raise ValueError('original failure')
    monkeypatch.setattr(runner, 'verify_target', original)
    monkeypatch.setattr(runner, 'verify_with_scoped_digests', scoped)
    if fault is None:
        result = runner.paired_verification(context, saved.append)
        assert calls == ['scoped', 'original'] and [r['mode'] for r in saved] == calls
        assert context == before and result['both_verifiers_completed']
        assert not result['controlled_speedup_established'] and not result['implementation_adopted_by_existing_launchers']
    else:
        with pytest.raises(ValueError): runner.paired_verification(context, saved.append)
        if fault == 'original_failure': assert len(saved) == 1 and saved[0]['mode'] == 'scoped'
        else: assert not saved and calls == ['scoped']

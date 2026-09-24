"""Keep actual checker gates while capturing only its output and fixed CLI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

import pytest
from scripts import body_projection_external_profile_admission_development as admission


@pytest.mark.parametrize('mode', ['once', 'none', 'twice', 'wrong_path', 'early_digest',
    'missing_execution', 'wrong_order', 'extra_argument'])
def test_private_checker_capture_preserves_originals(monkeypatch, tmp_path, mode):
    def checker():
        parser = argparse.ArgumentParser()
        parser.add_argument('--execution-sha256' if MODE == 'wrong_order' else '--result-sha256', required=True)
        if MODE != 'missing_execution': parser.add_argument('--execution-sha256', required=True)
        if MODE == 'extra_argument': parser.add_argument('--extra', required=True)
        args = parser.parse_args()
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive')
        if MODE == 'early_digest': digest(OUTPUT)
        if MODE == 'none': return
        payload = dict(result=args.result_sha256, execution=args.execution_sha256, failure_retained=True)
        write_json(OUTPUT if MODE != 'wrong_path' else 'wrong', payload)
        if MODE == 'twice': write_json(OUTPUT, payload)
        EXPECTED_DIGEST(digest(OUTPUT))
        print('COMPLETE', digest(OUTPUT))
    original = tmp_path/'original.json'; original.write_text('original')
    payload = dict(result=admission.RESULT_SHA, execution=admission.EXECUTION_SHA, failure_retained=True)
    expected = hashlib.sha256((json.dumps(payload, indent=2, allow_nan=False)+'\n').encode()).hexdigest()
    def check_digest(value): assert value == expected
    def forbidden(*a, **kw): pytest.fail('must not use original output or CLI')
    function = FunctionType(checker.__code__, dict(MODE=mode, OUTPUT=original,
        argparse=forbidden, write_json=forbidden, digest=forbidden, print=forbidden,
        EXPECTED_DIGEST=check_digest))
    monkeypatch.setattr(admission.completed, 'main', function)
    monkeypatch.setattr(admission, 'OUTPUT', tmp_path/'absent')
    before = dict(function.__globals__); argv = list(sys.argv)
    if mode == 'once': assert admission.capture_verification() == payload
    else:
        with pytest.raises(ValueError): admission.capture_verification()
    assert function.__globals__ == before and sys.argv == argv
    assert original.read_text() == 'original' and not admission.OUTPUT.exists()


def test_actual_checker_rejects_live_owner_before_artifact_read(monkeypatch, tmp_path):
    execution = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), owner={})
    (tmp_path/'execution.json').write_text(json.dumps(execution))
    monkeypatch.setattr(admission.completed, 'ROOT', tmp_path)
    monkeypatch.setattr(admission.completed, 'EXECUTION', 'execution.json')
    verified = []
    monkeypatch.setattr(admission.completed, 'verify', lambda value:verified.append(value))
    monkeypatch.setattr(admission.completed.run, 'owner_live', lambda owner:True)
    monkeypatch.setattr(admission.completed, 'verify_artifacts', lambda *a:pytest.fail('owner gate must precede artifact access'))
    monkeypatch.setattr(admission, 'OUTPUT', tmp_path/'absent')
    with pytest.raises(ValueError, match='owner must be ended'): admission.capture_verification()
    assert verified == [{'execution.json':admission.EXECUTION_SHA}]
    assert not admission.OUTPUT.exists()


@pytest.mark.parametrize('field', ['sensing_scope', 'source_sha256', 'observed_state_checks', 'timing_windows'])
def test_changed_full_witness_rejects_before_returned_artifact_reads(monkeypatch, field):
    witness = dict(utc='original', sensing_scope={'failure':True}, source_sha256={'original':'a'*64},
        observed_state_checks=[{'frame':1173}], timing_windows={'all_navigation':{'count':1425}})
    actual = deepcopy(witness); actual['utc'] = 'new'; actual[field] = {}
    monkeypatch.setattr(admission, 'reference_witness', lambda:witness)
    monkeypatch.setattr(admission, 'capture_verification', lambda:actual)
    monkeypatch.setattr(admission, 'read_json', lambda *a:pytest.fail('changed witness rejected before reads'))
    with pytest.raises(ValueError, match='including failures'): admission.admit_completed()


def test_admission_reauthenticates_returned_rows_and_preserves_negative_scope(monkeypatch, tmp_path):
    witness = dict(utc='old', sensing_scope={'failure':True}, artifact_sha256={'comparison.jsonl':'a'*64})
    monkeypatch.setattr(admission, 'reference_witness', lambda:witness)
    monkeypatch.setattr(admission, 'capture_verification', lambda:witness | {'utc':'new'})
    monkeypatch.setattr(admission.completed.run, 'OUTPUT', tmp_path)
    (tmp_path/'comparison.jsonl').write_text('{"frame": 0}\n')
    monkeypatch.setattr(admission, 'read_json', lambda root, name:{'name':name})
    calls = []
    monkeypatch.setattr(admission.completed, 'verify_artifacts', lambda root, ids:calls.append((root, ids)))
    result = admission.admit_completed()
    assert result == (witness, {'name':'result.json'}, {'name':'launch.json'}, [{'frame':0}])
    assert calls == [(tmp_path, witness['artifact_sha256'] | {'result.json':admission.RESULT_SHA,
        'launch.json':admission.LAUNCH_SHA})]

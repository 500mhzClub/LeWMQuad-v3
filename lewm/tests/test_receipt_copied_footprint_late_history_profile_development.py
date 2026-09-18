"""Exact profiler composition and completed-row identity checks."""
from copy import deepcopy
import sys

import pytest

from scripts import profile_go2_receipt_copied_footprint_late_history_v1 as profile


def rows():
    return [dict(frame=i, public_input_sha256='a'*64, original_decision_sha256='b'*64,
        candidate_decision_sha256='c'*64, candidate_normalized_decision_exact=True,
        complete_original_decision_reconstructed=True, public_input_arrays_unchanged=True) for i in range(1428)]


def test_only_declared_globals_change_in_actual_original_profiling_body():
    original = profile.original.replay; before = dict(original.__globals__)
    clone = profile.isolated_replay()
    assert clone.__code__ is original.__code__ and clone.__closure__ is original.__closure__
    assert clone.__defaults__ is original.__defaults__ and clone.__kwdefaults__ is original.__kwdefaults__
    assert clone.__globals__ is not original.__globals__
    expected = dict(OUTPUT=profile.OUTPUT, FrozenFootprintAnchoredController=profile.paired.ReceiptCopiedFootprintController,
        normalize_candidate=profile.paired.normalize_candidate, print=profile.progress)
    for key, value in before.items():
        assert original.__globals__[key] is value
        assert clone.__globals__[key] is expected.get(key, value)
    assert clone.__globals__['WINDOWS'] == {'early_navigation':(3, 12), 'repeated_hold':(395, 404), 'late_navigation':(1418, 1427)}


def test_complete_profile_rows_match_completed_receipt_copy_run():
    actual = rows(); profile.compare_profile_rows(actual, deepcopy(actual))


@pytest.mark.parametrize('field', ['public_input_sha256', 'original_decision_sha256', 'candidate_decision_sha256',
    'candidate_normalized_decision_exact', 'complete_original_decision_reconstructed', 'public_input_arrays_unchanged',
    'frame', 'population'])
def test_profile_cannot_change_inputs_decisions_flags_or_population(field):
    actual = rows(); prior = deepcopy(actual)
    if field == 'population': actual.pop()
    elif field == 'frame': actual[1]['frame'] = True
    elif field.endswith('sha256'): actual[-1][field] = '0'*64
    else: actual[-1][field] = 1
    with pytest.raises(ValueError): profile.compare_profile_rows(actual, prior)


def test_existing_profile_attempt_rejected_before_source_work(monkeypatch, tmp_path):
    monkeypatch.setattr(profile, 'OUTPUT', tmp_path)
    monkeypatch.setattr(profile, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(profile, 'prepared_sources', lambda:pytest.fail('exclusive attempt must reject first'))
    monkeypatch.setattr(profile.original.cv2.ocl, 'useOpenCL', lambda:False)
    monkeypatch.setattr(sys, 'argv', ['profile'])
    for key, value in dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled').items(): monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='exclusive'): profile.main()


def test_source_preflight_does_not_load_model_admit_raw_inputs_or_create_output(monkeypatch, tmp_path, capsys):
    output = tmp_path/'absent'
    monkeypatch.setattr(profile, 'OUTPUT', output)
    monkeypatch.setattr(profile, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(profile, 'prepared_sources', lambda:{'synthetic':'a'*64})
    monkeypatch.setattr(profile, 'admit_completed', lambda *a:pytest.fail('source-only preflight'))
    monkeypatch.setattr(profile, 'create_output', lambda *a:pytest.fail('source-only preflight'))
    monkeypatch.setattr(profile.original.cv2.ocl, 'useOpenCL', lambda:False)
    monkeypatch.setattr(profile.original.reference, 'hardware', lambda:dict(memory_available_bytes=48*1024**3,
        artifact_free_bytes=41*1024**3, physical_cpus=4))
    monkeypatch.setattr(sys, 'argv', ['profile', '--source-preflight-only'])
    for key, value in dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled').items(): monkeypatch.setenv(key, value)
    profile.main()
    assert 'RECEIPT_COPIED_FOOTPRINT_PROFILE_SOURCE_PREFLIGHT' in capsys.readouterr().out and not output.exists()


@pytest.mark.parametrize('mode',['once','none','twice','wrong_path'])
def test_frozen_verifier_is_reexecuted_without_writing_original_output(monkeypatch,mode):
    from types import FunctionType
    def emit():
        if MODE=='none':return
        write_json(OUTPUT if MODE!='wrong_path' else 'wrong',{'verified':True})
        if MODE=='twice':write_json(OUTPUT,{'verified':True})
    forbidden=lambda *a:pytest.fail('original output must not be written')
    original=FunctionType(emit.__code__,dict(MODE=mode,OUTPUT=profile.completed.OUTPUT,write_json=forbidden))
    monkeypatch.setattr(profile.completed,'main',original)
    before=dict(original.__globals__)
    if mode=='once':assert profile.capture_verification()=={'verified':True}
    else:
        with pytest.raises(ValueError):profile.capture_verification()
    assert original.__globals__==before and original.__globals__['write_json'] is forbidden


def test_entire_prior_verification_is_required_before_any_replay_artifact(monkeypatch,tmp_path):
    import json
    witness=dict(source_sha256={'fixed':'sha'},utc='old',result_sha256=profile.COPIED_SHA,
        original_owner_ended=True,observed_state_checks=[{'state':'original'}])
    path=tmp_path/'witness.json';path.write_text(json.dumps(witness))
    monkeypatch.setattr(profile,'ROOT',tmp_path);monkeypatch.setattr(profile,'VERIFICATION','witness.json')
    monkeypatch.setattr(profile,'verify',lambda *a:None)
    monkeypatch.setattr(profile,'capture_verification',lambda:witness|{'observed_state_checks':[]})
    monkeypatch.setattr(profile,'read_json',lambda *a:pytest.fail('changed prior evidence must reject before replay artifacts'))
    with pytest.raises(ValueError,match='must reconstruct'):profile.admit_completed({'fixed':'sha'})

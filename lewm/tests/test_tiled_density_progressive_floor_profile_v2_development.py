"""Check profiling composition, full history identity and read-only admission."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import FunctionType

import pytest

from scripts import profile_go2_tiled_density_progressive_floor_late_history_v2 as profile


def rows():
    return [dict(frame=i, public_input_sha256='a'*64, original_decision_sha256='b'*64,
        candidate_decision_sha256='c'*64, candidate_normalized_decision_exact=True,
        complete_original_decision_reconstructed=True, public_input_arrays_unchanged=True) for i in range(1428)]


def test_original_profiler_body_changes_only_controller_output_normalizer_and_progress():
    original = profile.original.replay; before = dict(original.__globals__)
    clone = profile.isolated_replay()
    assert clone.__code__ is original.__code__ and clone.__closure__ is original.__closure__
    assert clone.__defaults__ is original.__defaults__ and clone.__kwdefaults__ is original.__kwdefaults__
    assert clone.__globals__ is not original.__globals__
    expected = dict(OUTPUT=profile.OUTPUT, FrozenFootprintAnchoredController=profile.TiledDensityProgressiveFloorController,
        normalize_candidate=profile.paired.harness.normalize_candidate, print=profile.progress)
    for key, value in before.items():
        assert original.__globals__[key] is value
        assert clone.__globals__[key] is expected.get(key, value)
    assert clone.__globals__['WINDOWS'] == {'early_navigation':(3, 12), 'repeated_hold':(395, 404), 'late_navigation':(1418, 1427)}


def test_complete_profile_population_matches():
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


@pytest.mark.parametrize('value', [None, '', 'PENDING', 'a'*63, 'A'*64, 123])
def test_missing_or_placeholder_binding_rejects_before_checker(value, monkeypatch):
    monkeypatch.setattr(profile.completed, 'main', lambda:pytest.fail('invalid SHA must reject first'))
    with pytest.raises(ValueError, match='actual completed SHA256'): profile.capture_verification(value)


@pytest.mark.parametrize('mode', ['once', 'none', 'twice', 'wrong_path', 'cli_changed', 'early_digest'])
def test_frozen_checker_body_and_cli_are_private_and_never_write(monkeypatch, tmp_path, mode):
    def emit():
        parser = argparse.ArgumentParser()
        parser.add_argument('--result-sha256' if MODE != 'cli_changed' else '--other', required=True)
        args = parser.parse_args()
        if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive')
        if MODE == 'early_digest': digest(OUTPUT)
        if MODE == 'none': return
        write_json(OUTPUT if MODE != 'wrong_path' else 'wrong', dict(sha=args.result_sha256))
        if MODE == 'twice': write_json(OUTPUT, {})
        print('COMPLETE', digest(OUTPUT))
    receipt = tmp_path/'original.json'; receipt.write_text('original receipt')
    forbidden = lambda *a, **k:pytest.fail('original write or CLI must not be used')
    function = FunctionType(emit.__code__, dict(MODE=mode, OUTPUT=receipt, write_json=forbidden,
        argparse=forbidden, digest=forbidden, print=forbidden))
    monkeypatch.setattr(profile.completed, 'main', function)
    monkeypatch.setattr(profile, 'OUTPUT', tmp_path/'absent')
    before = dict(function.__globals__); argv = list(sys.argv)
    if mode == 'once': assert profile.capture_verification('a'*64) == {'sha':'a'*64}
    else:
        with pytest.raises(ValueError): profile.capture_verification('a'*64)
    assert function.__globals__ == before and sys.argv == argv
    assert receipt.read_text() == 'original receipt' and not profile.OUTPUT.exists()


def test_actual_frozen_checker_rejects_live_owner_before_reading_artifacts(monkeypatch, tmp_path):
    execution = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), owner={'live': True})
    (tmp_path/'execution.json').write_text(json.dumps(execution))
    monkeypatch.setattr(profile.completed, 'ROOT', tmp_path)
    monkeypatch.setattr(profile.completed, 'EXECUTION', 'execution.json')
    monkeypatch.setattr(profile.completed, 'verify', lambda *a:None)
    monkeypatch.setattr(profile.paired, 'owner_live', lambda owner:owner['live'])
    monkeypatch.setattr(profile.completed, 'verify_artifacts', lambda *a:pytest.fail('owner gate comes first'))
    monkeypatch.setattr(profile, 'OUTPUT', tmp_path/'absent')
    with pytest.raises(ValueError, match='owner must be ended'): profile.capture_verification('a'*64)
    assert not profile.OUTPUT.exists()


def test_changed_complete_witness_rejects_before_profile_artifact_reads(monkeypatch, tmp_path):
    witness = dict(utc='old', observed_state_checks=[{'frame': 1173}], sensing_scope={'failed':True})
    (tmp_path/'witness.json').write_text(json.dumps(witness))
    monkeypatch.setattr(profile, 'ROOT', tmp_path); monkeypatch.setattr(profile, 'VERIFICATION', 'witness.json')
    monkeypatch.setattr(profile, 'verification_sources', lambda *a:{})
    monkeypatch.setattr(profile, 'capture_verification', lambda *a:witness | {'sensing_scope':{'failed':False}})
    monkeypatch.setattr(profile, 'read_json', lambda *a:pytest.fail('changed evidence must reject before reads'))
    with pytest.raises(ValueError, match='must reconstruct'): profile.admit_completed({}, 'a'*64, 'b'*64)


@pytest.mark.parametrize('changed', ['status', 'result', 'launch', 'source'])
def test_completion_sources_require_exact_result_launch_and_original_sources(monkeypatch, tmp_path, changed):
    witness = dict(status='TILED_DENSITY_PROGRESSIVE_FLOOR_COMPLETION_VERIFIED', result_sha256='b'*64,
        artifact_sha256={'launch.json':profile.LAUNCH_SHA}, source_sha256={'fixed':'c'*64})
    if changed == 'status': witness['status'] = 'INCOMPLETE'
    if changed == 'result': witness['result_sha256'] = 'd'*64
    if changed == 'launch': witness['artifact_sha256']['launch.json'] = 'd'*64
    if changed == 'source': witness['source_sha256']['fixed'] = 'd'*64
    (tmp_path/'witness.json').write_text(json.dumps(witness))
    monkeypatch.setattr(profile, 'ROOT', tmp_path); monkeypatch.setattr(profile, 'VERIFICATION', 'witness.json')
    monkeypatch.setattr(profile, 'verify', lambda *a:None)
    monkeypatch.setattr(profile, 'discover_sources', lambda *a:pytest.fail('invalid witness must reject first'))
    with pytest.raises(ValueError): profile.verification_sources({'fixed':'c'*64}, 'a'*64, 'b'*64)


def environment(monkeypatch):
    monkeypatch.setattr(profile.original.cv2.ocl, 'useOpenCL', lambda:False)
    for key, value in dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled').items(): monkeypatch.setenv(key, value)


def test_existing_attempt_rejected_before_source_work(monkeypatch, tmp_path):
    environment(monkeypatch)
    monkeypatch.setattr(profile, 'OUTPUT', tmp_path)
    monkeypatch.setattr(profile, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(profile, 'prepared_sources', lambda:pytest.fail('exclusive attempt must reject first'))
    monkeypatch.setattr(sys, 'argv', ['profile'])
    with pytest.raises(ValueError, match='exclusive'): profile.main()


def test_preflight_without_completed_replay_never_admits_raw_model_inputs_or_creates_output(monkeypatch, tmp_path, capsys):
    environment(monkeypatch)
    monkeypatch.setattr(profile, 'OUTPUT', tmp_path/'absent')
    monkeypatch.setattr(profile, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(profile, 'prepared_sources', lambda:{'synthetic':'a'*64})
    monkeypatch.setattr(profile, 'verification_sources', lambda *a:pytest.fail('source-only preflight'))
    monkeypatch.setattr(profile, 'admit_completed', lambda *a:pytest.fail('source-only preflight'))
    monkeypatch.setattr(profile, 'create_output', lambda *a:pytest.fail('source-only preflight'))
    monkeypatch.setattr(profile.original.reference, 'hardware', lambda:dict(memory_available_bytes=64*1024**3,
        artifact_free_bytes=41*1024**3, physical_cpus=4))
    monkeypatch.setattr(sys, 'argv', ['profile', '--source-preflight-only'])
    profile.main()
    assert 'PROFILE_SOURCE_PREFLIGHT' in capsys.readouterr().out and not profile.OUTPUT.exists()


def test_capture_logging_digest_matches_original_json_writer_and_other_hashes_delegate(monkeypatch,tmp_path):
    payload={'state':{'failure':True,'number':0.001},'rows':1428}
    expected_path=tmp_path/'serialization.json'
    profile.write_json(expected_path,payload)
    expected=profile.digest(expected_path)
    original_digest=profile.completed.digest
    accesses=[]
    def recorded_digest(path):
        accesses.append(path)
        return original_digest(path)
    def emit():
        parser=argparse.ArgumentParser();parser.add_argument('--result-sha256',required=True)
        parser.parse_args()
        assert digest(OTHER)==EXPECTED
        write_json(OUTPUT,PAYLOAD)
        assert digest(OUTPUT)==EXPECTED
    function=FunctionType(emit.__code__,dict(OTHER=expected_path,EXPECTED=expected,PAYLOAD=payload))
    monkeypatch.setattr(profile.completed,'main',function)
    monkeypatch.setattr(profile.completed,'digest',recorded_digest)
    monkeypatch.setattr(profile,'OUTPUT',tmp_path/'absent')
    assert profile.capture_verification('a'*64)==payload
    assert accesses==[expected_path] and not profile.OUTPUT.exists()

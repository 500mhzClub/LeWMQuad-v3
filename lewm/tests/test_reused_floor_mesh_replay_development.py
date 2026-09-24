from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import pytest

from scripts import replay_go2_reused_floor_mesh_prefix_v1 as candidate


def test_original_complete_replay_and_state_checks_are_not_rewritten():
    assert hashlib.sha256(Path('scripts/replay_go2_receipt_copied_anchored_prefix_v1.py').read_bytes()).hexdigest() == (
        '0fdc4ac3493ce8c4744cab4a7270c3ff8b41163fe9575988a2111fbbe5bb1a7d')
    original = candidate.preceding.replay; before = original.__globals__.copy()
    new = candidate.isolated_replay()
    overrides = dict(ReceiptCopiedAnchoredController=candidate.ReusedFloorMeshController,
        normalize_candidate=candidate.normalize_candidate, OUTPUT=candidate.OUTPUT, print=candidate.progress)
    assert new.__code__ is original.__code__ and new.__closure__ is None
    assert new.__defaults__ is original.__defaults__ and new.__kwdefaults__ is original.__kwdefaults__
    assert new.__globals__.keys() == before.keys() | overrides.keys()
    for name, value in new.__globals__.items():
        assert value is (overrides[name] if name in overrides else before[name])
    assert all(original.__globals__[name] is value for name,value in before.items())
    assert candidate.FRAMES == 405 and candidate.STATE_FRAMES == (3,12,395,404)


def test_normalization_changes_only_three_declared_top_level_metadata_fields():
    d = dict(controller=candidate.CONTROLLER, **{candidate.FLAG:True,candidate.frozen.FLAG:True},
        requested_command=[.2,0.,0.], new_selection={'prediction':[1.,2.],
        'nested':{candidate.FLAG:True,'controller':candidate.CONTROLLER}})
    before = deepcopy(d); expected = deepcopy(d)
    expected.pop(candidate.FLAG);expected.pop(candidate.frozen.FLAG)
    expected['controller']='residual_anchored_continuation_controller_v1'
    assert candidate.normalize_candidate(d)==expected and d==before
    for patch in [{'controller':'other'},{candidate.FLAG:False},{candidate.FLAG:1},
                  {candidate.frozen.FLAG:False},{candidate.frozen.FLAG:1}]:
        with pytest.raises(ValueError):candidate.normalize_candidate(d|patch)


def test_source_preflight_cannot_start_raw_replay_or_admit_incomplete_profile(monkeypatch,tmp_path,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('runtime operation during source preflight')
    monkeypatch.setattr(candidate,'OUTPUT',tmp_path/'absent')
    monkeypatch.setattr(candidate,'validate_root',lambda *args,**kwargs:None)
    monkeypatch.setattr(candidate,'prepared_inputs',lambda:{'source_sha256':{}})
    monkeypatch.setattr(candidate,'discover_sources',lambda *args:{})
    monkeypatch.setattr(candidate,'verify',lambda *args:None)
    monkeypatch.setattr(candidate.profile.reference,'hardware',lambda:{})
    monkeypatch.setattr(candidate.optimized_profile,'resources_for',lambda *args:None)
    monkeypatch.setattr(candidate,'completed_profile',forbidden)
    monkeypatch.setattr(candidate.profile.reference,'admit_worker',forbidden)
    monkeypatch.setattr(candidate,'create_output',forbidden)
    monkeypatch.setattr(candidate,'isolated_replay',forbidden)
    monkeypatch.setattr(sys,'argv',[candidate.SOURCE,'--source-preflight-only'])
    candidate.main()
    assert 'REUSED_FLOOR_MESH_SOURCE_PREFLIGHT_PASS 0' in capsys.readouterr().out
    assert not candidate.OUTPUT.exists()


@pytest.mark.parametrize('fault',['status','missing_artifact','source','forecast_count','model_state','native','false_bool'])
def test_incomplete_or_changed_profile_is_rejected_before_raw_input_admission(fault,monkeypatch,tmp_path):
    report=dict(frames=405,raw_model_forecast_comparisons=402,
        model_state_sha256=candidate.profile.reference.MODEL_SHA,model_state_unchanged=True,
        complete_original_decisions_reconstructed=True,complete_normalized_candidate_decisions_exact=True,
        no_observation_405_consumed=True,invocation_frozen_footprint_receipts=True,
        normalization_outside_profiled_region=True,native_execution=False,policy_changed=False)
    names=['launch.json','comparison.jsonl','early_navigation.prof','early_navigation.json',
        'repeated_hold.prof','repeated_hold.json']
    result=dict(status='FROZEN_FOOTPRINT_CONTROLLER_WINDOWS_PROFILE_V1_COMPLETE',
        source_sha256={},artifact_sha256={n:'a'*64 for n in names},report=report,
        native_execution=False,goal_achieved=False)
    result['artifact_sha256']['launch.json']=candidate.PROFILE_LAUNCH_SHA
    if fault=='status':result['status']='INCOMPLETE'
    elif fault=='missing_artifact':result['artifact_sha256'].pop('repeated_hold.prof')
    elif fault=='source':result['source_sha256']['unexpected']='a'*64
    elif fault=='forecast_count':report['raw_model_forecast_comparisons']=401
    elif fault=='model_state':report['model_state_sha256']='b'*64
    elif fault=='native':result['native_execution']=True
    else:report['model_state_unchanged']=1
    monkeypatch.setattr(candidate.optimized_profile,'OUTPUT',tmp_path)
    monkeypatch.setattr(candidate,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(candidate,'read_json',lambda *args:result)
    with pytest.raises(ValueError):candidate.completed_profile(candidate.PROFILE_RESULT_SHA,{'source_sha256':{}})


def test_unreviewed_completed_profile_identity_is_rejected():
    with pytest.raises(ValueError,match='exact independently checked'):
        candidate.completed_profile('a'*64,{'source_sha256':{}})


def test_raw_launch_requires_explicit_completed_profile_identity(monkeypatch):
    monkeypatch.setattr(sys,'argv',[candidate.SOURCE])
    monkeypatch.setattr(candidate,'prepared_inputs',lambda:pytest.fail('must reject before input work'))
    with pytest.raises(SystemExit) as error:candidate.main()
    assert error.value.code==2

"""Fixed profiling windows, complete decisions and unchanged input ownership."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import profile_go2_frozen_footprint_controller_windows_v1 as profile


def test_profile_does_not_sum_overlapping_cumulative_times():
    result = profile.profile_summary({('parent.py',1,'parent'):(1,1,.2,1.,{}),
        ('child.py',2,'child'):(2,2,.8,.8,{})})
    assert result['total_exclusive_profiled_s'] == 1.
    assert result['modules_by_exclusive_time'][0] == {'filename':'child.py','self_time_s':.8}
    assert result['cumulative_times_must_not_be_summed'] is True
    assert result['profiler_overhead_removed'] is False


@pytest.mark.parametrize('name', ['sealed_test.json', 'data/sealed/source.py', 'data/sealed_future/source.py'])
def test_protected_profile_names_reject_before_serialization(name):
    with pytest.raises(ValueError): profile.profile_summary({(name,1,'f'):(1,1,.1,.1,{})})


@pytest.mark.parametrize('entry', [(2,1,.1,.1,{}),(1,1,-.1,.1,{}),(1,1,.1,float('nan'),{})])
def test_invalid_profile_measurements_reject(entry):
    with pytest.raises(ValueError): profile.profile_summary({('source.py',1,'f'):entry})


@pytest.mark.parametrize('memory,disk,cpus', [(47,100,16),(64,40,16),(64,100,2)])
def test_resource_shortfall_rejects(memory,disk,cpus):
    with pytest.raises(ValueError): profile.resources_for(dict(memory_available_bytes=memory*1024**3,
        artifact_free_bytes=disk*1024**3,physical_cpus=cpus))


def synthetic_replay(monkeypatch, tmp_path, fault=None):
    decisions = [dict(controller='residual_anchored_continuation_controller_v1', terminal=None,
        requested_command=[0.,0.,0.], mission_receipt={'hold_required':False},
        new_selection=None if i<3 else {'action':'hold','prediction':[[float(i)]],
            'complete_receipt':{'frame':i}}) for i in range(406)]
    tape = [dict(tick=i,completed=True,pre_sample_index=749+50*i,post_sample_index=799+50*i,
        requested_command=[0.,0.,0.]) for i in range(405)]
    calls = []; normalized = []; parameter = SimpleNamespace(grad=None)
    model = SimpleNamespace(state_dict=lambda:{},parameters=lambda:[parameter])

    class Reader:
        def __init__(self, directory): pass
        def packet(self, frame):
            assert frame < 405
            return np.array([frame]),np.array([frame]),np.array([frame]),frame

    class Controller:
        def __init__(self, actual_model, geometry, **kwargs):
            assert actual_model is model
        def observe(self, p,d,fast,*,now_ns,auxiliary_depth,auxiliary_rgb):
            frame=int(p[0]);calls.append(frame)
            result=deepcopy(decisions[frame])
            result.update(controller=profile.completed_replay.CONTROLLER,
                **{profile.completed_replay.FLAG:True})
            if frame==17:
                if fault=='receipt':result['new_selection']['complete_receipt']['frame']=-1
                elif fault=='input':p[0]+=1
                elif fault=='metadata':result[profile.completed_replay.FLAG]=False
                elif fault=='gradient':parameter.grad=object()
            return result

    original_normalizer=profile.normalize_candidate
    def normalize(value):
        normalized.append(value)
        return original_normalizer(value)

    monkeypatch.setattr(profile,'OUTPUT',tmp_path)
    monkeypatch.setattr(profile,'FrozenFootprintAnchoredController',Controller)
    monkeypatch.setattr(profile,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(profile,'normalize_candidate',normalize)
    monkeypatch.setattr(profile.reference.original,'assigned_model',lambda *args:model)
    monkeypatch.setattr(profile,'state_digest',lambda *args:profile.reference.MODEL_SHA)
    monkeypatch.setattr(profile,'ArticulatedCollisionGeometry',lambda *args:object())
    monkeypatch.setattr(profile,'read_json',lambda directory,name:
        tape if name=='command_tape.json' else ([{}]*405 if name=='auxiliary_camera_audit.json' else {}))
    monkeypatch.setattr(profile,'read_rows',lambda directory:
        (dict(tick=i,decision=d) for i,d in enumerate(decisions)))
    monkeypatch.setattr(profile,'public_acquisition',lambda value:value)
    monkeypatch.setattr(profile,'packet',lambda *args,**kwargs:(np.array([1]),np.array([2])))
    return calls,normalized,decisions


def test_complete_synthetic_replay_profiles_only_fixed_windows_and_preserves_all_decisions(monkeypatch,tmp_path):
    calls,normalized,decisions=synthetic_replay(monkeypatch,tmp_path)
    result=profile.replay()
    assert calls==list(range(405)) and len(normalized)==405
    assert result['frames']==405 and result['raw_model_forecast_comparisons']==402
    assert result['normalization_outside_profiled_region'] is True
    assert result['complete_normalized_candidate_decisions_exact'] is True
    assert result['sensor_acquisition_profiled'] is False
    assert result['real_time_qualified'] is False and result['navigation_qualified'] is False
    rows=[json.loads(line) for line in (tmp_path/'comparison.jsonl').read_text().splitlines()]
    for i,row in enumerate(rows):
        assert row['original_decision_sha256']==profile.reference.saved.identity(decisions[i])
        assert row['candidate_decision_sha256']==profile.reference.saved.identity(normalized[i])
    for name,(first,last) in profile.WINDOWS.items():
        assert [r['frame'] for r in result['windows'][name]['observations']]==list(range(first,last+1))
        assert (tmp_path/(name+'.prof')).is_file() and (tmp_path/(name+'.json')).is_file()
        measured=json.loads((tmp_path/(name+'.json')).read_text())['functions']
        observed=[r for r in measured if r['function']=='observe']
        assert len(observed)==1 and observed[0]['calls']==10
        assert not any(r['function'] in ('normalize','normalize_candidate') for r in measured)


@pytest.mark.parametrize('fault',['receipt','input','metadata','gradient'])
def test_changed_evidence_arrays_metadata_or_gradients_reject(monkeypatch,tmp_path,fault):
    calls,_,_=synthetic_replay(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError):profile.replay()
    assert calls==list(range(405 if fault=='gradient' else 18))
    assert not (tmp_path/'early_navigation.json').exists()


def test_source_preflight_cannot_admit_worker_or_run_profile(monkeypatch,tmp_path,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('runtime during source-only preflight')
    monkeypatch.setattr(profile,'OUTPUT',tmp_path/'uncreated')
    monkeypatch.setattr(profile,'validate_root',lambda *args,**kwargs:None)
    monkeypatch.setattr(profile,'completed_inputs',lambda:({}, {'verification_source_sha256':{}}))
    monkeypatch.setattr(profile,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(profile,'discover_sources',lambda *args:{})
    monkeypatch.setattr(profile,'verify',lambda *args:None)
    monkeypatch.setattr(profile.reference,'hardware',lambda:{})
    monkeypatch.setattr(profile,'resources_for',lambda *args:None)
    monkeypatch.setattr(profile.reference,'admit_worker',forbidden)
    monkeypatch.setattr(profile,'create_output',forbidden)
    monkeypatch.setattr(profile,'replay',forbidden)
    monkeypatch.setattr(sys,'argv',[profile.SOURCE,'--source-preflight-only'])
    profile.main()
    assert 'FROZEN_FOOTPRINT_CONTROLLER_PROFILE_SOURCE_PREFLIGHT_PASS 0' in capsys.readouterr().out
    assert not profile.OUTPUT.exists()

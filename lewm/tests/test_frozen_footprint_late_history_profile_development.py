"""Fixed full-history profiling with substituted packets/models, no native data."""
from collections import deque
from copy import deepcopy
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import profile_go2_frozen_footprint_late_history_v1 as profile


def synthetic(monkeypatch, tmp_path, fault=None):
    decisions = [dict(controller='residual_anchored_continuation_controller_v1', terminal=None,
        requested_command=[0.,0.,0.], mission_receipt={'hold_required':False},
        new_selection=None if i<3 else dict(action='hold',prediction=[[float(i)]],receipt={'frame':i}))
        for i in range(profile.FRAMES+1)]
    tape = [dict(tick=i, completed=True, pre_sample_index=749+50*i,post_sample_index=799+50*i,
        requested_command=[0.,0.,0.]) for i in range(profile.FRAMES)]
    parameter = SimpleNamespace(grad=None); model = SimpleNamespace(state_dict=lambda:{},parameters=lambda:[parameter])
    called = []; normalized = []
    class Reader:
        def __init__(self, directory): pass
        def packet(self, frame):
            assert frame < profile.FRAMES
            return np.array([frame]),np.array([frame]),np.array([frame]),frame
    class Controller:
        def __init__(self, assigned, geometry, **kwargs):
            assert assigned is model
            self.history = []; self.memory = SimpleNamespace(index={'synthetic':True})
            self.mapper = SimpleNamespace(floor={},occupied={}); self.residual = SimpleNamespace(pending=None)
        def observe(self,p,d,fast,*,now_ns,auxiliary_depth,auxiliary_rgb):
            frame = int(p[0]); assert len(self.history) == frame
            self.history.append(frame); called.append(frame)
            result = deepcopy(decisions[frame]); result.update(controller=profile.previous.completed_replay.CONTROLLER,
                **{profile.previous.completed_replay.FLAG:True})
            if frame == 1001:
                if fault == 'receipt': result['new_selection']['receipt']['frame'] = -1
                elif fault == 'input': p[0] = -1
                elif fault == 'metadata': result[profile.previous.completed_replay.FLAG] = False
                elif fault == 'gradient': parameter.grad = object()
                elif fault == 'terminal': result['terminal'] = 'unexpected'
            return result
    original = profile.normalize_candidate
    def normalize(value):
        normalized.append(deepcopy(value)); return original(value)
    monkeypatch.setattr(profile,'OUTPUT',tmp_path)
    monkeypatch.setattr(profile,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(profile,'FrozenFootprintAnchoredController',Controller)
    monkeypatch.setattr(profile,'normalize_candidate',normalize)
    monkeypatch.setattr(profile.reference.original,'assigned_model',lambda *args:model)
    monkeypatch.setattr(profile,'state_digest',lambda *args:profile.reference.MODEL_SHA)
    monkeypatch.setattr(profile,'ArticulatedCollisionGeometry',lambda *args:object())
    monkeypatch.setattr(profile,'read_json',lambda root,name:
        tape if name=='command_tape.json' else ([{}]*profile.FRAMES if name=='auxiliary_camera_audit.json' else {}))
    def rows(directory):
        for i,decision in enumerate(decisions):
            assert i < profile.FRAMES, 'the following observation must never be consumed'
            yield dict(tick=i,decision=decision)
    monkeypatch.setattr(profile,'read_rows',rows)
    monkeypatch.setattr(profile,'public_acquisition',lambda value:value)
    monkeypatch.setattr(profile,'packet',lambda *args,**kwargs:(np.array([1]),np.array([2])))
    return called,normalized,decisions,tape


def test_complete_history_and_all_fixed_windows_without_reading_following_observation(monkeypatch,tmp_path):
    called,normalized,decisions,_ = synthetic(monkeypatch,tmp_path)
    report = profile.replay()
    assert called == list(range(1428)) and len(normalized) == 1428
    assert report['raw_model_forecast_comparisons'] == 1425
    assert report['no_observation_1428_consumed'] and report['last_replayed_observation'] == 1427
    assert not report['retained_state_identity_established'] and not report['speedup_established']
    assert not report['real_time_qualified'] and not report['navigation_qualified']
    rows = [json.loads(line) for line in (tmp_path/'comparison.jsonl').read_text().splitlines()]
    assert len(rows) == 1428
    for frame,row in enumerate(rows):
        assert row['original_decision_sha256'] == profile.reference.saved.identity(decisions[frame])
        assert row['candidate_decision_sha256'] == profile.reference.saved.identity(normalized[frame])
        expected = next((name for name,(first,last) in profile.WINDOWS.items() if first<=frame<=last),None)
        assert row['profiled_window'] == expected
    for name,(first,last) in profile.WINDOWS.items():
        assert [r['frame'] for r in report['windows'][name]['observations']] == list(range(first,last+1))
        functions = json.loads((tmp_path/(name+'.json')).read_text())['functions']
        assert sum(r['calls'] for r in functions if r['function']=='observe') == 10
        assert not any(r['function'] in ('normalize','normalize_candidate','state_sizes') for r in functions)
        assert report['state_size_snapshots'][name]['history']['entries'] == last+1


@pytest.mark.parametrize('fault', ['receipt','input','metadata','gradient','terminal'])
def test_late_divergence_or_mutation_is_terminal(monkeypatch,tmp_path,fault):
    called,_,_,_ = synthetic(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError): profile.replay()
    assert called == list(range(1428 if fault=='gradient' else 1002))
    assert not (tmp_path/'late_navigation.json').exists()


def test_wrong_late_command_endpoint_rejected_before_observation(monkeypatch,tmp_path):
    called,_,_,tape = synthetic(monkeypatch,tmp_path)
    tape[1001]['post_sample_index'] += 1
    with pytest.raises(ValueError,match='command endpoints'): profile.replay()
    assert called == list(range(1001))


def test_read_only_state_sizes_do_not_serialize_payloads_or_mutate_arrays():
    pixels = np.arange(12,dtype=np.float32).reshape(3,4); original = pixels.copy()
    controller = SimpleNamespace(memory=SimpleNamespace(pixels=pixels,index={1:'a',2:'b'}),
        mapper=SimpleNamespace(floor={'a':1},occupied={}),history=deque([1,2,3]),residual=SimpleNamespace(pending=None))
    result = profile.state_sizes(controller)
    assert result['memory']['fields']['pixels'] == dict(type='numpy.ndarray',shape=[3,4],bytes=48)
    assert result['memory']['fields']['index']['entries'] == 2
    assert result['history']['entries'] == 3
    np.testing.assert_array_equal(pixels,original)


@pytest.mark.parametrize('fault', [None,'visibility','frame','success'])
def test_original_visibility_failure_is_retained_explicitly(monkeypatch,fault):
    record = dict(strict_physical_visibility_pass=False,hard_measurement_failed_frames=[1173],verified_round_trip=False)
    if fault=='visibility':record['strict_physical_visibility_pass']=True
    elif fault=='frame':record['hard_measurement_failed_frames']=[]
    elif fault=='success':record['verified_round_trip']=True
    monkeypatch.setattr(profile,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(profile,'read_json',lambda *args:record)
    if fault:
        with pytest.raises(ValueError):profile.sensing_scope()
    else:
        scope=profile.sensing_scope()
        assert scope['failure_frame_inside_profiled_history'] and scope['known_invalid_sensing_retained']
        assert not scope['qualified_sensing_prefix_claimed'] and not scope['navigation_verified']


def test_source_preflight_cannot_admit_replay_or_create_output(monkeypatch,tmp_path,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('runtime during source preflight')
    for name,value in dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0').items():
        monkeypatch.setenv(name,value)
    monkeypatch.setattr(profile,'OUTPUT',tmp_path/'absent')
    monkeypatch.setattr(profile,'validate_root',lambda *args,**kwargs:None)
    monkeypatch.setattr(profile,'prepared_sources',lambda:{})
    monkeypatch.setattr(profile,'sensing_scope',lambda:{})
    monkeypatch.setattr(profile.reference,'hardware',lambda:{})
    monkeypatch.setattr(profile,'resources_for',lambda *args:None)
    monkeypatch.setattr(profile.reference,'admit_worker',forbidden)
    monkeypatch.setattr(profile,'create_output',forbidden);monkeypatch.setattr(profile,'replay',forbidden)
    monkeypatch.setattr(sys,'argv',[profile.SOURCE,'--source-preflight-only'])
    profile.main()
    assert not profile.OUTPUT.exists()
    assert 'LATE_HISTORY_PROFILE_SOURCE_PREFLIGHT_PASS 0' in capsys.readouterr().out

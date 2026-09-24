"""Reject changed early behavior and incomplete claims of boundary recovery."""
from copy import deepcopy
import pytest
from scripts.replay_go2_contact_anchored_direct_flow_controller_prefix_v1 import compare, CONTROLLER, FLAG
from lewm.direct_flow_commitment_contact_controller_development import DirectFlowCommitmentContactController
from lewm.commitment_contact_anchored_controller_development import CommitmentContactAnchoredController
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.novel_maze_round_trip_scene_development import public_mission


def early():
    original = dict(controller='commitment_contact_anchored_continuation_controller_v1',tick=3,
        terminal=None,failure=None,requested_command=[0.,0.,0.],new_selection={'prediction':{'x':[1]}},
        original_visual_evidence={'status':'CURRENT_VISUAL_POSE'},evidence={'pose':'registered'})
    candidate = deepcopy(original);candidate.update(controller=CONTROLLER,**{FLAG:True})
    return original,candidate


def boundary():
    original,candidate = early();original.update(tick=560,terminal='SENSOR_OR_MODEL_FAILURE',failure='missing',new_selection=None,evidence=None)
    original['original_visual_evidence'] = dict(status='VISUAL_TERMINAL_FAILURE',camera_selection={'a':1},
        continuity_evidence={'b':2},reference_selection={'c':3})
    candidate.update(tick=561,new_selection={'prediction':{'x':[1]},'action':None})
    candidate['original_visual_evidence'] = dict(status='CURRENT_VISUAL_POSE',direct_corner_flow_fallback=dict(
        accepted=True,original_camera_selection={'a':1},original_auxiliary_continuity={'b':2},original_reference_selection={'c':3}))
    return original,candidate


def test_complete_early_decision_and_prediction_are_compared():
    a,b=early();r=compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=3)
    assert r['original_forecast_compared'] and not r['stop']
    b['new_selection']['prediction']['x'][0]=2
    with pytest.raises(ValueError,match='complete original'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=3)


def test_raw_observer_mismatch_is_never_normalized_away():
    a,b=boundary()
    with pytest.raises(ValueError,match='observer evidence'):compare(a,b,a['requested_command'],{},frame=561)


def test_recovered_boundary_always_stops_even_with_same_zero_request():
    a,b=boundary();r=compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=561)
    assert r['stop'] and r['full_controller_recovered'] and not r['requested_command_changed']


@pytest.mark.parametrize('change',[{'evidence':None},{'new_selection':None},{'tick':560}])
def test_partial_recovery_cannot_claim_full_controller_recovery(change):
    a,b=boundary();b.update(change)
    with pytest.raises(ValueError,match='registered pose'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=561)


def test_wrong_executed_original_command_is_rejected():
    a,b=early()
    with pytest.raises(ValueError,match='executed command'):compare(a,b,[.2,0.,0.],b['original_visual_evidence'],frame=3)


def test_wrong_new_action_command_is_rejected():
    a,b=boundary();b['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError,match='selected action'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=561)


def test_floor_failure_is_a_preserved_negative_result():
    a,b=boundary();b.update(terminal='SENSOR_OR_MODEL_FAILURE',failure='floor registration failed',evidence=None)
    r=compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=561)
    assert r['stop'] and not r['full_controller_recovered']
    b['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError,match='zero command'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=561)


@pytest.mark.parametrize('key',['original_camera_selection','original_auxiliary_continuity','original_reference_selection'])
def test_original_failure_evidence_cannot_be_dropped(key):
    a,b=boundary();b['original_visual_evidence']['direct_corner_flow_fallback'][key]={}
    with pytest.raises(ValueError,match='original failure'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=561)


def test_integration_preserves_original_planning_and_registration_classes():
    options=dict(public_mission=public_mission(2),navigation_ticks=3000,condition='supervised_rollout',variant='full',persistent=True)
    model=object();geometry=object()
    original=CommitmentContactAnchoredController(model,geometry,**options)
    candidate=DirectFlowCommitmentContactController(model,geometry,**options)
    assert type(candidate.motion) is DirectFlowDualCameraVisualMotion
    for name in ('selector','registration','mapper','memory','mission','residual'):
        assert type(getattr(original,name)) is type(getattr(candidate,name))
        assert getattr(original,name) is not getattr(candidate,name)
    assert candidate.model is original.model is model
    assert candidate.observe.__func__ is original.observe.__func__
    assert candidate.advance.__func__ is original.advance.__func__


@pytest.mark.parametrize('original_diverges', [False, True])
def test_full_loop_uses_two_models_reconstructs_original_and_stops_at_boundary(tmp_path, monkeypatch, original_diverges):
    from contextlib import contextmanager
    from scripts import replay_go2_contact_anchored_direct_flow_controller_prefix_v1 as run
    monkeypatch.setattr(run, 'BOUNDARY', 5)
    monkeypatch.setattr(run, 'OUTPUT', tmp_path); (tmp_path/run.observer.NAME).write_bytes(b'')
    originals = []; candidates = []
    for frame in range(6):
        a,b = boundary() if frame == 5 else early()
        a['tick'] = 4 if frame == 5 else frame; b['tick'] = frame
        a['selected_action'] = b['selected_action'] = None
        if frame < 3: a['new_selection'] = b['new_selection'] = None
        originals.append(a); candidates.append(b)
    source = run.native.OUTPUT/run.CASE[0]; consumed = []; observations = []; models = []
    def rows(directory):
        for frame in range(7):
            if directory == source:
                consumed.append(frame)
                yield dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame,
                    decision=deepcopy(originals[frame]))
            else:
                yield dict(tick=frame, public_packet_sha256='public', original=deepcopy(originals[frame]['original_visual_evidence']),
                    candidate=deepcopy(candidates[frame]['original_visual_evidence']))
    monkeypatch.setattr(run.observer, 'read_rows', rows)
    tape = [dict(tick=i, completed=True, pre_sample_index=749+50*i, post_sample_index=799+50*i,
        requested_command=[0.,0.,0.]) for i in range(6)]
    monkeypatch.setattr(run, 'read_json', lambda root, name:tape if name == 'command_tape.json' else [{}]*6 if name == 'auxiliary_camera_audit.json' else {})
    class Model:
        def state_dict(self): return {}
        def parameters(self): return []
    def assigned(*a):
        m=Model();models.append(m);return m
    monkeypatch.setattr(run.native, 'assigned_model', assigned)
    monkeypatch.setattr(run, 'state_digest', lambda state:run.MODEL_SHA)
    monkeypatch.setattr(run, 'ArticulatedCollisionGeometry', lambda *a:object())
    class Original:
        def __init__(self, model, *a, **k): self.model=model
        def observe(self, policy, *a, **k):
            frame=policy['frame'];observations.append(('original',frame));value=deepcopy(originals[frame])
            if original_diverges and frame == 4: value['new_selection']['prediction']['x'][0]=2
            return value
    class Candidate(Original):
        def observe(self, policy, *a, **k):
            frame=policy['frame'];observations.append(('candidate',frame));return deepcopy(candidates[frame])
    monkeypatch.setattr(run, 'CommitmentContactAnchoredController', Original)
    monkeypatch.setattr(run, 'DirectFlowCommitmentContactController', Candidate)
    class Reader:
        def __init__(self, directory): pass
        def packet(self, frame): return {'frame':frame},{},{},1_500_000_000+100_000_000*frame
    monkeypatch.setattr(run.observer, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(run.observer, 'packet', lambda *a, **k:({},{}))
    monkeypatch.setattr(run.observer, 'public_acquisition', lambda x:x)
    monkeypatch.setattr(run.observer, 'fingerprint', lambda x:'public')
    monkeypatch.setattr(run, 'current_dual_camera_pose', lambda *a, **k:None)
    monkeypatch.setattr(run, 'current_measured_floor_pose', lambda *a, **k:None)
    saved=[]
    @contextmanager
    def writer(root): yield saved.append
    monkeypatch.setattr(run.observer, 'writer', writer)
    if original_diverges:
        with pytest.raises(ValueError, match='original controller decision'): run.replay()
        assert consumed == list(range(5))
    else:
        result=run.replay()
        assert result['frames']==6 and result['original_forecasts_compared']==2
        assert result['boundary_comparison']['full_controller_recovered'] is True
        assert consumed==list(range(6))
    assert len(models)==2 and models[0] is not models[1]
    assert observations==[(kind,i) for i in consumed for kind in ('original','candidate')]

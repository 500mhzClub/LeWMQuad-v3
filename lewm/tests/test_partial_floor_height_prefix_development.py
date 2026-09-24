"""No policy-state drift before failure and no borrowed outcome after repair."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from lewm import partial_floor_height_prefix_development as comparison
from lewm.tests.test_partial_floor_height_development import repaired, clock
from scripts import replay_go2_partial_floor_height_prefix_v1 as runner


def rows(frame):
    raw = dict(decision_ns=1_500_000_000+100_000_000*frame, current_pose=dict(frame=frame))
    old = dict(controller='direct_flow_floor_transport_controller_v1', tick=frame,
        terminal=None, failure=None, requested_command=[0., 0., 0.], original_visual_evidence=raw,
        evidence={'original':True}, mission_receipt={'frame':frame},
        new_selection=None if frame < 3 else dict(prediction={'raw':frame}, action='hold'))
    new = deepcopy(old)|dict(controller='partial_height_direct_flow_controller_v1',
        partial_floor_height_constraint_enabled=True)
    if frame == 504:
        old.update(tick=503, terminal='SENSOR_OR_MODEL_FAILURE', failure=comparison.CONFLICT, evidence=None)
        new.update(evidence=dict(schema=comparison.SCHEMA, original_visual_evidence=deepcopy(raw),
            decision_ns=raw['decision_ns'], current_pose=dict(frame=frame)),
            new_selection=dict(prediction={'raw':frame}, action='left_turn'), requested_command=[0., 0., .45])
    return old, new


@pytest.mark.parametrize('fault', [None, 'earlier_state', 'earlier_raw', 'actual_command', 'failure',
    'boundary_raw', 'boundary_pose', 'forecast', 'action', 'later'])
def test_comparison_rejects_unrelated_changes_and_wrong_boundary(fault):
    frame = 503 if fault in ('earlier_state', 'earlier_raw') else 504
    old, new = rows(frame); command = old['requested_command']
    if fault == 'earlier_state': new['mission_receipt']['frame'] -= 1
    elif fault == 'earlier_raw': new['original_visual_evidence']['current_pose']['frame'] -= 1
    elif fault == 'actual_command': command = [1., 0., 0.]
    elif fault == 'failure': old['failure'] = 'unrelated'
    elif fault == 'boundary_raw': new['original_visual_evidence']['changed'] = True
    elif fault == 'boundary_pose': new['evidence']['current_pose']['frame'] -= 1
    elif fault == 'forecast': new['new_selection'].pop('prediction')
    elif fault == 'action': new['requested_command'] = [0., 0., -.45]
    elif fault == 'later': frame = 505
    if fault is None:
        report = comparison.compare_step(old, new, command, frame=frame)
        assert report['stop'] and report['controller_recovered'] and report['requested_command_changed']
    else:
        with pytest.raises(ValueError): comparison.compare_step(old, new, command, frame=frame)


def test_negative_boundary_is_an_outcome_and_does_not_require_recovery():
    old, new = rows(504); new = deepcopy(old)|{k:new[k] for k in ('controller', 'partial_floor_height_constraint_enabled')}
    report = comparison.compare_step(old, new, old['requested_command'], frame=504)
    assert report['stop'] and report['complete_original_decision_exact']
    assert not report['controller_recovered'] and not report['partial_height_admitted']


@pytest.mark.parametrize('fault', [None, 'anchor', 'serialized', 'correction'])
def test_live_partial_pose_and_anchor_are_validated_before_serialization(monkeypatch, fault):
    e, _, anchor, _, _, _, kw = repaired()
    live = dict(evidence=e, original_visual_evidence=e['original_visual_evidence'])
    monkeypatch.setattr(comparison, 'current_dual_camera_pose', lambda *a, **k:None)
    if fault == 'anchor': anchor = deepcopy(anchor); anchor['current_pose']['frame'] += 1
    elif fault == 'correction': e['partial_floor_height']['correction']['normal_height_increment_m'] += .001
    recorded = json.loads(json.dumps(live))
    if fault == 'serialized': recorded['changed'] = True
    args = (live, recorded, dict(boundary_reached=True, partial_height_admitted=True), None, None, None)
    if fault is None:
        comparison.validate_live(*args, now_ns=kw['now_ns'], prior_anchor=anchor)
    else:
        with pytest.raises(ValueError): comparison.validate_live(*args, now_ns=kw['now_ns'], prior_anchor=anchor)


@pytest.mark.parametrize('outcome', ['recovered', 'negative', 'earlier_change', 'input_mutation',
    'model_change', 'anchor_promotion', 'live_rejection', 'truncated'])
def test_replay_stops_at_boundary_and_preserves_failures(monkeypatch, tmp_path, outcome):
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path); weight = {'hash':runner.MODEL_STATE}
    model = SimpleNamespace(state_dict=lambda:dict(weight), parameters=lambda:[])
    monkeypatch.setattr(runner, 'load_assigned', lambda *a:(model, 'jepa', 'full'))
    monkeypatch.setattr(runner, 'state_digest', lambda d:d['hash'])
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:object())
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda *a:SimpleNamespace(free=2**40))
    consumed = []; packets = []
    def original_rows(*args):
        for i in range(505 if outcome != 'truncated' else 504):
            if outcome in ('earlier_change', 'input_mutation') and i > 5:
                pytest.fail('consumed a later observation after altered state')
            consumed.append(i)
            yield dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=rows(i)[0])
        if outcome != 'truncated': pytest.fail('consumed following physical outcome')
    monkeypatch.setattr(runner, 'read_rows', original_rows)
    def at(i):
        packets.append(i); return {'frame':i}, {}, {}, 1_500_000_000+i*100_000_000
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', lambda *a:SimpleNamespace(frames=[None]*515, packet=at))
    tape = [dict(requested_command=[0., 0., 0.], completed=True) for _ in range(514)]
    monkeypatch.setattr(runner, 'read_json', lambda p,n:tape if n == 'command_tape.json' else [{}]*515)
    monkeypatch.setattr(runner, 'packet', lambda *a, **k:({}, {}))
    monkeypatch.setattr(runner, 'public_acquisition', lambda x:x)
    def validate(*args, **kwargs):
        if outcome == 'live_rejection' and args[2]['boundary_reached']: raise ValueError('synthetic live rejection')
    monkeypatch.setattr(runner, 'validate_live', validate)
    class Controller:
        def __init__(self): self.registration = SimpleNamespace(anchor={'original':True})
        def observe(self, policy, *args, **kwargs):
            i = policy['frame']; old, new = rows(i)
            if i == 5:
                if outcome == 'earlier_change': new['mission_receipt']['changed'] = True
                elif outcome == 'input_mutation': policy['changed'] = True
                elif outcome == 'model_change': weight['hash'] = 'changed'
            if i == 504:
                if outcome == 'negative': new = old|{k:new[k] for k in ('controller', 'partial_floor_height_constraint_enabled')}
                elif outcome == 'anchor_promotion': self.registration.anchor = {'new':True}
            return new
    monkeypatch.setattr(runner, 'PartialHeightDirectFlowController', lambda *a, **k:Controller())
    if outcome in ('recovered', 'negative'):
        report = runner.replay({'correction_admission':{}})
        assert report['frames'] == 505 and report['prior_commands_compared'] == 504
        assert report['exact_original_decisions'] == (504 if outcome == 'recovered' else 505)
        assert report['raw_model_forecast_comparisons'] == 501
        assert report['full_controller_recovered_at_boundary'] == (outcome == 'recovered')
        assert packets == consumed == list(range(505)) and not report['following_recorded_observations_consumed']
    else:
        with pytest.raises(ValueError): runner.replay({'correction_admission':{}})
        if outcome in ('earlier_change', 'input_mutation', 'anchor_promotion', 'live_rejection'):
            from scripts.maze_decision_stream_development import read_rows
            saved = list(read_rows(tmp_path))
            assert 'comparison_failure' in saved[-1] and saved[-1]['tick'] == consumed[-1]

"""Comparison does not hide physical changes or consume a changed-action outcome."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import replay_go2_reactive_floor_transport_prefix_v1 as runner


def decision(*, old=False, command=None):
    return dict(evidence={'pose':[0.,0.,0.]}, original_visual_evidence={'fixture':True},
        memory_receipt={'retained_cells':10}, observed_goal_distance_m=1.,
        auxiliary_floor_partition_receipt={'returns':20},
        mission_receipt={'observed_settling':{'motion_source':
            'consecutive_admitted_floor_registered_visual_positions' if old else
            'consecutive_admitted_visual_positions_in_floor_reference', 'measured_motion_quiet':True}},
        requested_command=command or [0.,0.,0.], terminal=None, failure=None,
        learned_model_used=False, candidate_future_outcomes_evaluated=False, new_selection=None)


@pytest.mark.parametrize('fault', [None, 'pose', 'raw', 'map', 'distance', 'partition', 'settling', 'label'])
def test_only_exact_mission_source_wording_can_differ(fault):
    old = decision(old=True); new = decision(); saved = deepcopy(new)
    if fault == 'pose': new['evidence']['pose'][0] = .001
    elif fault == 'raw': new['original_visual_evidence']['fixture'] = False
    elif fault == 'map': new['memory_receipt']['retained_cells'] += 1
    elif fault == 'distance': new['observed_goal_distance_m'] += .001
    elif fault == 'partition': new['auxiliary_floor_partition_receipt']['returns'] -= 1
    elif fault == 'settling': new['mission_receipt']['observed_settling']['measured_motion_quiet'] = False
    elif fault == 'label': new['mission_receipt']['observed_settling']['motion_source'] = 'unvalidated'
    if fault is None:
        runner.compare_shared(new, old); assert new == saved
    else:
        with pytest.raises(ValueError): runner.compare_shared(new, old)


@pytest.mark.parametrize('boundary', ['command', 'terminal', 'limit'])
def test_replay_never_requests_observation_after_changed_command_or_fixed_limit(monkeypatch, tmp_path, boundary):
    limit = 4; last = 2 if boundary != 'limit' else limit-1
    packets = []; decision_reads = []
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'MAX_FRAMES', limit)
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:object())
    class Controller:
        def __init__(self, *a, **k): pass
        def observe(self, policy, *a, **k):
            result = decision()
            if policy['frame'] == 2 and boundary == 'command': result['requested_command'] = [0.,0.,.45]
            if policy['frame'] == 2 and boundary == 'terminal': result['terminal'] = 'VIEW_BUDGET_EXHAUSTED'
            return result
    monkeypatch.setattr(runner, 'ReactiveFloorTransportController', Controller)
    def get_packet(i):
        assert i <= last, 'read an observation after the prospective divergence'
        packets.append(i); return {'frame':i}, {}, {}, 1_500_000_000+100_000_000*i
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', lambda *a:SimpleNamespace(frames=range(6), packet=get_packet))
    monkeypatch.setattr(runner, 'packet', lambda *a, **k:({},{}))
    monkeypatch.setattr(runner, 'public_acquisition', lambda x:x)
    def read_json(root, name):
        if name == 'auxiliary_camera_audit.json': return [{}]*6
        return [dict(requested_command=[0.,0.,0.], completed=True) for _ in range(5)]
    monkeypatch.setattr(runner, 'read_json', read_json)
    def rows(root):
        for i in range(6):
            assert i <= last, 'read a later complete decision'
            decision_reads.append(i)
            yield dict(tick=i, pre_sample_index=749+50*i, decision=decision(old=True))
    monkeypatch.setattr(runner, 'read_rows', rows)
    report = runner.replay()
    assert packets == decision_reads == list(range(last+1))
    assert report['frames'] == last+1 and not report['following_recorded_observations_consumed']
    assert report['first_requested_command_difference'] == (2 if boundary == 'command' else None)
    assert report['first_terminal_policy_difference'] == (2 if boundary == 'terminal' else None)
    assert report['stopped_at_first_command_or_terminal_difference'] == (boundary != 'limit')

"""Admission and causal stream boundaries; synthetic controllers, no model claim."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import pytest

from scripts import replay_go2_measured_plane_controller_prefix_v1 as replay
from lewm.tests.test_measured_plane_controller_prefix_comparison_development import example


def positive():
    return dict(status='MEASURED_PLANE_OBSERVER_HISTORY_COMPLETION_VERIFIED',
        original_launch_sha256=replay.completed.LAUNCH_SHA, original_owner_ended=True,
        complete_output_stream_checked=True, actual_consumed_raw_packets_reconstructed=3838,
        native_completion_admitted=False, report=dict(frames=3838, planned_frames=3838,
            complete_planned_history=True, stop_reason='FIXED_HISTORY_END',
            candidate_failure_preserved=False, navigation_recovered=False,
            final_comparison=dict(candidate_visual_failure=None,candidate_floor_failure=None)))


def test_positive_observer_receipt_admitted():
    replay.require_positive_observer(positive())


@pytest.mark.parametrize('fault', ['short','failure','unverified','live','native','launch'])
def test_incomplete_or_wrong_observer_proof_rejected(fault):
    proof = positive()
    if fault == 'short': proof['report']['frames'] = 3837
    elif fault == 'failure': proof['report']['final_comparison']['candidate_floor_failure'] = 'gate'
    elif fault == 'unverified': proof['complete_output_stream_checked'] = False
    elif fault == 'live': proof['original_owner_ended'] = False
    elif fault == 'native': proof['native_completion_admitted'] = True
    elif fault == 'launch': proof['original_launch_sha256'] = '0'*64
    with pytest.raises(ValueError): replay.require_positive_observer(proof)


def endpoint(frame, decision):
    row = dict(tick=frame,observation_index=frame,pre_sample_index=749+50*frame,decision=decision)
    tape = dict(tick=frame,completed=True,pre_sample_index=749+50*frame,
        post_sample_index=799+50*frame,requested_command=decision['requested_command'])
    return row,tape


@pytest.mark.parametrize('fault', ['post','pre','command','completed'])
def test_recorded_action_requires_actual_physical_endpoints(fault):
    row,tape = endpoint(3,example()[0])
    replay.command_endpoint(row,tape,3)
    if fault == 'post': tape['post_sample_index'] += 1
    elif fault == 'pre': row['pre_sample_index'] += 1
    elif fault == 'command': tape['requested_command'] = [.2,0.,0.]
    else: tape['completed'] = False
    with pytest.raises(ValueError): replay.command_endpoint(row,tape,3)


@pytest.mark.parametrize('mutate_model', [False,True])
def test_replay_never_consumes_observation_after_first_changed_command(monkeypatch,tmp_path,mutate_model):
    """Poison the next stream element, and catch silent model changes at completion."""
    output = tmp_path/'output'; output.mkdir()
    (output/'context_decisions.jsonl.gz').touch()
    monkeypatch.setattr(replay,'OUTPUT',output)
    old,new,_,observer = example()
    old['tick'] = new['tick'] = 0
    new['requested_command'] = [.2,0.,0.]
    new['new_selection']['action'] = 'forward'
    recorded,tape = endpoint(0,old)
    observer.update(tick=0,comparison=dict(raw_packet_sha256='public'))
    consumed = []
    def rows(directory):
        consumed.append(str(directory))
        yield observer if directory == replay.run.OUTPUT else recorded
        raise AssertionError('consumed an outcome after a changed command')
    monkeypatch.setattr(replay.run.pipeline,'read_rows',rows)
    saved = []
    @contextmanager
    def writer(_): yield saved.append
    monkeypatch.setattr(replay.run.pipeline,'writer',writer)
    monkeypatch.setattr(replay.run,'read_json',lambda _,name:
        [tape] if name == 'command_tape.json' else [{}] if name == 'auxiliary_camera_audit.json' else {})
    monkeypatch.setattr(replay.run.pipeline,'ExtendedBudgetRGBDReplay',lambda _:SimpleNamespace(packet=lambda f:(1,2,3,4)))
    monkeypatch.setattr(replay.run.pipeline,'rgb_packet',lambda *a,**kw:(5,6))
    monkeypatch.setattr(replay.run,'public_acquisition',lambda v:v)
    monkeypatch.setattr(replay.run,'fingerprint',lambda _: 'public')
    models = []
    def model(_):
        m = SimpleNamespace(training=False, changed=False,parameters=lambda: [])
        m.state_dict = lambda: 'bad' if m.changed else replay.MODEL_SHA
        models.append(m)
        return m
    monkeypatch.setattr(replay.native,'assigned_model',model)
    monkeypatch.setattr(replay,'state_digest',lambda v:v)
    monkeypatch.setattr(replay,'ArticulatedCollisionGeometry',lambda _:object())
    monkeypatch.setattr(replay,'ResidualAnchoredContinuationController',lambda *a,**kw:
        SimpleNamespace(observe=lambda *a,**kw:deepcopy(old)))
    def candidate(model,*a,**kw):
        def observe(*a,**kw):
            model.changed = mutate_model
            return deepcopy(new)
        return SimpleNamespace(observe=observe)
    monkeypatch.setattr(replay,'MeasuredPlaneResidualController',candidate)
    if mutate_model:
        with pytest.raises(ValueError,match='states'): replay.replay()
    else:
        report = replay.replay()
        assert report['frames'] == 1 and report['boundary_comparison']['requested_command_changed']
        assert report['following_changed_command_outcome_consumed'] is False
        assert report['native_execution'] is False and report['navigation_recovered'] is False
    assert len(saved) == 1 and len(consumed) == 2 and len(models) == 2

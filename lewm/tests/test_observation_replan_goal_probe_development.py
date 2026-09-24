import numpy as np
from lewm.observation_replan_goal_probe_development import ObservationReplanGoalProbe
from lewm.overlap_retention_goal_probe_development import OverlapRetentionGoalProbe
from lewm import matched_model_goal_probe_development as core


class Selector:
    mode='WAYPOINT';condition='direct';variant='full'
    def __init__(self):self.calls=[]
    def choose(self,model,history,mapper,geometry,*,now_ns):
        self.calls.append(now_ns)
        return dict(action='forward' if len(self.calls)==1 else 'hold',view_budget_exhausted=False)


def controller(cls,monkeypatch,*,arrive=False):
    # Synthetic leaf inputs isolate the real mission clock, commitment, dwell
    # and failure logic. Native tests still use full public-packet admission.
    def pose(evidence,*,identity,now_ns):
        frame=(now_ns-1_500_000_000)//100_000_000
        return np.array([1.2 if arrive and frame>=4 else 0.,0.,0.]),np.eye(3),dict(frame=frame)
    monkeypatch.setattr(core,'current_joint_pose',pose)
    monkeypatch.setattr(core,'causal_history_tensors',lambda packets,now:dict(synthetic=True))
    c=cls(object(),object(),condition='direct',variant='full',persistent=True)
    c.selector=Selector();return c


def test_new_observation_can_replace_a_command_before_five_ticks(monkeypatch):
    old=controller(OverlapRetentionGoalProbe,monkeypatch)
    new=controller(ObservationReplanGoalProbe,monkeypatch)
    outputs=[]
    for c in (old,new):
        outputs.append([c.advance({},None,now_ns=1_500_000_000+i*100_000_000) for i in range(8)])
    assert len(old.selector.calls)==1 and len(new.selector.calls)==5
    assert outputs[0][3]['requested_command']==outputs[1][3]['requested_command']==[.2,0.,0.]
    assert outputs[0][4]['requested_command']==[.2,0.,0.]
    assert outputs[1][4]['requested_command']==[0.,0.,0.]
    assert all(r['plan_offset']==1 for r in outputs[1][3:])
    assert new.selector is not old.selector


def test_goal_dwell_and_bad_clock_still_stop_without_selection(monkeypatch):
    c=controller(ObservationReplanGoalProbe,monkeypatch,arrive=True)
    rows=[c.advance({},None,now_ns=1_500_000_000+i*100_000_000) for i in range(15)]
    assert rows[13]['terminal'] is None and rows[14]['terminal']=='OBSERVED_GOAL_CANDIDATE'
    assert rows[14]['quiet_intervals']==10 and len(c.selector.calls)==1
    assert all(r['requested_command']==[0.,0.,0.] for r in rows[4:])
    c=controller(ObservationReplanGoalProbe,monkeypatch)
    row=c.advance({},None,now_ns=1)
    assert row['terminal']=='SENSOR_OR_MODEL_FAILURE' and row['requested_command']==[0.,0.,0.]
    assert c.advance({},None,now_ns=1_500_000_000)['terminal']==row['terminal']
    assert not c.selector.calls


def test_native_goal_and_actuator_gate_functions_are_original_objects():
    from scripts import observation_replan_goal_audit_development as new
    from scripts import overlap_retention_goal_audit_development as old
    assert new.native_goal is old.native_goal and new.audit_commands is old.audit_commands

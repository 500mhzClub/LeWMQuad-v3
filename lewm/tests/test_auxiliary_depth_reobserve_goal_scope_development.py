import ast
from pathlib import Path
import numpy as np
from scripts import auxiliary_depth_reobserve_goal_episode_development as episode
from scripts import auxiliary_depth_goal_episode_development as prior_episode
from scripts import auxiliary_depth_reobserve_goal_audit_development as audit
from scripts import auxiliary_depth_goal_audit_development as prior_audit
from lewm.auxiliary_depth_reobserve_goal_probe_development import AuxiliaryDepthReobserveGoalProbe
from lewm.auxiliary_depth_goal_probe_development import AuxiliaryDepthGoalProbe
from lewm import matched_model_goal_probe_development as mission
from scripts import run_go2_auxiliary_depth_reobserve_goal_probe_v1 as runner
from scripts import run_go2_auxiliary_depth_goal_probe_v1 as prior_runner


def test_sensor_session_selector_and_native_auditors_unchanged():
    assert runner.CASES==prior_runner.CASES and runner.PREVIOUS==prior_runner.OUTPUT
    assert episode.AuxiliaryDepthGoalSession is prior_episode.AuxiliaryDepthGoalSession
    for name in ('native_goal','audit_commands','audit_sensors','audit_auxiliary','audit_rasters_and_footprints','audit_setup','audit_stops'):
        assert getattr(audit,name) is getattr(prior_audit,name)
    a=AuxiliaryDepthReobserveGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    b=AuxiliaryDepthGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    assert type(a.selector) is type(b.selector) and type(a.mapper) is type(b.mapper) and type(a.motion) is type(b.motion)
    assert AuxiliaryDepthReobserveGoalProbe.observe is AuxiliaryDepthGoalProbe.observe


def test_exact_original_collection_and_audit_except_controller_identity():
    for new,old,name in ((episode,prior_episode,'collect'),(audit,prior_audit,'audit')):
        source=Path(new.__file__).read_text().replace('AuxiliaryDepthReobserveGoalProbe(','AuxiliaryDepthGoalProbe(')
        source=source.replace('auxiliary_depth_reobserve_goal_probe_v1;','auxiliary_depth_goal_probe_v1;')
        source=source.replace('AUXILIARY_DEPTH_REOBSERVE_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED','AUXILIARY_DEPTH_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED')
        def body(s):
            return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
        assert body(source)==body(Path(old.__file__).read_text())


def test_real_inherited_mission_budget_still_ends_active_wait(monkeypatch):
    c=AuxiliaryDepthReobserveGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    c.tick=242;c.infeasible_wait_count=10;c.infeasible_wait_active=True
    monkeypatch.setattr(mission,'current_joint_pose',lambda *a,**k:(np.zeros(3),np.eye(3),dict(frame=243)))
    r=c.advance({},None,now_ns=1500000000+243*100000000)
    assert r['terminal']=='MISSION_TICK_BUDGET_EXHAUSTED' and not r['infeasible_wait_active']
    assert r['requested_command']==[0.,0.,0.] and r['new_selection'] is None


def test_real_inherited_arrival_dwell_still_ends_active_wait(monkeypatch):
    c=AuxiliaryDepthReobserveGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    c.tick=239;c.infeasible_wait_count=2;c.quiet=9;c.was_within_goal=True
    monkeypatch.setattr(mission,'current_joint_pose',lambda *a,**k:(np.array([1.2,0.,0.]),np.eye(3),dict(frame=240)))
    r=c.advance({},None,now_ns=1500000000+240*100000000)
    assert r['terminal']=='OBSERVED_GOAL_CANDIDATE' and not r['infeasible_wait_active']
    assert r['requested_command']==[0.,0.,0.] and r['quiet_intervals']==10

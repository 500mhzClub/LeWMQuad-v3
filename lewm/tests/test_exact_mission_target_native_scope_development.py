import ast
from pathlib import Path
from scripts import exact_mission_target_goal_episode_development as episode
from scripts import observed_floor_contact_goal_episode_development as prior_episode
from scripts import exact_mission_target_goal_audit_development as audit
from scripts import observed_floor_contact_goal_audit_development as prior_audit
from scripts import run_go2_exact_mission_target_goal_probe_v1 as runner
from scripts import run_go2_observed_floor_contact_goal_probe_v1 as prior_runner


def body(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_exact_native_collection_audit_and_goal_gates_except_target_controller():
    assert runner.CASES==prior_runner.CASES and runner.PREVIOUS==prior_runner.OUTPUT
    assert episode.AuxiliaryDownward45GoalSession is prior_episode.AuxiliaryDownward45GoalSession
    for name in ('native_goal','audit_commands','audit_sensors','audit_auxiliary','audit_rasters_and_footprints','audit_setup','audit_stops','auxiliary_packet'):
        assert getattr(audit,name) is getattr(prior_audit,name)
    for new,old,name in ((episode,prior_episode,'collect'),(audit,prior_audit,'audit')):
        source=Path(new.__file__).read_text().replace('ExactMissionTargetGoalProbe(','ObservedFloorContactGoalProbe(')
        source=source.replace('exact_mission_target_goal_probe_v1;','observed_floor_contact_goal_probe_v1;')
        source=source.replace('EXACT_MISSION_TARGET_GOAL_PROBE_TERMINAL','OBSERVED_FLOOR_CONTACT_GOAL_PROBE_TERMINAL')
        assert body(source,name)==body(Path(old.__file__).read_text(),name)
    assert body(Path(runner.__file__).read_text(),'causal_prefix')==body(Path(prior_runner.__file__).read_text(),'causal_prefix')
    assert body(Path(episode.__file__).read_text(),'artifacts')==body(Path(prior_episode.__file__).read_text(),'artifacts')

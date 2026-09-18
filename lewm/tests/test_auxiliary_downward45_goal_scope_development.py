import ast
from pathlib import Path
from scripts import auxiliary_downward45_goal_episode_development as episode
from scripts import auxiliary_depth_reobserve_goal_episode_development as prior_episode
from scripts import auxiliary_downward45_goal_audit_development as audit
from scripts import auxiliary_depth_reobserve_goal_audit_development as prior_audit
from scripts import auxiliary_downward45_goal_session_development as session
from scripts import auxiliary_depth_goal_session_development as prior_session
from scripts import auxiliary_downward45_sensor_audit_development as sensor_audit
from scripts import auxiliary_downward45_depth_capture_development as capture
from scripts import auxiliary_downward45_packet_replay_development as packet
from scripts import run_go2_auxiliary_downward45_goal_probe_v1 as runner
from scripts import run_go2_auxiliary_depth_reobserve_goal_probe_v1 as prior_runner


def body(source, name, kind=ast.FunctionDef):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,kind) and n.name==name))


def test_fixed_models_and_all_native_goal_actuator_and_stop_gates_retained():
    assert runner.CASES==prior_runner.CASES and runner.PREVIOUS==prior_runner.OUTPUT
    for name in ('native_goal','audit_commands','audit_sensors','audit_setup','audit_stops'):
        assert getattr(audit,name) is getattr(prior_audit,name)
    assert audit.audit_auxiliary is sensor_audit.audit_auxiliary
    assert audit.audit_rasters_and_footprints is sensor_audit.audit_rasters_and_footprints
    assert audit.auxiliary_packet is packet.packet
    assert episode.AuxiliaryDownward45GoalSession is session.AuxiliaryDownward45GoalSession
    assert session.capture is capture.capture and session.packet is packet.packet


def test_exact_collection_and_replay_control_flow_except_explicit_calibration_classes():
    for new,old,name in ((episode,prior_episode,'collect'),(audit,prior_audit,'audit')):
        source=Path(new.__file__).read_text().replace('AuxiliaryDownward45GoalProbe(','AuxiliaryDepthReobserveGoalProbe(')
        source=source.replace('AuxiliaryDownward45GoalSession(','AuxiliaryDepthGoalSession(')
        source=source.replace('auxiliary_downward45_goal_probe_v1;','auxiliary_depth_reobserve_goal_probe_v1;')
        source=source.replace('AUXILIARY_DOWNWARD45_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED','AUXILIARY_DEPTH_REOBSERVE_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED')
        assert body(source,name)==body(Path(old.__file__).read_text(),name)
    source=Path(session.__file__).read_text().replace('AuxiliaryDownward45GoalSession','AuxiliaryDepthGoalSession')
    assert body(source,'AuxiliaryDepthGoalSession',ast.ClassDef)==body(Path(prior_session.__file__).read_text(),'AuxiliaryDepthGoalSession',ast.ClassDef)


def test_warmup_comparison_and_artifact_retention_are_unchanged():
    assert body(Path(runner.__file__).read_text(),'causal_prefix')==body(Path(prior_runner.__file__).read_text(),'causal_prefix')
    assert body(Path(episode.__file__).read_text(),'artifacts')==body(Path(prior_episode.__file__).read_text(),'artifacts')

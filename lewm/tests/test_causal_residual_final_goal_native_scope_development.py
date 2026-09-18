import ast
from pathlib import Path
from scripts import causal_residual_final_goal_episode_development as episode
from scripts import executed_horizon_final_goal_episode_development as prior_episode
from scripts import causal_residual_final_goal_audit_development as audit
from scripts import executed_horizon_final_goal_audit_development as prior_audit
from scripts import run_go2_causal_residual_final_goal_probe_v1 as runner
from scripts import run_go2_executed_horizon_final_goal_probe_v1 as prior_runner


def body(source, name):
    return ast.dump(next(n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name))


def test_native_collection_and_all_goal_gates_change_only_controller_identity():
    assert runner.CASES == prior_runner.CASES and runner.PREVIOUS == prior_runner.OUTPUT
    assert episode.AuxiliaryDownward45GoalSession is prior_episode.AuxiliaryDownward45GoalSession
    for name in ('native_goal', 'audit_commands', 'audit_sensors', 'audit_auxiliary',
            'audit_rasters_and_footprints', 'audit_setup', 'audit_stops', 'auxiliary_packet'):
        assert getattr(audit, name) is getattr(prior_audit, name)
    for new, old, name in ((episode, prior_episode, 'collect'), (audit, prior_audit, 'audit')):
        source = Path(new.__file__).read_text().replace('CausalResidualFinalGoalProbe(', 'ExecutedHorizonFinalGoalProbe(')
        source = source.replace('causal_residual_final_goal_probe_v1;', 'executed_horizon_final_goal_probe_v1;')
        source = source.replace('CAUSAL_RESIDUAL_FINAL_GOAL_PROBE_TERMINAL', 'EXECUTED_HORIZON_FINAL_GOAL_PROBE_TERMINAL')
        assert body(source, name) == body(Path(old.__file__).read_text(), name)
    assert body(Path(runner.__file__).read_text(), 'causal_prefix') == body(Path(prior_runner.__file__).read_text(), 'causal_prefix')
    assert body(Path(episode.__file__).read_text(), 'artifacts') == body(Path(prior_episode.__file__).read_text(), 'artifacts')

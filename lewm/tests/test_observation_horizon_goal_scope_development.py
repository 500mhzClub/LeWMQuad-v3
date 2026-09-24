import ast
from pathlib import Path
from scripts import observation_horizon_goal_audit_development as current
from scripts import observation_replan_goal_audit_development as previous
from scripts import run_go2_observation_horizon_goal_probe_v1 as runner
from lewm.tests.test_observation_horizon_goal_selection_development import (
    test_candidate_forecasts_use_actual_eight_command_prefixes_and_short_clocks,
    test_surface_veto_and_nominal_radius_survive_shorter_scoring,
    test_old_forecast_cannot_enter_any_new_timed_guard,
    test_actual_observation_cadence_and_failure_stop_remain_inherited)


def test_fixed_models_and_original_native_goal_actuator_gates():
    assert current.native_goal is previous.native_goal and current.audit_commands is previous.audit_commands
    assert {c[4] for c in runner.CASES}=={'seed_2026091001_full_jepa','seed_2026091001_full_direct'}
    assert {c[1] for c in runner.CASES}=={'family_episode_039'}
    def body(module):
        text=Path(module.__file__).read_text().replace('ObservationHorizonGoalProbe(', 'ObservationReplanGoalProbe(')
        return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='audit'))
    assert body(current)==body(previous)

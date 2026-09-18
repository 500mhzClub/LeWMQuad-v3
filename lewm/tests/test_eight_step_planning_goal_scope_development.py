import ast
from pathlib import Path
from scripts import eight_step_planning_goal_audit_development as current
from scripts import observation_horizon_goal_audit_development as previous
from scripts import run_go2_eight_step_planning_goal_probe_v1 as runner
from lewm.tests.test_eight_step_planning_development import (
    test_later_segment_rejects_action_that_passed_first_and_final_endpoints,
    test_waypoint_scores_terminal_forecast_and_retains_surface_and_phase_vetoes,
    test_scan_keeps_existing_scores_and_all_eight_clock_checks_are_required,
    test_cadence_sensor_failure_and_arrival_contract_are_inherited)


def test_fixed_models_and_unchanged_original_native_audit():
    assert current.native_goal is previous.native_goal and current.audit_commands is previous.audit_commands
    assert {c[4] for c in runner.CASES}=={'seed_2026091001_full_jepa','seed_2026091001_full_direct'}
    assert {c[1] for c in runner.CASES}=={'family_episode_039'}
    assert runner.PREVIOUS.name=='go2_observation_horizon_goal_probe_v1_attempt_001'
    def body(module):
        source=Path(module.__file__).read_text().replace('EightStepPlanningGoalProbe(', 'ObservationHorizonGoalProbe(')
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='audit'))
    assert body(current)==body(previous)

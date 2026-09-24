import ast
from pathlib import Path
from scripts import training_bias_goal_audit_development as current
from scripts import eight_step_planning_goal_audit_development as previous
from scripts import run_go2_training_bias_goal_probe_v1 as runner
from lewm import training_bias_goal_probe_development as controller
from lewm.eight_step_planning_development import plan
from lewm.tests.test_training_bias_goal_selection_development import (
    test_corrected_selector_uses_exact_causal_forecast_transformation,
    test_route_view_and_scan_state_machine_are_unchanged,
    test_one_command_and_original_failure_arrival_contract_remain_inherited)


def test_fixed_assignments_and_original_path_native_actuator_gates():
    assert current.native_goal is previous.native_goal and current.audit_commands is previous.audit_commands
    assert controller.plan is plan
    assert {c[4] for c in runner.CASES}=={'seed_2026091001_full_jepa','seed_2026091001_full_direct'}
    assert {c[1] for c in runner.CASES}=={'family_episode_039'}
    assert runner.PREVIOUS.name=='go2_eight_step_planning_goal_probe_v1_attempt_001'
    def body(module):
        source=Path(module.__file__).read_text().replace('TrainingBiasGoalProbe(', 'EightStepPlanningGoalProbe(')
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='audit'))
    assert body(current)==body(previous)

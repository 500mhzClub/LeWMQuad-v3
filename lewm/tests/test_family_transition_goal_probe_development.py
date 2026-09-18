"""New model/observer composition preserves the frozen goal and command algorithm."""
import ast
from pathlib import Path
from lewm.family_transition_goal_probe_development import FamilyTransitionGoalProbe
from lewm.learned_goal_probe_development import LearnedGoalProbe
from lewm.corner_support_joint_observer_development import CornerSupportVisualLedMotion


def test_composition_changes_observer_and_metadata_only():
    probe = FamilyTransitionGoalProbe(object())
    assert isinstance(probe.motion, CornerSupportVisualLedMotion)
    assert FamilyTransitionGoalProbe.advance is LearnedGoalProbe.advance
    assert FamilyTransitionGoalProbe.observe is LearnedGoalProbe.observe
    row = probe._result([0., 0., 0.], None, None)
    assert row['controller'] == 'family_transition_corner_goal_probe_v1'
    assert row['goal_initial_body_xy_m'] == [1.2, 0.] and row['navigation_qualified'] is False


def test_exact_native_command_and_goal_audit_are_inherited():
    from scripts import family_transition_goal_audit_development as new
    from scripts import learned_goal_probe_audit_development as old
    def function(module, name):
        tree = ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ('audit_commands', 'native_goal'):
        assert function(new, name) == function(old, name)
    assert new.audit_sensors is old.audit_sensors and new.audit_stops is old.audit_stops

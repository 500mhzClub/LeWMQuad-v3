import ast
from pathlib import Path
from lewm.overlap_retention_goal_probe_development import OverlapRetentionGoalProbe
from lewm.retained_patch_contact_goal_probe_development import RetainedPatchContactGoalProbe
from lewm.overlap_retention_joint_observer_development import OverlapRetentionVisualLedMotion


def test_observer_adapter_preserves_mission_and_terminal_stop():
    c = OverlapRetentionGoalProbe(object(), object(), condition='direct', variant='full', persistent=True)
    assert isinstance(c.motion, OverlapRetentionVisualLedMotion)
    assert OverlapRetentionGoalProbe.advance is RetainedPatchContactGoalProbe.advance
    result = c.observe({}, {}, {}, now_ns=1)
    assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and result['requested_command'] == [0., 0., 0.]
    assert c.observe({}, {}, {}, now_ns=2)['terminal'] == result['terminal']
    assert c.memory.failed and result['floor_partition_receipt'] is None


def test_native_goal_and_actuator_audits_unchanged():
    from scripts import overlap_retention_goal_audit_development as new
    from scripts import retained_patch_contact_goal_audit_development as old
    def extract(module, name):
        tree = ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ('audit_commands', 'native_goal'):
        assert extract(new, name) == extract(old, name)

"""Check exact simulator/audit composition without launching a scene."""
import pytest
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from scripts import measured_plane_extended_maze_development as candidate
from scripts import extended_budget_anchored_maze_development as original


@pytest.mark.parametrize('name',['collect','audit'])
def test_complete_original_function_code_and_dependencies_retained(name):
    old,new = getattr(original,name),getattr(candidate,name)
    assert new is not old and new.__code__ is old.__code__
    assert new.__defaults__ == old.__defaults__ and new.__kwdefaults__ == old.__kwdefaults__
    assert new.__closure__ == old.__closure__
    assert new.__globals__ is not old.__globals__
    assert new.__globals__.keys() == old.__globals__.keys()
    changes = {k for k in old.__globals__ if new.__globals__[k] is not old.__globals__[k]}
    assert changes == {'ResidualAnchoredContinuationController'}
    assert new.__globals__['ResidualAnchoredContinuationController'] is MeasuredPlaneResidualController
    assert old.__globals__['ResidualAnchoredContinuationController'] is ResidualAnchoredContinuationController


def test_actual_extended_session_and_all_original_raw_audits_remain_bound():
    g = candidate.collect.__globals__
    assert g['RendererWitnessDualCameraMazeSession'] is original.ExtendedBudgetRendererSession
    assert g['writer'] is original.writer
    assert g['NAVIGATION_TICKS'] == 4000 and g['MAX_OBSERVATIONS'] == 4014
    a = candidate.audit.__globals__
    for key in ('audit_sensors','audit_commands','renderer_audit','read_rows','packet'):
        assert a[key] is original.audit.__globals__[key]
    assert a['IntentReturnRGBDReplay'] is original.ExtendedBudgetRGBDReplay
    assert candidate.artifacts is original.artifacts


def test_definition_records_estimator_change_and_simulation_limits():
    d = candidate.definition()
    assert d['implementation_class'] == 'MeasuredPlaneResidualController'
    assert d['navigation_ticks'] == 4000 and d['max_command_ticks'] == 4013
    assert d['measured_plane_constrained_estimator'] and d['original_floor_gate_unchanged']
    assert d['physics_paused_during_compute'] and d['complete_raw_controller_audit_retained']
    assert not any(d[k] for k in ('native_pose_input','navigation_qualified','real_time_qualified',
        'hardware_qualified','goal_achieved'))

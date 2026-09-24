"""Conflict ranking, explicit unknown semantics, and failure-stop regressions."""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from lewm.geometry_progress_predictive_selection_development import score_candidates
from lewm.surface_memory_candidate_filter_development import filter_selection
from lewm.surface_memory_goal_probe_development import SurfaceMemoryGoalProbe


class Memory:
    def __init__(self, conflicts):
        self.conflicts = conflicts
        self.calls = []

    def footprint(self, geometry, displacement, yaw, *, now_ns, persistent):
        self.calls.append((tuple(displacement), yaw, now_ns, persistent))
        return dict(possible_intersection=self.conflicts[len(self.calls)-1],
            free_space_established=False, motion_permitted=False)


def selection():
    p = np.zeros((6, 8, 5), np.float64)
    p[:, :, 3] = 1.
    p[:, :, 4] = -10.
    p[:, :, 0] = np.arange(6)[:, None]/10
    return score_candidates(p, goal_body_xy_m=[1.2, 0.], contact_penalty_m=1.2) | dict(prediction=p.tolist())


def test_conflict_removes_best_candidate_without_changing_utilities_or_predictions():
    original = selection(); saved = deepcopy(original)
    memory = Memory([False, False, False, False, False, True])
    result = filter_selection(original, memory, object(), now_ns=100, persistent=True)
    assert result['action_index'] == 4 and result['unfiltered_action'] == original['action']
    assert original == saved and result['prediction'] == original['prediction']
    assert result['candidates'] == original['candidates'] and len(memory.calls) == 6
    assert all(call[2:] == (100, True) for call in memory.calls)
    assert result['experimental_motion_without_clearance_certificate'] and not result['free_space_established']


def test_all_conflicts_explicitly_produce_no_action_and_zero_request():
    result = filter_selection(selection(), Memory([True]*6), object(), now_ns=100, persistent=False)
    assert result['action'] is None and result['action_index'] is None
    assert result['requested_command'] == [0., 0., 0.] and result['no_surface_conflict_candidates'] == 0


def test_no_conflicts_preserves_ranking_and_undefined_yaw_rejects():
    original = selection()
    result = filter_selection(original, Memory([False]*6), object(), now_ns=100, persistent=False)
    assert result['action_index'] == original['action_index']
    original['prediction'][0][0][3] = 0.
    with pytest.raises(ValueError, match='yaw'):
        filter_selection(original, Memory([False]*6), object(), now_ns=100, persistent=False)


def test_invalid_sensor_input_latches_zero_without_model_call_or_new_memory():
    probe = SurfaceMemoryGoalProbe(object(), object(), persistent=True)
    first = probe.observe({}, {}, {}, now_ns=1)
    second = probe.observe({}, {}, {}, now_ns=2)
    assert first['terminal'] == second['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert first['requested_command'] == second['requested_command'] == [0., 0., 0.]
    assert second['memory_receipt'] is None and second['evidence'] is None
    assert not probe.memory.route


def test_external_goal_and_exact_actuator_audits_remain_unchanged():
    from scripts import surface_memory_goal_audit_development as new
    from scripts import family_transition_goal_audit_development as old
    def function(module, name):
        tree = ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ('audit_commands', 'native_goal'):
        assert function(new, name) == function(old, name)

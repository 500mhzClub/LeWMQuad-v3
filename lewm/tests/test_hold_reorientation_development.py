from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.hold_reorientation_development import HoldReorientation
from lewm.hold_reorientation_controller_development import HoldReorientationSelector, HoldReorientationController
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationSelector, ResidualAnchoredContinuationController)


def selection(frame, **changes):
    s = dict(action='hold', action_index=0, requested_command=[0., 0., 0.], mode='WAYPOINT',
        view_budget_exhausted=False, intermediate_target_is_mission_goal=False,
        first_prediction_horizon_ns=100_000_000, prediction_horizon_ns=800_000_000,
        target_offsets_ns=list(range(100_000_000, 800_000_001, 100_000_000)),
        score_contract='causal_executed_waypoint_potential_minus_full_plan_contact',
        executed_waypoint_scoring=True, actual_commitment_horizon_ns=100_000_000,
        path_constraint_horizon_ns=800_000_000, native_state_used=False,
        prediction=np.zeros((6, 8, 5)).tolist(), phase_allowed_actions=list(ACTIONS),
        candidates=[dict(action=a, utility_m=u) for a, u in zip(ACTIONS, [0., -1., -2., -3., -.2, -.1])],
        surface_checks=[dict(possible_intersection=False) for _ in ACTIONS],
        nominal_path_checks=[dict(all_predicted_segments_nominally_clear=True) for _ in ACTIONS],
        causal_score_residual_receipt=dict(frame=frame, measured_ns=1_500_000_000+frame*100_000_000))
    return s | changes


def choose(state, frame, s=None):
    return state.reconsider(selection(frame) if s is None else s,
        frame=frame, now_ns=1_500_000_000+frame*100_000_000)


def ready():
    state = HoldReorientation()
    for frame in range(3, 13):
        s = selection(frame)
        assert choose(state, frame, s) is s
    return state


def test_first_change_requires_ten_prior_holds_and_retains_all_original_evidence():
    state = ready(); s = selection(13); before = deepcopy(s)
    result = choose(state, 13, s)
    assert s == before and result is not s
    assert result['action'] == 'right_turn' and result['requested_command'] == [0., 0., -.45]
    assert result['hold_reorientation']['preceding_discretionary_holds'] == 10
    restored = deepcopy(result)
    restored.pop('hold_reorientation')
    for k in ('action', 'action_index', 'requested_command'): restored[k] = s[k]
    assert restored == s
    assert state.holds == 0
    # No automatic continuation, even when the same candidate stays preferable.
    assert choose(state, 14)['action'] == 'hold'


@pytest.mark.parametrize('gate', ['phase', 'surface', 'path'])
def test_each_original_veto_blocks_the_preferred_turn(gate):
    state = ready(); s = selection(13)
    if gate == 'phase': s['phase_allowed_actions'].remove('right_turn')
    if gate == 'surface': s['surface_checks'][5]['possible_intersection'] = True
    if gate == 'path': s['nominal_path_checks'][5]['all_predicted_segments_nominally_clear'] = False
    r = choose(state, 13, s)
    assert r['action'] == 'left_turn' and r['hold_reorientation']['eligible_turns'] == ['left_turn']


def test_no_eligible_turn_preserves_hold_even_when_translation_is_available():
    state = ready(); s = selection(13)
    s['phase_allowed_actions'] = ['hold', 'forward']
    assert choose(state, 13, s) is s and state.holds == 10
    # A fresh observation can later admit a turn; nothing was precommitted.
    assert choose(state, 14)['action'] == 'right_turn'


@pytest.mark.parametrize('changes', [dict(action='left_arc'), dict(action=None),
    dict(mode='VIEW_ACQUISITION'), dict(view_budget_exhausted=True),
    dict(intermediate_target_is_mission_goal=True), dict(nominal_clearance_reentry=True),
    dict(residual_first_interval_feasibility={}), dict(residual_hold_feasibility={}),
    dict(residual_anchored_continuation={})])
def test_existing_special_case_passes_through_and_resets_credit(changes):
    state = ready(); s = selection(13, **changes)
    assert choose(state, 13, s) is s and state.holds == 0
    assert choose(state, 14)['action'] == 'hold'


def test_gap_or_goal_change_cannot_reuse_prior_hold_credit():
    state = ready()
    assert choose(state, 14)['action'] == 'hold' and state.holds == 1
    state = ready(); state.reset_goal()
    assert choose(state, 13)['action'] == 'hold' and state.holds == 1


@pytest.mark.parametrize('frame', [12, 11, True])
def test_duplicate_backwards_and_boolean_frames_rejected(frame):
    with pytest.raises(ValueError, match='clock'): choose(ready(), frame)


def test_misdated_receipt_and_winning_nonhold_are_rejected():
    s = selection(13); s['causal_score_residual_receipt']['frame'] = 12
    with pytest.raises(ValueError, match='current original'): choose(ready(), 13, s)
    s = selection(13); s['candidates'][1]['utility_m'] = .1
    with pytest.raises(ValueError, match='hold must win'): choose(ready(), 13, s)


def test_exact_model_rank_stable_tie_and_invalid_prediction():
    s = selection(13); s['candidates'][4]['utility_m'] = -.1
    assert choose(ready(), 13, s)['action'] == 'left_turn'
    s = selection(13); s['prediction'][0][0][0] = float('nan')
    with pytest.raises(ValueError, match='current original'): choose(ready(), 13, s)


def test_selector_calls_entire_original_chain_and_observed_frame_guard(monkeypatch):
    residual = SimpleNamespace(frame=13)
    selector = HoldReorientationSelector(residual=residual, condition='jepa', variant='full',
        goal_initial_body_xy_m=[3., 2.])
    selector.hold_reorientation = ready()
    calls = []
    def original(self, model, history, mapper, geometry, *, now_ns):
        calls.append((model, history, mapper, geometry, now_ns)); return selection(13)
    monkeypatch.setattr(ResidualAnchoredContinuationSelector, 'choose', original)
    mapper = SimpleNamespace(failed=False, surface=SimpleNamespace(failed=False,
        last_ns=2_800_000_000, route=[None]*14))
    model, history, geometry = object(), object(), object()
    assert selector.choose(model, history, mapper, geometry, now_ns=2_800_000_000)['action'] == 'right_turn'
    assert calls == [(model, history, mapper, geometry, 2_800_000_000)]
    mapper.surface.failed = True
    with pytest.raises(ValueError, match='same admitted'):
        selector.choose(model, history, mapper, geometry, now_ns=2_800_000_000)


def test_new_goal_resets_only_target_specific_hold_state_and_execution_is_inherited():
    residual = object()
    selector = HoldReorientationSelector(residual=residual, condition='jepa', variant='full',
        goal_initial_body_xy_m=[3., 2.])
    selector.hold_reorientation = ready()
    selector.set_goal([3., 2.]); assert selector.hold_reorientation.holds == 10
    selector.set_goal([0., 0.]); assert selector.hold_reorientation.holds == 0
    assert selector.residual is residual and selector.hold_reorientation.last_frame == 12
    assert HoldReorientationController.advance is ResidualAnchoredContinuationController.advance
    assert HoldReorientationController.observe is ResidualAnchoredContinuationController.observe

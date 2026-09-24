"""Preserve recovery paths, complete decisions and a strict counterfactual stop."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from lewm.commitment_contact_score_development import score_commitment_contact
from lewm.commitment_contact_anchored_controller_development import (
    CommitmentContactAnchoredController, CommitmentContactAnchoredSelector,
    ordinary_commitment_contact, CONTROLLER, FLAG, RECOVERY_FIELDS)
from lewm.commitment_contact_anchored_prefix_development import PrefixComparison
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.tests.test_commitment_contact_development import selection, decisions


def test_original_contact_scorer_and_anchored_controller_bytes_are_retained():
    bindings = {
        'lewm/commitment_contact_score_development.py':
            'cb51e94a31fc748592212811a7a7085356fd9aab3c4bae6653b41a18dda562b1',
        'lewm/residual_anchored_continuation_controller_development.py':
            'f35ff18c78b4db81c0c3c766eed1015823bf972d484600907954941ef7fb946a',
    }
    for name, expected in bindings.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected


def test_ordinary_waypoint_uses_exact_existing_cost_and_geometry_checks():
    old = selection(); before = deepcopy(old)
    new = ordinary_commitment_contact(old)
    assert new == score_commitment_contact(old)
    assert old == before and old['action'] == 'hold' and new['action'] == 'forward'
    assert new['scored_contact_horizon_ns'] == 100_000_000
    assert new['path_constraint_horizon_ns'] == 800_000_000
    for key in ('prediction', 'surface_checks', 'nominal_path_checks', 'phase_allowed_actions'):
        assert new[key] == old[key]


@pytest.mark.parametrize('name', RECOVERY_FIELDS)
def test_active_original_recoveries_are_returned_by_identity(name):
    old = selection(); old[name] = {'executed_recovery_witness': [1, 2, 3]}
    before = deepcopy(old)
    assert ordinary_commitment_contact(old) is old and old == before


@pytest.mark.parametrize('patch', [dict(mode='VIEW_ACQUISITION'),
    dict(intermediate_target_is_mission_goal=True), dict(nominal_clearance_reentry=True)])
def test_view_final_goal_and_nominal_reentry_keep_original_behavior(patch):
    old = selection() | patch
    assert ordinary_commitment_contact(old) is old


def test_original_observation_and_failure_decision_remain_exact():
    options = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
        require_return_after_goal=True), navigation_ticks=40, condition='supervised_rollout', variant='full', persistent=True)
    a = ResidualAnchoredContinuationController(object(), object(), **options)
    b = CommitmentContactAnchoredController(object(), object(), **options)
    assert b.selector.residual is b.residual and b.memory is b.mapper.surface
    assert type(b.selector) is CommitmentContactAnchoredSelector
    for name in ('observe', 'advance'):
        assert getattr(type(b), name) is getattr(type(a), name)
    old = a.observe({}, {}, {}, now_ns=1); new = b.observe({}, {}, {}, now_ns=1)
    assert old['terminal'] == new['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    new.pop(FLAG); new['controller'] = old['controller']
    assert old == new


def pair():
    old, new = decisions()
    old['controller'] = 'residual_anchored_continuation_controller_v1'
    new.pop('commitment_contact_policy_enabled')
    new.update(controller=CONTROLLER, **{FLAG: True})
    return old, new


@pytest.mark.parametrize('fault', [None, 'evidence', 'residual', 'prediction', 'constraint',
    'wrong_command', 'order', 'objective', 'identity'])
def test_complete_prefix_comparison_and_first_changed_command_stop(fault):
    old, new = pair(); comparison = PrefixComparison()
    if fault == 'evidence': new['evidence']['observed'] = -1
    elif fault == 'residual': new['causal_residual_receipt']['pending_forecast_tick'] = -1
    elif fault == 'prediction': new['new_selection']['prediction'][0][0][0] += .001
    elif fault == 'constraint': new['new_selection']['surface_checks'][0]['possible_intersection'] = True
    elif fault == 'wrong_command': new['requested_command'] = [1., 0., 0.]
    elif fault == 'objective': new['model_condition'] = 'jepa'
    elif fault == 'identity': new['controller'] = 'other'
    frame = 1 if fault == 'order' else 0
    if fault:
        with pytest.raises(ValueError): comparison.compare(old, new, old['requested_command'], frame=frame)
    else:
        result = comparison.compare(old, new, old['requested_command'], frame=frame)
        assert result['stop'] and result['requested_command_changed']
        assert comparison.first_command_difference == 0
        with pytest.raises(ValueError): comparison.compare(old, new, old['requested_command'], frame=1)


def test_warmup_does_not_claim_a_forecast_or_consume_later_counterfactual_data():
    old, new = pair()
    for d in (old, new):
        d.update(new_selection=None, requested_command=[0., 0., 0.], selected_action=None)
    comparison = PrefixComparison()
    result = comparison.compare(old, new, old['requested_command'], frame=0)
    assert not result['stop'] and not result['raw_model_forecasts_compared']
    assert not result['ordinary_contact_cost_horizon_changed']
    assert not result['requested_command_changed'] and comparison.next_frame == 1

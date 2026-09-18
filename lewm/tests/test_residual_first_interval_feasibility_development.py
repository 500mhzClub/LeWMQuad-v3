"""Synthetic geometry and causal-state checks; no navigation success claim."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.online_executed_residual_development import OnlineExecutedResidual
from lewm.observation_horizon_predictive_selection_development import score_candidates
from lewm.observation_horizon_surface_filter_development import filter_selection
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.residual_first_interval_feasibility_development import causal_correction, correct_first_interval_feasibility

NOW = 2_500_000_000


class Surface:
    def __init__(self):
        self.position = np.array([-.005, 0., .3])
        self.rotation = np.eye(3)
        self.last_ns = NOW
        self.route = [None]*11
        self.failed = False
        self.calls = []
        self.block = False

    def footprint(self, geometry, xy, yaw, *, now_ns, persistent):
        self.calls.append(dict(xy=list(xy), yaw=yaw, now_ns=now_ns, persistent=persistent))
        return dict(possible_intersection=self.block)


def fixture(*, late_collision=False, bias=.01):
    residual = OnlineExecutedResidual()
    residual.frame = 10; residual.now_ns = NOW
    residual.pose = dict(position=[0., 0., .3], rotation=np.eye(3).tolist(),
                         rgb_sha256='a'*64, depth_sha256='b'*64)
    for tick in range(2, 10):
        residual.history.append(dict(tick=tick, available_tick=tick+1,
            measured_ns=1_500_000_000+(tick+1)*100_000_000,
            predicted_body_xy_m=[bias, 0.], observed_body_xy_m=[0., 0.], residual_xy_m=[bias, 0.]))
    receipt = residual.snapshot()
    mapper = SimpleNamespace(surface=Surface(), map_from_initial=np.eye(3),
                             occupied={(9, -1), (9, 0)}, failed=False)
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.; p[:, :, 4] = -5.
    p[:, 0, 0] = .01
    p[1, :, 4] = -10.  # Forward wins the original corrected 100ms score.
    p[2, -1, 0] = -.1  # Left arc wins incidental 800ms scoring toward negative X.
    if late_collision: p[:, 4, 0] = .02
    s = score_candidates(p, goal_body_xy_m=[-1., 0.], contact_penalty_m=1.2)
    s.update(prediction=p.tolist(), first_prediction_horizon_ns=100_000_000,
        target_offsets_ns=list(range(100_000_000, 800_000_001, 100_000_000)),
        mode='WAYPOINT', phase_allowed_actions=list(ACTIONS), phase_admissible_candidates=6,
        intermediate_target_is_mission_goal=False)
    s = filter_selection(s, mapper.surface, object(), now_ns=NOW, persistent=True)
    s = constrain(s, mapper.surface.position, np.eye(3), mapper.occupied)
    s = plan(s, mapper.surface.position, np.eye(3), mapper.occupied)
    s = score_waypoint_execution(s, receipt)
    assert s['action'] is None
    mapper.surface.calls.clear()
    return s, receipt, mapper, residual


def apply(s, receipt, mapper):
    return correct_first_interval_feasibility(s, receipt, mapper, object(), now_ns=NOW)


def test_rechecks_corrected_geometry_preserves_raw_targets_and_short_score():
    s, receipt, mapper, memory = fixture(); original = deepcopy(s)
    r = apply(s, receipt, mapper)
    assert r['action'] == 'forward' and r['requested_command'] == candidate_commands('forward')[0]
    assert s == original
    for key in ('prediction', 'candidates', 'surface_checks', 'nominal_action_checks', 'nominal_path_checks'):
        assert r[key] == original[key]
    check = r['residual_first_interval_feasibility']
    assert check['eligible_actions'] == list(ACTIONS)
    assert all(not x['nominal_disk_connector_clear'] for x in r['nominal_action_checks'])
    for i, path in enumerate(check['corrected_nominal_path_checks']):
        assert path['all_predicted_segments_nominally_clear']
        assert len(path['segments']) == 8
        # The first correction changes segment 0's end AND segment 1's start.
        assert path['segments'][0]['predicted_end_map_xy_m'] == [-.005, 0.]
        assert path['segments'][1]['predicted_start_map_xy_m'] == [-.005, 0.]
        for h in range(1, 8):
            assert path['segments'][h]['predicted_end_map_xy_m'] == original['nominal_path_checks'][i]['segments'][h]['predicted_end_map_xy_m']
    assert len(mapper.surface.calls) == 6
    assert all(c == dict(xy=[0., 0.], yaw=0., now_ns=NOW, persistent=True) for c in mapper.surface.calls)
    memory.remember(dict(tick=10, terminal=None, new_selection=r, requested_command=r['requested_command']))
    assert memory.pending['predicted_body_xy_m'] == [.01, 0.]
    assert check['corrected_first_body_xy_m'][1] == [0., 0.]


def test_later_collision_still_blocks_every_action():
    s, receipt, mapper, _ = fixture(late_collision=True)
    r = apply(s, receipt, mapper)
    assert r['action'] is None
    for path in r['residual_first_interval_feasibility']['corrected_nominal_path_checks']:
        assert path['segments'][0]['nominal_disk_connector_clear']
        assert not path['segments'][4]['nominal_disk_connector_clear']
        assert not path['all_predicted_segments_nominally_clear']


@pytest.mark.parametrize('veto', ['original_surface', 'corrected_surface', 'phase'])
def test_surface_and_phase_vetoes_are_retained(veto):
    s, receipt, mapper, _ = fixture()
    if veto == 'original_surface':
        for c in s['surface_checks']: c['possible_intersection'] = True
    elif veto == 'corrected_surface': mapper.surface.block = True
    else: s['phase_allowed_actions'] = ['right_turn']
    r = apply(s, receipt, mapper)
    assert r['action'] == ('right_turn' if veto == 'phase' else None)


@pytest.mark.parametrize('patch', [dict(action='hold'), dict(mode='VIEW_ACQUISITION'),
    dict(view_budget_exhausted=True), dict(intermediate_target_is_mission_goal=True),
    dict(nominal_clearance_reentry=True)])
def test_existing_feasible_and_special_policies_are_unchanged(patch):
    s, receipt, mapper, _ = fixture(); s.update(patch)
    assert apply(s, receipt, mapper) is s
    assert mapper.surface.calls == []


def test_zero_correction_and_current_clearance_violation_do_not_activate():
    s, receipt, mapper, _ = fixture(bias=0.)
    assert apply(s, receipt, mapper) is s
    s, receipt, mapper, _ = fixture(); mapper.surface.position[0] = .01
    assert apply(s, receipt, mapper) is s
    assert mapper.surface.calls == []


def test_empty_history_is_inactive_and_pre_episode_residual_is_rejected():
    s, _, mapper, memory = fixture()
    memory.history.clear()
    assert apply(s, memory.snapshot(), mapper) is s
    assert mapper.surface.calls == []
    memory.frame = 0; memory.now_ns = 1_500_000_000
    memory.history.append(dict(tick=-1, available_tick=0, measured_ns=memory.now_ns,
        predicted_body_xy_m=[.01, 0.], observed_body_xy_m=[0., 0.], residual_xy_m=[.01, 0.]))
    with pytest.raises(ValueError): causal_correction(memory.snapshot(), now_ns=memory.now_ns)


@pytest.mark.parametrize('fault', ['future', 'duplicate', 'availability', 'clock', 'bias',
    'privileged', 'integrated', 'weights', 'arithmetic', 'nonfinite', 'pending'])
def test_invalid_residual_evidence_is_rejected(fault):
    _, receipt, _, _ = fixture()
    if fault == 'future': receipt['residuals'][-1]['tick'] = 10
    elif fault == 'duplicate': receipt['residuals'][-1]['tick'] = 8
    elif fault == 'availability': receipt['residuals'][-1]['available_tick'] = 11
    elif fault == 'clock': receipt['measured_ns'] -= 100_000_000
    elif fault == 'bias': receipt['correction_xy_m'][0] += .001
    elif fault == 'privileged': receipt['native_outcomes_used'] = True
    elif fault == 'integrated': receipt['command_integrated_pose_used'] = True
    elif fault == 'weights': receipt['model_weights_changed'] = True
    elif fault == 'arithmetic': receipt['residuals'][0]['observed_body_xy_m'][0] = .001
    elif fault == 'nonfinite': receipt['residuals'][0]['residual_xy_m'][0] = float('nan')
    else: receipt['pending_forecast_tick'] = 10
    with pytest.raises(ValueError): causal_correction(receipt, now_ns=NOW)


@pytest.mark.parametrize('fault', ['raw_path', 'raw_first', 'score_xy', 'surface_clock', 'map_failed'])
def test_stale_or_inconsistent_selection_is_rejected(fault):
    s, receipt, mapper, _ = fixture()
    if fault == 'raw_path': s['nominal_path_checks'][0]['segments'][0]['radius_m'] = .4
    elif fault == 'raw_first': s['nominal_action_checks'][0]['radius_m'] = .4
    elif fault == 'score_xy': s['candidates'][0]['causal_scoring_body_xy_m'][0] = .1
    elif fault == 'surface_clock': mapper.surface.last_ns -= 100_000_000
    else: mapper.failed = True
    with pytest.raises(ValueError): apply(s, receipt, mapper)


def test_controller_keeps_observer_and_residual_owner_and_latches_sensor_failure():
    from lewm.residual_first_interval_controller_development import ResidualFirstIntervalController, ResidualFirstIntervalSelector
    from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
    assert ResidualFirstIntervalController.observe is MeasuredFloorTransportController.observe
    assert ResidualFirstIntervalController.advance is MeasuredFloorTransportController.advance
    c = ResidualFirstIntervalController(object(), object(), public_mission=dict(
        goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    assert isinstance(c.selector, ResidualFirstIntervalSelector) and c.selector.residual is c.residual
    r = c.observe({}, {}, {}, now_ns=1)
    assert r['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and r['requested_command'] == [0., 0., 0.]
    before = c.residual.snapshot()
    assert c.advance({}, {}, now_ns=2)['terminal'] == r['terminal']
    assert c.residual.snapshot() == before

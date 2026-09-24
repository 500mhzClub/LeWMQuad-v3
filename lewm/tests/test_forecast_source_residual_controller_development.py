"""Real unchanged planning functions consume the declared source in each mode."""
from copy import deepcopy
from functools import partial
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

from lewm import forecast_source_residual_controller_development as candidate
from lewm.forecast_source_selection_development import SOURCES
from lewm.mission_target_eight_step_selection_development import MissionTargetEightStepSelector
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationSelector
from lewm.observed_floor_waypoint_development import segment_cells
from lewm.tests.test_forecast_source_selection_development import FixedModel, history
from lewm.tests.test_residual_first_interval_feasibility_development import fixture as residual_fixture, NOW


def case(phase='waypoint', block=False, occupied=()):
    _, _, mapper, residual = residual_fixture(bias=0.)
    mapper.surface.position = np.array([0., 0., .3]); mapper.surface.block = block
    mapper.occupied = set(occupied)
    goal = [.2, 0.] if phase == 'final' else [1., 0.]
    mapper.floor = segment_cells([0., 0.], goal)
    proposal = dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL' if phase == 'final' else 'OBSERVED_ROUTE',
        route_cells=[] if phase == 'view' else [[4, 0]], unknown_connector_cells=[])
    mapper.waypoint = lambda *args, **kwargs: deepcopy(proposal)
    kwargs = dict(residual=residual, condition='jepa', variant='full', goal_initial_body_xy_m=goal)
    return mapper, residual, kwargs


@pytest.mark.parametrize('phase', ['waypoint', 'final', 'view'])
@pytest.mark.parametrize('block', [False, True])
def test_learned_mode_preserves_complete_original_selection(phase, block):
    mapper, residual, kwargs = case(phase, block)
    old = ResidualAnchoredContinuationSelector(**kwargs)
    new = candidate.ForecastSourceResidualSelector(forecast_source='frozen_world_model', **kwargs)
    model = FixedModel(); h = history()
    expected = old.choose(model, h, mapper, object(), now_ns=NOW)
    actual = new.choose(model, h, mapper, object(), now_ns=NOW)
    provenance = actual.pop('forecast_provenance')
    assert actual == expected
    assert provenance['learned_forecasts_used']
    assert new.residual is residual
    assert (new.scan_sign, new.scan_index, new.scan_target, new.mode) == (
        old.scan_sign, old.scan_index, old.scan_target, old.mode)


@pytest.mark.parametrize('phase', ['waypoint', 'final', 'view'])
@pytest.mark.parametrize('block', [False, True])
def test_nominal_source_survives_all_scoring_and_recovery_layers(phase, block):
    mapper, _, kwargs = case(phase, block)
    selector = candidate.ForecastSourceResidualSelector(forecast_source='nominal_requested_twist', **kwargs)
    model = FixedModel(); before = deepcopy(kwargs['residual'].snapshot())
    result = selector.choose(model, history(), mapper, object(), now_ns=NOW)
    assert not model.calls
    assert result['head'] == 'nominal_requested_twist'
    assert not result['model_prediction_corrected'] and not result['translation_bias_training_only']
    assert result['forecast_provenance']['nominal_requested_twist_forecasts_used']
    assert result['planned_segment_count'] == 8 and result['actual_commitment_horizon_ns'] == 100_000_000
    if phase == 'final': assert result['final_goal_execution_horizon_scoring']
    elif phase == 'waypoint': assert result['executed_waypoint_scoring']
    else: assert result['phase_allowed_actions'] == ['hold', 'left_turn', 'right_turn']
    if block: assert result['action'] is None and result['requested_command'] == [0., 0., 0.]
    assert kwargs['residual'].snapshot() == before


def test_waypoint_forecast_binding_never_mutates_original_module_globals():
    from lewm import mission_target_waypoint_selection_development as original
    function = original.MissionTargetWaypointSelector.choose
    before = function.__globals__.copy()
    mapper, _, kwargs = case()
    selector = candidate.ForecastSourceResidualSelector(forecast_source='nominal_requested_twist', **kwargs)
    selector.choose(FixedModel(), history(), mapper, object(), now_ns=NOW)
    assert all(function.__globals__[k] is v for k, v in before.items())
    with pytest.raises(AttributeError): selector.forecast_source = 'frozen_world_model'


def test_cooperative_order_keeps_all_original_recovery_methods():
    from lewm.observed_round_trip_controller_development import RoundTripMissionSelector
    from lewm.view_reentry_round_trip_controller_development import ViewReentrySelector
    from lewm.residual_first_interval_controller_development import ResidualFirstIntervalSelector
    cls = candidate.ForecastSourceResidualSelector
    mro = cls.__mro__
    expected = [ResidualAnchoredContinuationSelector, ResidualFirstIntervalSelector,
        ViewReentrySelector, RoundTripMissionSelector, candidate.ForecastSourceEightStepSelector,
        MissionTargetEightStepSelector, candidate.ForecastSourceWaypointSelector]
    assert [mro.index(kind) for kind in expected] == sorted(mro.index(kind) for kind in expected)
    assert cls.choose is ResidualAnchoredContinuationSelector.choose


@pytest.mark.parametrize('phase', ['view', 'waypoint'])
@pytest.mark.parametrize('mode', SOURCES)
def test_positive_clearance_reentry_uses_the_declared_forecast(phase, mode):
    mapper, _, kwargs = case(phase, occupied=((-9, 0),))
    selector = candidate.ForecastSourceResidualSelector(forecast_source=mode, **kwargs)
    model = FixedModel()
    result = selector.choose(model, history(), mapper, object(), now_ns=NOW)
    assert result['nominal_clearance_reentry'] and result['action'] == 'forward'
    assert result['original_nominal_path_veto_preserved'] and not result['reentry_guaranteed']
    if phase == 'view':
        assert result['view_reentry_translation_enabled']
        assert result['selected_action_requires_view_phase_exception']
        assert result['phase_allowed_actions'] == ['hold', 'left_turn', 'right_turn']
    assert result['model_prediction_corrected'] == (mode == 'frozen_world_model')
    assert len(model.calls) == int(mode == 'frozen_world_model')


@pytest.mark.parametrize('mode', SOURCES)
def test_anchored_hold_recovery_preserves_raw_forecasts_and_their_source(mode):
    mapper, _, kwargs = case('waypoint', occupied=((12, 0),))
    mapper.surface.position[1] = .025
    residual = residual_fixture(bias=.02)[3]
    kwargs['residual'] = residual
    # Synthetic footprint geometry vetoes rotation but permits straight
    # translation. The obstacle blocks the nominal forward endpoint; the
    # observed residual shift can clear the entire eight-step path.
    original_footprint = mapper.surface.footprint
    def footprint(geometry, xy, yaw, **options):
        result = original_footprint(geometry, xy, yaw, **options)
        return result | {'possible_intersection': abs(yaw) > 1e-12}
    mapper.surface.footprint = footprint
    selector = candidate.ForecastSourceResidualSelector(forecast_source=mode, **kwargs)
    model = FixedModel()
    result = selector.choose(model, history(), mapper, object(), now_ns=NOW)
    assert result['action'] == 'forward' and 'residual_anchored_continuation' in result
    check = result['residual_anchored_continuation']
    assert check['corrected_nominal_path_checks'][1]['all_predicted_segments_nominally_clear']
    assert not result['nominal_path_checks'][1]['all_predicted_segments_nominally_clear']
    assert result['forecast_provenance']['forecast_source'] == mode
    assert result['model_prediction_corrected'] == (mode == 'frozen_world_model')
    residual.remember(dict(tick=10, terminal=None, new_selection=result,
        requested_command=result['requested_command']))
    assert residual.pending['predicted_body_xy_m'] == result['prediction'][1][0][:2]
    assert residual.pending['predicted_body_xy_m'] != check['corrected_first_body_xy_m'][1]
    assert len(model.calls) == int(mode == 'frozen_world_model')


@pytest.mark.parametrize('mode', SOURCES)
def test_exhausted_view_budget_does_not_invent_a_forecast(mode):
    mapper, _, kwargs = case('view')
    selector = candidate.ForecastSourceResidualSelector(forecast_source=mode, **kwargs)
    selector.scan_sign = 1; selector.scan_index = 4; selector.scan_target = 0.
    model = FixedModel()
    result = selector.choose(model, history(), mapper, object(), now_ns=NOW)
    assert result['view_budget_exhausted'] and result['action'] is None
    assert 'prediction' not in result and 'forecast_provenance' not in result
    assert not model.calls


@pytest.mark.parametrize('mode', SOURCES)
def test_four_public_packets_reach_forecast_selection_with_synthetic_pose(monkeypatch, mode):
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.tests import test_joint_pulse_execution_development as pulse
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb, depth_digest
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    monkeypatch.setattr(pulse, 'visual', partial(visual, origin=1_500_000_000))
    # The old two-frame fixture deliberately raises the body by 20 mm per
    # frame. Keep the synthetic floor height fixed for this four-frame test;
    # otherwise its fourth packet correctly exceeds the original 50-mm gate.
    render = packets.__globals__['render_plane']
    packet_function = FunctionType(packets.__code__, packets.__globals__ | {
        'render_plane': lambda transform, normal, offset: render(transform, normal, .32)},
        packets.__name__, packets.__defaults__)
    model = FixedModel()
    controller = candidate.ForecastSourceResidualController(model, ArticulatedCollisionGeometry(URDF),
        forecast_source=mode, public_mission=dict(goal_initial_body_xy_m=[1., 0.],
            return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    previous = None
    for frame in range(4):
        p, d, auxiliary, raw, now = packet_function(frame, previous)
        image = from_captured_rgb(p['image']['rgb'], auxiliary, p,
            measured_ns=now, available_ns=now, now_ns=now)
        raw.update(observer_variant='front_first_dual_camera_anchor_v1', camera_selection_current=True,
            camera_selection=dict(selected_camera='primary' if frame else None,
                initial_paired_reference=frame == 0, auxiliary_attempted=False))
        raw['current_pose'].update(auxiliary_rgb_sha256=image['rgb_sha256'],
            auxiliary_depth_sha256=depth_digest(auxiliary))
        raw['calibration_ids'] = dict(rgb=p['image']['calibration_id'], depth=d['calibration_id'],
            body=p['sensor_state']['sensed']['gyro']['calibration_id'],
            auxiliary_rgb=image['calibration_id'], auxiliary_depth=auxiliary['calibration_id'])
        controller.motion = SimpleNamespace(observe=lambda *a, **k: raw)
        result = controller.observe(p, d, None, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
        assert result['terminal'] is None, result['failure']
        previous = raw
    assert result['new_selection']['forecast_provenance']['forecast_source'] == mode
    assert result['new_selection']['model_prediction_corrected'] == (mode == 'frozen_world_model')
    assert len(controller.memory.route) == 4 and controller.residual.frame == 3
    assert controller.residual.pending['tick'] == 3
    assert len(model.calls) == int(mode == 'frozen_world_model')


@pytest.mark.parametrize('mode', SOURCES)
def test_controller_retains_observation_and_residual_ownership_and_failure_latch(mode):
    from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController as Old
    new = candidate.ForecastSourceResidualController
    assert new.observe is Old.observe and new.advance is Old.advance
    controller = new(object(), object(), forecast_source=mode, public_mission=dict(
        goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    assert controller.selector.residual is controller.residual and controller.memory is controller.mapper.surface
    failed = controller.observe({}, {}, {}, now_ns=1)
    assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and failed['requested_command'] == [0., 0., 0.]
    assert failed['assigned_forecast_source'] == mode and not failed['fully_nonpredictive_controller']
    before = controller.residual.snapshot()
    assert controller.advance({}, {}, now_ns=2)['terminal'] == failed['terminal']
    assert controller.residual.snapshot() == before

"""Actual image inference and common pre-action state for the new comparators."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.forecast_source_residual_controller_development import ForecastSourceResidualController
from lewm.measured_plane_comparator_controllers_development import (
    MeasuredPlaneForecastSourceController, MeasuredPlaneReactiveController)
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from lewm.reactive_floor_transport_controller_development import ReactiveFloorTransportController
from lewm.tests.test_forecast_source_selection_development import FixedModel
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF


MISSION = dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
    require_return_after_goal=True)


class FixedHeadModel(FixedModel):
    def __init__(self, condition):
        super().__init__()
        if condition == 'direct':
            self.corrected_heads = ('direct_outcomes',)
            self.register_buffer('direct_outcomes_xy_bias', self.rollout_outcomes_xy_bias.clone())
            del self.rollout_outcomes_xy_bias

    def forward(self, **kwargs):
        result = super().forward(**kwargs)
        if self.corrected_heads == ('direct_outcomes',):
            result['direct_outcomes'] = result.pop('rollout_outcomes')
        return result


def controllers(condition, variant):
    models = [FixedHeadModel(condition) for _ in range(3)]
    geometry = ArticulatedCollisionGeometry(URDF)
    shared = dict(public_mission=deepcopy(MISSION), navigation_ticks=40)
    learned = shared | dict(condition=condition, variant=variant, persistent=True)
    return models, [
        MeasuredPlaneResidualController(models[0], geometry, **learned),
        MeasuredPlaneForecastSourceController(models[1], geometry,
            forecast_source='frozen_world_model', **learned),
        MeasuredPlaneForecastSourceController(models[2], geometry,
            forecast_source='nominal_requested_twist', **learned),
        MeasuredPlaneReactiveController(geometry, **shared)]


@pytest.mark.parametrize('condition,variant', [('jepa', 'full'), ('direct', 'no_rgb')])
def test_actual_images_share_pose_map_and_mission_through_first_choice(condition, variant):
    models, arms = controllers(condition, variant)
    states = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    assert all(type(c.motion) is MeasuredPlaneVisualMotion for c in arms)
    assert len({id(c.motion) for c in arms}) == len(arms)
    assert len({id(c.mapper) for c in arms}) == len(arms)
    assert not hasattr(arms[-1], 'model') and not hasattr(arms[-1], 'residual')
    for frame, original in enumerate(sequence()):
        rows = []
        for controller in arms:
            p, d, f, options = [move_test_origin(deepcopy(v)) for v in original]
            rows.append(controller.observe(p, d, f, **options))
        for controller, row in zip(arms, rows, strict=True):
            assert row['terminal'] is None, row['failure']
            assert row['measured_plane_constrained_estimator']
            assert len(controller.memory.route) == frame+1
            for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt'):
                assert row[key] == rows[0][key], key
            assert controller.mapper.floor == arms[0].mapper.floor
            assert controller.mapper.occupied == arms[0].mapper.occupied
            assert controller.memory.route == arms[0].memory.route
            np.testing.assert_array_equal(controller.memory.position, arms[0].memory.position)
            np.testing.assert_array_equal(controller.memory.rotation, arms[0].memory.rotation)
            if frame < 3:
                assert row['requested_command'] == [0., 0., 0.]
        # Complete learned-arm decisions stay equal apart from explicit source
        # provenance and root labels. No selective action-only comparison.
        original_row = deepcopy(rows[0]); compared = deepcopy(rows[1])
        compared['controller'] = original_row['controller']
        for key in ('assigned_forecast_source', 'shared_observed_residual_correction_retained',
                    'fully_nonpredictive_controller'):
            compared.pop(key)
        if compared['new_selection'] is not None:
            compared['new_selection'].pop('forecast_provenance', None)
        assert compared == original_row
    assert [len(model.calls) for model in models] == [1, 1, 0]
    for model, state in zip(models, states, strict=True):
        assert state.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
        assert all(parameter.grad is None for parameter in model.parameters())
    for model in models[:2]:
        assert bool(model.calls[0]['rgb'].any()) == (variant == 'full')
    nominal = rows[2]['new_selection']
    assert nominal['forecast_provenance']['nominal_requested_twist_forecasts_used']
    assert not nominal['model_prediction_corrected']
    assert nominal['planned_segment_count'] == 8
    assert not rows[2]['fully_nonpredictive_controller']
    assert arms[2].residual.pending['tick'] == 3
    reactive = rows[3]
    assert reactive['fully_nonpredictive_controller']
    assert reactive['reactive_is_whole_method_comparison']
    assert not reactive['isolated_prediction_ranking_ablation']
    assert not reactive['candidate_future_outcomes_evaluated']
    assert not reactive['predictive_surface_or_path_gates_applied']
    assert not reactive['learned_model_used'] and not reactive['learned_residual_used']
    assert 'prediction' not in reactive['new_selection']
    assert not reactive['new_selection']['command_integrated_pose_used']
    # Duplicate acquisition must stop every arm without extending the map.
    for controller in arms:
        failed = controller.observe(p, d, f, **options)
        assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
        assert failed['requested_command'] == [0., 0., 0.]
        assert len(controller.memory.route) == 4
        again = controller.observe({}, {}, {}, now_ns=1)
        assert again['terminal'] == failed['terminal']
        assert len(controller.memory.route) == 4


@pytest.mark.parametrize('new,old', [
    (MeasuredPlaneForecastSourceController, ForecastSourceResidualController),
    (MeasuredPlaneReactiveController, ReactiveFloorTransportController)])
def test_observation_and_control_methods_are_inherited_unchanged(new, old):
    assert new.observe is old.observe
    assert new.advance is old.advance


@pytest.mark.parametrize('source', [None, 'reactive', 'learned', 1])
def test_forecast_source_is_explicit_and_cannot_claim_reactive_mode(source):
    with pytest.raises(ValueError, match='forecast source required'):
        MeasuredPlaneForecastSourceController(object(), object(), forecast_source=source,
            public_mission=MISSION, navigation_ticks=40, condition='jepa', variant='full', persistent=True)

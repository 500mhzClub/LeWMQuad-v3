"""Common real-image perception and separate predictive/reactive treatments."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from lewm import extended_return_budget_comparator_controllers_development as controls
from lewm.extended_return_budget_controller_development import (
    ExtendedReturnBudgetChainedController, ExtendedReturnBudgetFootprintScope,
    ExtendedReturnBudgetMemory, current_measured_floor_pose)
from lewm.packed_fused_scoped_controller_development import index_owners
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.measured_plane_full_history_timing_development import observer_state_tree
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def arms(condition, variant, *, geometry=True):
    models = [FixedHeadModel(condition) for _ in range(3)]
    geometries = [ArticulatedCollisionGeometry(URDF) if geometry else None for _ in range(4)]
    shared = dict(public_mission=deepcopy(MISSION), navigation_ticks=8000)
    learned = shared | dict(condition=condition, variant=variant, persistent=True)
    return models, [
        ExtendedReturnBudgetChainedController(models[0], geometries[0], **learned),
        controls.ExtendedReturnBudgetForecastSourceController(models[1], geometries[1],
            forecast_source='frozen_world_model', **learned),
        controls.ExtendedReturnBudgetForecastSourceController(models[2], geometries[2],
            forecast_source='nominal_requested_twist', **learned),
        controls.ExtendedReturnBudgetReactiveController(geometries[3], **shared)]


def perception_state(controller):
    return observer_state_tree({name:getattr(controller, name)
        for name in ('motion', 'registration', 'mapper', 'memory', 'mission')})


def test_fresh_composition_shares_exact_perception_types_and_no_mutable_state():
    models, controllers = arms('direct', 'no_rgb', geometry=False)
    reference = fingerprint(perception_state(controllers[0]))
    all_indices = []
    for controller in controllers:
        assert fingerprint(perception_state(controller)) == reference
        assert controller.mission.navigation_ticks == 8000
        assert controller.memory is controller.mapper.surface
        indices = [getattr(owner, name) for owner, name in index_owners(controller.memory)]
        assert all(type(index) is SinglePassMeasuredSampleBoundsIndex for index in indices)
        all_indices.extend(indices)
    assert len({id(index) for index in all_indices}) == 32
    for name in ('motion', 'registration', 'mapper', 'memory', 'mission'):
        assert len({id(getattr(controller, name)) for controller in controllers}) == 4
    reactive = controllers[-1]
    assert not hasattr(reactive, 'model') and not hasattr(reactive, 'residual')
    assert all(not model.calls for model in models)
    assert reactive.advance.__func__.__globals__['current_measured_floor_pose'] is current_measured_floor_pose
    for controller in controllers[1:3]:
        assert controller.selector.residual is controller.residual
        with pytest.raises(AttributeError): controller.selector.forecast_source = 'nominal_requested_twist'


@pytest.mark.parametrize('condition,variant', [('jepa', 'full'), ('direct', 'no_rgb')])
def test_actual_images_preserve_perception_and_frozen_forecast_decisions(monkeypatch, condition, variant):
    models, controllers = arms(condition, variant)
    weights = [{key:value.clone() for key, value in model.state_dict().items()} for model in models]
    scope_calls = []
    original_scope = ExtendedReturnBudgetFootprintScope.__init__
    def scope(self, memory, geometry):
        scope_calls.append(id(memory)); original_scope(self, memory, geometry)
    monkeypatch.setattr(ExtendedReturnBudgetFootprintScope, '__init__', scope)
    for frame, original in enumerate(sequence()):
        rows = []
        for controller in controllers:
            p, d, f, options = [move_test_origin(deepcopy(value)) for value in original]
            rows.append(controller.observe(p, d, f, **options))
        for controller, row in zip(controllers, rows, strict=True):
            assert row['terminal'] is None, row['failure']
            assert row['extended_return_budget_enabled'] and row['chained_anchor_reacquisition_enabled']
            assert row['shared_navigation_budget_ticks'] == 8000
            assert len(controller.memory.route) == frame+1
            for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt'):
                assert row[key] == rows[0][key], key
            assert controller.mapper.floor == controllers[0].mapper.floor
            assert controller.mapper.occupied == controllers[0].mapper.occupied
            assert controller.memory.route == controllers[0].memory.route
            np.testing.assert_array_equal(controller.memory.position, controllers[0].memory.position)
            np.testing.assert_array_equal(controller.memory.rotation, controllers[0].memory.rotation)
        # Keep every ordinary learned decision field; strip only the explicit
        # forecast-source labels and provenance added by the new control.
        normalized = deepcopy(rows[1]); normalized['controller'] = rows[0]['controller']
        for key in ('assigned_forecast_source', 'shared_observed_residual_correction_retained',
                'fully_nonpredictive_controller'): normalized.pop(key)
        if normalized['new_selection'] is not None:
            normalized['new_selection'].pop('forecast_provenance', None)
        assert normalized == rows[0]
    assert [len(model.calls) for model in models] == [1, 1, 0]
    assert set(scope_calls) == {id(controller.memory) for controller in controllers[:3]}
    assert all(type(controller.memory) is ExtendedReturnBudgetMemory for controller in controllers)
    for model, before in zip(models, weights, strict=True):
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())
    for model in models[:2]: assert bool(model.calls[0]['rgb'].any()) == (variant == 'full')
    nominal = rows[2]['new_selection']
    assert nominal['forecast_provenance']['nominal_requested_twist_forecasts_used']
    assert not nominal['model_prediction_corrected'] and nominal['planned_segment_count'] == 8
    assert not rows[2]['fully_nonpredictive_controller'] and controllers[2].residual.pending['tick'] == 3
    reactive = rows[3]
    assert reactive['fully_nonpredictive_controller'] and reactive['reactive_is_whole_method_comparison']
    assert not reactive['isolated_prediction_ranking_ablation']
    assert not reactive['learned_model_used'] and not reactive['learned_residual_used']
    assert not reactive['candidate_future_outcomes_evaluated'] and not reactive['predictive_surface_or_path_gates_applied']
    assert 'prediction' not in reactive['new_selection']
    assert not reactive['new_selection']['command_integrated_pose_used']
    for controller in controllers:
        failed = controller.observe(p, d, f, **options)
        assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and failed['requested_command'] == [0., 0., 0.]
        assert len(controller.memory.route) == 4
        again = controller.observe({}, {}, {}, now_ns=1)
        assert again['terminal'] == failed['terminal'] and len(controller.memory.route) == 4


@pytest.mark.parametrize('source', [None, 'reactive', 'learned', 1])
def test_explicit_forecast_source_required_before_construction(source):
    with pytest.raises(ValueError, match='forecast source required'):
        controls.ExtendedReturnBudgetForecastSourceController(None, None, forecast_source=source,
            public_mission=MISSION, navigation_ticks=8000, condition='direct', variant='no_rgb', persistent=True)


@pytest.mark.parametrize('budget', [-1, 0, 8001, True, 1.5, None])
def test_reactive_budget_cannot_escape_shared_admission(budget):
    with pytest.raises(ValueError, match='bounded extended'):
        controls.ExtendedReturnBudgetReactiveController(None, public_mission=MISSION, navigation_ticks=budget)

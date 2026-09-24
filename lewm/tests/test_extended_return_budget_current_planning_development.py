"""Real paired observations preserve contact memory while routing sees current cells."""
from copy import deepcopy

import pytest
import torch

from lewm import extended_return_budget_current_planning_development as current
from lewm.extended_return_budget_controller_development import (
    ExtendedReturnBudgetChainedController, ExtendedReturnBudgetFloorMap,
    ExtendedReturnBudgetFootprintScope, ExtendedReturnBudgetMemory)
from lewm.extended_return_budget_memory_development import ExtendedReturnBudgetRecordingFloorGeometry
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def test_composition_retains_extended_observation_context_and_exact_view_methods():
    mapper = current.ExtendedReturnBudgetCurrentPlanningMap
    assert mapper.observe is ExtendedReturnBudgetFloorMap.observe
    assert mapper.observe.__globals__['RecordingFloorGeometry'] is ExtendedReturnBudgetRecordingFloorGeometry
    assert mapper._observe_both is current.CurrentObservationPlanningMap._observe_both
    assert mapper.planning_view is current.CurrentObservationPlanningMap.planning_view
    assert current.ExtendedReturnBudgetCurrentPlanningController.observe is ExtendedReturnBudgetChainedController.observe
    assert current.ExtendedReturnBudgetCurrentPlanningController.advance is ExtendedReturnBudgetChainedController.advance


@pytest.mark.parametrize('condition,variant', [('jepa', 'full'), ('direct', 'no_rgb')])
def test_real_images_keep_forecasts_contact_and_tracking_with_current_routing(monkeypatch, condition, variant):
    models = [FixedHeadModel(condition), FixedHeadModel(condition)]
    weights = [{key:value.clone() for key,value in model.state_dict().items()} for model in models]
    options = dict(public_mission=deepcopy(MISSION), navigation_ticks=8000,
        condition=condition, variant=variant, persistent=True)
    old = ExtendedReturnBudgetChainedController(models[0], ArticulatedCollisionGeometry(URDF), **options)
    new = current.ExtendedReturnBudgetCurrentPlanningController(models[1], ArticulatedCollisionGeometry(URDF), **options)
    scopes = []; original_scope = ExtendedReturnBudgetFootprintScope.__init__
    def scope(self, memory, geometry):
        scopes.append(id(memory)); original_scope(self, memory, geometry)
    monkeypatch.setattr(ExtendedReturnBudgetFootprintScope, '__init__', scope)
    assert new.memory is new.mapper.surface and type(new.memory) is ExtendedReturnBudgetMemory
    assert new.selector.residual is new.residual and new.mission.navigation_ticks == 8000
    retained_ids = [id(getattr(new, name)) for name in ('mission', 'motion', 'registration', 'memory', 'residual', 'history')]
    previous_view = None
    for frame, source in enumerate(sequence()):
        rows = []
        for controller in (old, new):
            p, d, f, kwargs = [move_test_origin(deepcopy(value)) for value in source]
            rows.append(controller.observe(p, d, f, **kwargs))
        baseline, actual = rows
        assert baseline['terminal'] is None, baseline['failure']
        assert actual['terminal'] is None, actual['failure']
        for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
                'observed_goal_distance_m', 'auxiliary_floor_partition_receipt'):
            assert baseline[key] == actual[key], key
        assert old.memory.route == new.memory.route
        assert old.mapper.floor == new.mapper.floor and old.mapper.occupied == new.mapper.occupied
        assert retained_ids == [id(getattr(new, name)) for name in ('mission', 'motion', 'registration', 'memory', 'residual', 'history')]
        for xy, yaw in (([0., 0.], 0.), ([.01, 0.], .04)):
            assert old.memory.footprint(old.geometry, xy, yaw, now_ns=kwargs['now_ns'], persistent=True) == new.memory.footprint(
                new.geometry, xy, yaw, now_ns=kwargs['now_ns'], persistent=True)
        if previous_view is not None:
            with pytest.raises(ValueError): previous_view.current(kwargs['now_ns'])
        view = new.mapper.planning_view(now_ns=kwargs['now_ns']); previous_view = view
        assert view.surface is new.memory and all(value == frame for value in view.floor.values())
        assert set(view.floor) <= set(new.mapper.floor) and set(view.occupied) <= set(new.mapper.occupied)
        with pytest.raises(TypeError): view.floor[(0, 0)] = frame
        if frame < 3:
            stripped = {key:value for key,value in actual.items() if key not in current.METADATA}
            stripped['controller'] = baseline['controller']; assert stripped == baseline
        else:
            assert actual['new_selection']['prediction'] == baseline['new_selection']['prediction']
            assert actual['new_selection']['planning_map_receipt'] == view.receipt
            assert actual['new_selection']['proposal']['observed_floor_cells'] == len(view.floor)
            actual['new_selection']['planning_map_receipt']['camera_witnesses'].clear()
            assert len(view.receipt['camera_witnesses']) == 2
    assert set(scopes) == {id(old.memory), id(new.memory)}
    assert [len(model.calls) for model in models] == [1, 1]
    for key, value in current.METADATA.items(): assert actual[key] == value
    assert not actual['memoryless_controller'] and not actual['accumulated_planning_cells_queried']
    for controller in (old, new):
        failed = controller.observe(p, d, f, **kwargs)
        assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and failed['requested_command'] == [0., 0., 0.]
        assert len(controller.memory.route) == 4
    # Duplicate acquisition fails in tracking before the map is called. Keep
    # the last valid evidence; terminal control must never query it for a new
    # observation or invoke the model again.
    assert new.mapper.planning_view(now_ns=kwargs['now_ns']) is previous_view
    with pytest.raises(ValueError): new.mapper.planning_view(now_ns=kwargs['now_ns']+100_000_000)
    again = new.observe({}, {}, {}, now_ns=1)
    assert again['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and again['requested_command'] == [0., 0., 0.]
    assert [len(model.calls) for model in models] == [1, 1]
    for model, before in zip(models, weights, strict=True):
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())

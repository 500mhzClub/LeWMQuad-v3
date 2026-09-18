"""Same measured perception and model forecasts, current planning cells only."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.measured_plane_current_observation_planning_controller_development import MeasuredPlaneCurrentObservationPlanningController
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from lewm.current_observation_planning_map_development import CurrentObservationPlanningMap
from lewm.residual_current_observation_planning_controller_development import ResidualCurrentObservationPlanningController, METADATA
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF


@pytest.mark.parametrize('condition,variant', [('jepa', 'full'), ('direct', 'no_rgb')])
def test_actual_image_to_first_action_preserves_perception_forecasts_and_memory_scope(condition, variant):
    models = [FixedHeadModel(condition), FixedHeadModel(condition)]
    original_states = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    options = dict(public_mission=deepcopy(MISSION), navigation_ticks=40,
        condition=condition, variant=variant, persistent=True)
    baseline = MeasuredPlaneResidualController(models[0], ArticulatedCollisionGeometry(URDF), **options)
    candidate = MeasuredPlaneCurrentObservationPlanningController(models[1], ArticulatedCollisionGeometry(URDF), **options)
    assert type(candidate.motion) is type(baseline.motion) is MeasuredPlaneVisualMotion
    assert type(candidate.mapper) is CurrentObservationPlanningMap
    assert candidate.memory is candidate.mapper.surface
    assert candidate.selector.residual is candidate.residual
    previous_view = None
    for frame, source in enumerate(sequence()):
        decisions = []
        for controller in (baseline, candidate):
            p, d, f, kwargs = [move_test_origin(deepcopy(v)) for v in source]
            decisions.append(controller.observe(p, d, f, **kwargs))
        old, new = decisions
        assert old['terminal'] is None and new['terminal'] is None
        for key in ('evidence', 'original_visual_evidence', 'memory_receipt', 'mission_receipt',
                'observed_goal_distance_m', 'auxiliary_floor_partition_receipt'):
            assert old[key] == new[key], key
        assert baseline.mapper.floor == candidate.mapper.floor
        assert baseline.mapper.occupied == candidate.mapper.occupied
        assert baseline.memory.route == candidate.memory.route
        np.testing.assert_array_equal(baseline.memory.position, candidate.memory.position)
        np.testing.assert_array_equal(baseline.memory.rotation, candidate.memory.rotation)
        if previous_view is not None:
            with pytest.raises(ValueError): previous_view.current(kwargs['now_ns'])
        view = candidate.mapper.planning_view(now_ns=kwargs['now_ns']); previous_view = view
        assert view.surface is candidate.memory
        assert set(view.floor) <= set(candidate.mapper.floor)
        assert set(view.occupied) <= set(candidate.mapper.occupied)
        if frame < 3:
            stripped = {k: v for k, v in new.items() if k not in METADATA}
            stripped['controller'] = old['controller']
            assert stripped == old
        else:
            assert new['new_selection']['prediction'] == old['new_selection']['prediction']
            assert new['new_selection']['planning_map_receipt'] == view.receipt
            assert new['new_selection']['proposal']['observed_floor_cells'] == len(view.floor)
    assert [len(model.calls) for model in models] == [1, 1]
    for key, expected in METADATA.items(): assert new[key] == expected
    assert not new['memoryless_controller'] and not new['accumulated_planning_cells_queried']
    assert new['tracking_and_floor_anchor_history_retained'] and new['learned_temporal_history_and_residual_retained']
    for controller in (baseline, candidate):
        size = len(controller.memory.route)
        failed = controller.observe(p, d, f, **kwargs)
        assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
        assert failed['requested_command'] == [0., 0., 0.]
        assert len(controller.memory.route) == size
    for model, original in zip(models, original_states, strict=True):
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
        assert all(parameter.grad is None for parameter in model.parameters())


def test_existing_current_planning_methods_are_inherited_without_change():
    assert MeasuredPlaneCurrentObservationPlanningController.observe is ResidualCurrentObservationPlanningController.observe
    assert MeasuredPlaneCurrentObservationPlanningController.advance is ResidualCurrentObservationPlanningController.advance

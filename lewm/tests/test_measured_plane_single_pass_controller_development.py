"""Real synthetic image-to-action integration with complete state comparison."""
from copy import deepcopy

import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController
from lewm.single_pass_body_projected_controller_development import SinglePassBodyProjectedController
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.measured_plane_single_pass_comparison_development import normalize, state


@pytest.mark.parametrize('condition,variant', [('jepa', 'full'), ('direct', 'no_rgb')])
def test_complete_decisions_and_retained_state_match_with_actual_image_inference(condition, variant):
    options = dict(public_mission=deepcopy(MISSION), navigation_ticks=40,
        condition=condition, variant=variant, persistent=True)
    models = [FixedHeadModel(condition), FixedHeadModel(condition)]
    arms = [cls(model, ArticulatedCollisionGeometry(URDF), **options)
        for cls, model in zip((MeasuredPlaneResidualController, MeasuredPlaneSinglePassController), models, strict=True)]
    assert fingerprint(state(arms[0])) == fingerprint(state(arms[1]))
    for original in sequence():
        decisions = []
        for controller in arms:
            p, d, f, options = [move_test_origin(deepcopy(v)) for v in original]
            decisions.append(controller.observe(p, d, f, **options))
        assert all(d['terminal'] is None for d in decisions)
        assert normalize(decisions[1]) == decisions[0]
        assert fingerprint(state(arms[0])) == fingerprint(state(arms[1]))
    assert [len(model.calls) for model in models] == [1, 1]
    for controller in arms:
        before = len(controller.memory.route)
        result = controller.observe(p, d, f, **options)
        assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
        assert result['requested_command'] == [0., 0., 0.]
        assert len(controller.memory.route) == before
    # State normalization must still expose a real retained-evidence mutation.
    arms[1].memory.route[-1]['rgb_sha256'] = 'changed'
    assert fingerprint(state(arms[0])) != fingerprint(state(arms[1]))


def test_original_control_methods_and_only_declared_normalization():
    assert MeasuredPlaneSinglePassController.observe is SinglePassBodyProjectedController.observe
    assert MeasuredPlaneSinglePassController.advance is SinglePassBodyProjectedController.advance
    controller = MeasuredPlaneSinglePassController(object(), object(), public_mission=MISSION,
        navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    decision = controller.observe({}, {}, {}, now_ns=1)
    expected = normalize(decision)
    for key, bad in [('controller', 'other'), ('measured_plane_constrained_estimator', False),
                     ('single_pass_measured_bound_queries_enabled', False)]:
        with pytest.raises(ValueError): normalize(decision | {key: bad})
    revised = deepcopy(decision)
    revised['new_selection'] = {'deliberately_changed': True}
    assert normalize(revised) != expected
    assert normalize(revised)['new_selection'] == revised['new_selection']

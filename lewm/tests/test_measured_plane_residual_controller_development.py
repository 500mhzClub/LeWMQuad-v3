"""Complete observation-to-planning path with actual synthetic visual inference."""
from copy import deepcopy
import numpy as np

from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from lewm.tests.test_forecast_source_selection_development import FixedModel
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence


def move_test_origin(value):
    # Test fixtures start at 1.6 s; the controller requires a 1.5 s initial
    # observation. Move actual acquisition timestamps together, not intervals.
    if isinstance(value, dict):
        return {k: (v-100_000_000 if k in ('measured_ns', 'available_ns', 'decision_ns',
            'image_ns', 'sensor_anchor_ns', 'now_ns') else move_test_origin(v)) for k,v in value.items()}
    return value


def test_actual_synthetic_images_reach_memory_residual_and_learned_selection():
    model = FixedModel()
    c = MeasuredPlaneResidualController(model, ArticulatedCollisionGeometry(URDF),
        public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=True), navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    assert type(c.registration) is MeasuredFloorTransportRegistration
    assert c.observe.__func__ is ResidualAnchoredContinuationController.observe
    assert c.advance.__func__ is ResidualAnchoredContinuationController.advance
    for frame, original in enumerate(sequence()):
        p, d, f, options = [move_test_origin(deepcopy(v)) for v in original]
        row = c.observe(p, d, f, **options)
        assert row['terminal'] is None, row['failure']
        assert row['measured_plane_constrained_estimator']
        assert row['original_visual_evidence']['measured_plane_constrained_estimator']
        assert row['evidence']['floor_registration']['reference']['frame'] == 0
        correction = row['evidence']['floor_registration']['correction']
        assert correction['maximum_correction_m'] == .05
        assert correction['maximum_tilt_correction_rad'] == .10
        np.testing.assert_allclose(c.memory.position, 0., atol=1e-10)
        assert c.residual.frame == frame and len(c.memory.route) == frame+1
    assert len(model.calls) == 1
    assert c.residual.pending['tick'] == 3
    before = len(c.memory.route)
    failed = c.observe(p, d, f, **options)
    assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert failed['requested_command'] == [0., 0., 0.]
    assert len(c.memory.route) == before

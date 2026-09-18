"""Complete synthetic sensor-to-action equivalence, including actual chaining."""
from copy import deepcopy

import pytest
import torch

from lewm import joint_temporal_anchor_continuity_development as primary
from lewm import dual_camera_anchor_pose_development as dual
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.measured_plane_chained_anchor_development import (
    MeasuredPlaneChainedAnchorController, MeasuredPlaneChainedAnchorVisualMotion)
from lewm.measured_plane_chained_single_pass_controller_development import MeasuredPlaneChainedSinglePassController
from lewm.tests.test_measured_plane_chained_anchor_development import empty
from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from scripts.measured_plane_chained_single_pass_comparison_development import normalize
from scripts.measured_plane_full_history_timing_development import observed_state


@pytest.mark.parametrize('condition,variant,camera', [
    ('jepa', 'full', None), ('direct', 'no_rgb', None),
    ('direct', 'no_rgb', 'primary'), ('direct', 'no_rgb', 'auxiliary')])
def test_complete_decisions_state_and_model_match_with_actual_image_tracking(monkeypatch, condition, variant, camera):
    models = [FixedHeadModel(condition), FixedHeadModel(condition)]
    initial = [{k: v.clone() for k, v in model.state_dict().items()} for model in models]
    options = dict(public_mission=deepcopy(MISSION), navigation_ticks=40,
        condition=condition, variant=variant, persistent=True)
    arms = [cls(model, ArticulatedCollisionGeometry(URDF), **options)
        for cls, model in zip((MeasuredPlaneChainedAnchorController,
            MeasuredPlaneChainedSinglePassController), models, strict=True)]
    assert all(type(c.motion) is MeasuredPlaneChainedAnchorVisualMotion for c in arms)
    assert arms[0].motion is not arms[1].motion
    assert fingerprint(observed_state(arms[0])) == fingerprint(observed_state(arms[1]))
    items = list(sequence(blank_primary_at=(0, 1, 2, 3) if camera == 'auxiliary' else ()))
    chained = False
    for frame, source in enumerate(items + [items[-1]]):
        if frame == 1 and camera is not None:
            # Remove only descriptor associations; actual flow, image inliers,
            # measured-plane refinement and temporal admission remain in use.
            monkeypatch.setattr(primary, 'matched_points', empty)
            monkeypatch.setattr(dual, 'matched_points', empty)
        rows = []
        for controller in arms:
            p, d, f, kwargs = [move_test_origin(deepcopy(v)) for v in source]
            rows.append(controller.observe(p, d, f, **kwargs))
        expected, actual = rows
        assert normalize(actual) == expected
        assert fingerprint(observed_state(arms[0])) == fingerprint(observed_state(arms[1]))
        if frame < len(items):
            assert actual['terminal'] is None, actual.get('failure')
        else:
            assert actual['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
            assert actual['requested_command'] == [0., 0., 0.]
        if camera is not None and frame == 2:
            visual = actual['original_visual_evidence']
            assert visual['chained_anchor_fallback']['accepted'] is True
            assert visual['camera_selection']['selected_camera'] == camera
            assert visual['measured_plane_selected_pair']['applied'] is True
            assert visual['measured_plane_selected_pair']['original_inliers_preserved'] is True
            chained = True
    assert chained is (camera is not None)
    assert [len(m.calls) for m in models] == [1, 1]
    for model, before in zip(models, initial, strict=True):
        assert all(torch.equal(before[k], value) for k, value in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    # Only existing performance details are normalized. New scientific evidence
    # or changed tracking flags must not disappear from the comparison.
    changed = deepcopy(actual)
    changed['unexpected_tracking_change'] = {'frame': 2}
    assert normalize(changed) != expected
    assert normalize(changed)['unexpected_tracking_change'] == {'frame': 2}
    for key, value in (('controller', 'other'), ('chained_anchor_reacquisition_enabled', False),
                       ('direct_corner_flow_missingness_fallback_enabled', False)):
        with pytest.raises(ValueError):
            normalize(actual | {key: value})

import numpy as np
import pytest
from lewm import joint_measured_floor_plane_development as planes
from lewm.robust_height_floor_candidates_development import select_clouds as original
from lewm.transport_conditioned_partial_floor_development import select_clouds, transported_normal


@pytest.fixture(autouse=True)
def extent(monkeypatch):
    monkeypatch.setattr(planes, 'MINIMUM_SECOND_EXTENT_M', .02)


def test_residual_normal_is_separate_from_pool_gravity_direction():
    x = np.r_[np.linspace(-.005, .005, 400), np.full(5, .035)]
    points = np.column_stack((x, np.sin(np.arange(len(x)))*.4, -.3+.1*x))
    clouds = [np.empty((0,3)), points]; up = [0.,0.,1.]
    masks, receipt = original(clouds, up)
    actual, _ = select_clouds(clouds, up, residual_normal=np.array([-.1,0.,1.])/np.sqrt(1.01))
    assert all(np.array_equal(a,b) for a,b in zip(actual,masks))
    pruned, details = select_clouds(clouds, up, residual_normal=up)
    assert details['selected_count'] == 400 < receipt['selected_count']
    assert details['transport_conditioned_maximum_residual_m'] <= .003
    assert all(not np.any(a & ~b) for a,b in zip(pruned,masks))


def test_full_plane_preserves_original_selection_and_receipt():
    rng = np.random.default_rng(2026091562)
    points = np.column_stack((rng.uniform(-.5,.5,(400,2)), np.full(400,-.3)))
    clouds = [points[:200],points[200:]]
    masks,receipt = original(clouds,[0.,0.,1.])
    actual,details = select_clouds(clouds,[0.,0.,1.],residual_normal=[.1,0.,np.sqrt(.99)])
    assert details == receipt and all(np.array_equal(a,b) for a,b in zip(actual,masks))


def test_reference_normal_is_transported_through_anchor_correction():
    def pose(R): return dict(current_pose=dict(rotation_initial_body_from_current_body=R))
    # The anchor corrects the raw frame by +90 degrees about y.
    A = [[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
    anchor = pose(A) | dict(original_visual_evidence=pose(np.eye(3)),
        floor_registration=dict(reference=dict(joint_plane=dict(normal_body=[0.,0.,1.]))))
    np.testing.assert_allclose(transported_normal(anchor,pose(np.eye(3))),[-1.,0.,0.],atol=1e-14)

"""Gate decomposition preserves rejection and confers no pose permission."""
import numpy as np
from lewm.rigid_consensus_diagnostic_development import diagnose
from lewm.rgbd_correspondence_motion_development import project


def points(scale):
    a = np.array([[2.326, y, z+.043] for y in np.linspace(-scale, scale, 7)
        for z in np.linspace(-scale, scale, 7)])
    uv, valid = project(a)
    assert valid.all()
    return a, uv


def test_compact_noncollinear_exact_matches_fail_only_grid_coverage():
    a, uv = points(.06)
    r = diagnose(a, a.copy(), uv, uv.copy(), gyro_rotation=np.eye(3), frame=5)
    assert not r['original_accepted'] and r['decomposed_consensus_gate']
    assert r['gate_passes'] == dict(inlier_fraction=True, grid_support=False, displacement=True)
    assert r['inlier_fraction'] == 1. and not r['diagnostic_pose_permission']


def test_spread_exact_correspondences_keep_original_acceptance():
    a, uv = points(.8)
    r = diagnose(a, a.copy(), uv, uv.copy(), gyro_rotation=np.eye(3), frame=5)
    assert r['original_accepted'] and not r['diagnostic_pose_permission']

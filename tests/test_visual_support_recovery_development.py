import numpy as np

from lewm.visual_support_recovery_development import LocalSupportedView, support


def test_recovery_requires_recent_local_observed_support_and_measured_completion():
    view = LocalSupportedView()
    p = np.zeros(3)
    R = np.eye(3)
    turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert view.advance([10, 20], p, R, 0, 0) is None
    assert view.advance([100, 100], p, R, 100_000_000, 0) is None
    active = view.advance([10, 20], p, turn, 500_000_000, 0)
    assert active['measured_ns'] == 100_000_000
    # Merely recovering features does not pretend the target heading was reached.
    assert view.advance([100, 100], p, turn, 600_000_000, 0) is not None
    assert view.advance([50, 60], p, R, 700_000_000, 0) is None
    assert view.advance([10, 20], p, R, 800_000_000, 1) is None


def test_old_or_remote_views_do_not_supply_a_recovery_target():
    for position, now in [(np.array([.21, 0, 0]), 1_000_000_000),
                          (np.zeros(3), 11_000_000_000)]:
        view = LocalSupportedView()
        view.advance([100, 100], np.zeros(3), np.eye(3), 0, 0)
        assert view.advance([10, 20], position, np.eye(3), now, 0) is None


def test_missing_pose_or_witness_cannot_supply_support():
    assert support({'current_pose': None}) is None
    assert support({'current_pose': {'frame': 4, 'measured_ns': 400_000_000}}) is None


def test_persistent_local_view_remains_an_objective_without_becoming_current_support():
    view = LocalSupportedView(maximum_view_age_ns=None)
    view.advance([100, 100], np.zeros(3), np.eye(3), 0, 0)
    # An old local view can request recovery, but reaching its heading with
    # weak current images cannot establish completion.
    active = view.advance([0, 20], np.array([.07, 0., 0.]), np.eye(3), 31_000_000_000, 0)
    assert active is not None and active['measured_ns'] == 0
    assert view.advance([0, 20], np.array([.07, 0., 0.]), np.eye(3), 32_000_000_000, 0) is not None
    assert view.advance([50, 60], np.array([.07, 0., 0.]), np.eye(3), 33_000_000_000, 0) is None
    assert view.advance([0, 20], np.array([.21, 0., 0.]), np.eye(3), 34_000_000_000, 0) is None

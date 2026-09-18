import numpy as np

from lewm.earlier_visual_recovery_development import EarlierSupportedView
from lewm.visual_support_recovery_development import LocalSupportedView


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def test_earlier_onset_keeps_actual_counts_and_same_measured_reference():
    old = LocalSupportedView(maximum_view_age_ns=None)
    new = EarlierSupportedView(maximum_view_age_ns=None)
    position = np.zeros(3)
    for state in (old, new):
        assert state.advance([20, 96], position, rotation(0), 1, 0) is None
    assert old.advance([10, 60], position, rotation(.5), 2, 0) is None
    active = new.advance([10, 60], position, rotation(.5), 2, 0)
    assert active['trigger_ns'] == 2 and active['measured_ns'] == 1
    assert active['selected_features'] == [20, 96]
    np.testing.assert_array_equal(active['rotation'], rotation(0))


def test_release_requires_alignment_and_recovered_support():
    state = EarlierSupportedView(maximum_view_age_ns=None)
    p = np.zeros(3)
    state.advance([96, 0], p, rotation(0), 1, 0)
    state.advance([60, 0], p, rotation(.5), 2, 0)
    assert state.advance([60, 0], p, rotation(0), 3, 0) is not None
    assert state.advance([72, 0], p, rotation(.5), 4, 0) is not None
    assert state.advance([72, 0], p, rotation(0), 5, 0) is None


def test_no_recovery_without_strong_local_past_reference():
    state = EarlierSupportedView(maximum_view_age_ns=None)
    p = np.zeros(3)
    assert state.advance([95, 0], p, rotation(0), 1, 0) is None
    assert state.advance([30, 0], p, rotation(.5), 2, 0) is None
    state.advance([96, 0], p, rotation(0), 3, 0)
    assert state.advance([30, 0], np.array([.21, 0., 0.]), rotation(.5), 4, 0) is None
    assert state.advance([30, 0], p, rotation(.5), 2, 0) is None


def test_support_at_threshold_does_not_trigger_and_generation_resets():
    state = EarlierSupportedView(maximum_view_age_ns=None)
    p = np.zeros(3)
    state.advance([0, 96], p, rotation(0), 1, 0)
    assert state.advance([0, 72], p, rotation(.5), 2, 0) is None
    assert state.advance([0, 71], p, rotation(.5), 3, 0) is not None
    assert state.advance([0, 30], p, rotation(.5), 4, 1) is None
    assert state.good is None

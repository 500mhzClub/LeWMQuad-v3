import numpy as np
import pytest

from lewm.axis_aligned_fine_connectivity_development import AxisGraphClearance
from lewm.fine_stored_obstacle_routing_development import fine_distances


def test_axis_distance_matches_general_segment_geometry():
    rng = np.random.default_rng(20260917)
    cells = rng.integers(-100, 100, (400, 2))
    geometry = AxisGraphClearance(frozenset(map(tuple, cells)))
    for axis in (0, 1):
        for _ in range(80):
            a = rng.uniform(-1.5, 1.5, 2)
            b = a.copy(); b[axis] += rng.uniform(-.8, .8)
            expected = float(fine_distances(a, b, cells).min())
            assert geometry.minimum(a, b) == pytest.approx(expected, abs=1e-14)
            assert geometry.minimum(b, a) == pytest.approx(expected, abs=1e-14)
            assert (geometry.minimum(a, b) > .45 + 1e-12) == (expected > .45 + 1e-12)
    # Non-axis-aligned connectors must still use the general calculation.
    a, b = np.array([-.7, .2]), np.array([.8, -.5])
    assert geometry.minimum(a, b) == pytest.approx(float(fine_distances(a, b, cells).min()), abs=1e-14)


def test_closed_boundaries_clearance_threshold_and_empty_geometry():
    cells = np.array([[0, 0]], dtype=int)
    geometry = AxisGraphClearance(frozenset(map(tuple, cells)))
    for x in (0., .01, .46, .46 + 1e-12, .46 + 2e-12, -.45, -.45 - 2e-12):
        for y0, y1 in ((-.1, .1), (0., 0.), (.02, .2)):
            a, b = np.array([x, y0]), np.array([x, y1])
            expected = float(fine_distances(a, b, cells).min())
            assert geometry.minimum(a, b) == pytest.approx(expected, abs=1e-14)
            assert (geometry.minimum(a, b) > .45 + 1e-12) == (expected > .45 + 1e-12)
    assert AxisGraphClearance(frozenset()).minimum((0., 0.), (1., 0.)) is None
    with pytest.raises(ValueError):
        geometry.minimum((float('nan'), 0.), (0., 0.))

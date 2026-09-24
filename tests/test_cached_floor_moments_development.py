from functools import partial
import numpy as np
import pytest
from lewm.cached_floor_moments_development import FloorCloudMoments, fit_pair_with_moments
from lewm.gyro_coherent_floor_constraint_development import fit_pair


def inputs(outliers=False):
    rng = np.random.default_rng(2026091559)
    a = np.column_stack((rng.uniform(-1, 1, (400, 2)), rng.normal(-.3, .0004, 400)))
    angle = .13
    g = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    b = (a - [.04, -.02, .009]) @ g
    if outliers:
        a[:4, 2] += .012
        b[5:8] += .01 * (g.T @ [0., 0., 1.])
    a.flags.writeable = b.flags.writeable = False
    return a, b, g, dict(reference_pool_count=400, current_pool_count=400,
        reference_up=np.array([0., 0., 1.]), current_up=g.T @ [0., 0., 1.],
        minimum_second_eigenvalue_m2=.0004)


@pytest.mark.parametrize('outliers', [False, True])
def test_reused_clouds_preserve_exact_pair_fit_and_pruning(outliers):
    a, b, g, kw = inputs(outliers)
    moments = FloorCloudMoments()
    cached = partial(fit_pair_with_moments, moments=moments)
    expected = fit_pair(a, b, g, **kw)
    assert cached(a, b, g, **kw) == expected
    assert cached(a, b, g, **kw) == expected
    assert moments.misses == 2 and moments.hits >= 2
    if outliers:
        assert expected['refinement_steps'] > 1 and moments.pruned_reductions > 0
    # Reusing an individual cloud cannot reuse a previous pair's gyro or offsets.
    fresh = b.copy(); fresh[:, 2] += .001; fresh.flags.writeable = False
    assert cached(a, fresh, g, **kw) == fit_pair(a, fresh, g, **kw)
    assert moments.misses == 3
    reverse = dict(reference_pool_count=400, current_pool_count=400,
        reference_up=kw['current_up'], current_up=kw['reference_up'],
        minimum_second_eigenvalue_m2=.0004)
    assert cached(b, a, g.T, **reverse) == fit_pair(b, a, g.T, **reverse)


def test_cached_statistics_do_not_bypass_changed_acceptance_inputs():
    a, b, g, kw = inputs()
    cached = partial(fit_pair_with_moments, moments=FloorCloudMoments())
    cached(a, b, g, **kw)
    for changed in [dict(reference_pool_count=400.),
                    dict(minimum_second_eigenvalue_m2=10.),
                    dict(reference_up=[0., 0., -1.])]:
        with pytest.raises(ValueError) as original:
            fit_pair(a, b, g, **(kw | changed))
        with pytest.raises(type(original.value)) as actual:
            cached(a, b, g, **(kw | changed))
        assert str(actual.value) == str(original.value)
    with pytest.raises(ValueError, match='read-only'):
        cached(a.copy(), b, g, **kw)


def test_bounded_eviction_and_subset_reduction():
    a, b, _, _ = inputs()
    c = b.copy(); c[:, 2] += .01; c.flags.writeable = False
    moments = FloorCloudMoments(capacity=2)
    first = moments(a, a[np.ones(len(a), bool)], True)
    moments(b, b.copy(), True)
    assert moments(a, a.copy(), True) is first
    moments(c, c.copy(), True)
    assert len(moments.entries) == 2 and id(b) not in moments.entries
    subset = a[:200].copy()
    result = moments(a, subset, False)
    assert moments.pruned_reductions == 1
    np.testing.assert_array_equal(result[0], subset.mean(0))
    assert not np.array_equal(result[0], first[0])
    moments(b, b.copy(), True)
    assert moments.misses == 4 and len(moments.entries) == 2

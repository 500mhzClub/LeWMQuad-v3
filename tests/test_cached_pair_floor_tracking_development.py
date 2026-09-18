import numpy as np
import pytest
from lewm.cached_pair_floor_tracking_development import ObservationPairFits
from lewm.gyro_coherent_floor_constraint_development import fit_pair


def inputs():
    rng = np.random.default_rng(20260915)
    a = np.column_stack((rng.uniform(-1, 1, (400, 2)), np.full(400, -.3)))
    b = a + np.array([.01, .02, .003])
    a.flags.writeable = b.flags.writeable = False
    return a, b, np.eye(3), dict(reference_pool_count=400, current_pool_count=400,
        reference_up=[0., 0., 1.], current_up=[0., 0., 1.], minimum_second_eigenvalue_m2=.0004)


def test_repeated_fit_has_independent_receipts_and_clear_recomputes():
    a, b, g, kw = inputs(); cache = ObservationPairFits()
    expected = fit_pair(a, b, g, **kw)
    first = cache(a, b, g, **kw)
    assert first == expected
    first['reference_normal_body'][0] = 99
    first['reference_frame'] = 42
    second = cache(a, b, g, **kw)
    assert second == expected and (cache.hits, cache.misses) == (1, 1)
    second['retained_counts'][0] = 0
    assert cache(a, b, g, **kw) == expected
    cache.clear()
    assert not cache.entries
    assert cache(a, b, g, **kw) == expected and (cache.hits, cache.misses) == (0, 1)


@pytest.mark.parametrize('changed', ['reference', 'current', 'gyro', 'reference_up',
    'current_up', 'reference_pool_count', 'current_pool_count', 'minimum_second_eigenvalue_m2'])
def test_every_fit_input_change_recomputes(changed):
    a, b, g, kw = inputs(); cache = ObservationPairFits()
    cache(a, b, g, **kw)
    if changed in ('reference', 'current'):
        fresh = (a if changed == 'reference' else b).copy()
        fresh[:, 2] += .001; fresh.flags.writeable = False
        if changed == 'reference': a = fresh
        else: b = fresh
    elif changed == 'gyro':
        angle = .001
        g = np.array([[np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    elif changed.endswith('_up'):
        kw[changed] = [0., .001, np.sqrt(1-.001**2)]
    elif changed.endswith('pool_count'):
        kw[changed] = 401
    else: kw[changed] = .0005
    assert cache(a, b, g, **kw) == fit_pair(a, b, g, **kw)
    assert (cache.hits, cache.misses) == (0, 2)


def test_writable_clouds_and_changed_invalid_inputs_cannot_hit():
    a, b, g, kw = inputs(); cache = ObservationPairFits()
    cache(a, b, g, **kw)
    with pytest.raises(ValueError, match='read-only'):
        cache(a.copy(), b, g, **kw)
    with pytest.raises(ValueError, match='pool counts'):
        cache(a, b, g, **(kw | dict(reference_pool_count=400.)))
    with pytest.raises(ValueError):
        cache(a, b, g.reshape(-1), **kw)
    assert cache.hits == 0

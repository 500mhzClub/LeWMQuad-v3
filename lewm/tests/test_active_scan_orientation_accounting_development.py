import numpy as np
import pytest

from scripts.diagnose_go2_active_scan_orientation_development_v2 import physical_array_digest


def test_per_array_bindings_produce_one_hashable_order_independent_group():
    a = np.array([1., 2.])
    b = np.array([[3., 4.]])
    key = physical_array_digest({'a': a, 'b': b})
    groups = {key: ['first']}
    groups.setdefault(physical_array_digest({'b': b.copy(), 'a': a.copy()}), []).append('second')
    assert groups == {key: ['first', 'second']} and len(key) == 64


@pytest.mark.parametrize('change', ['value', 'shape', 'dtype', 'name'])
def test_different_physics_arrays_do_not_collapse(change):
    baseline = {'a': np.array([1., 2.])}
    altered = {'a': baseline['a'].copy()}
    if change == 'value':
        altered['a'][0] += .01
    elif change == 'shape':
        altered['a'] = altered['a'].reshape(1, 2)
    elif change == 'dtype':
        altered['a'] = altered['a'].astype(np.float32)
    else:
        altered = {'b': altered['a']}
    assert physical_array_digest(altered) != physical_array_digest(baseline)

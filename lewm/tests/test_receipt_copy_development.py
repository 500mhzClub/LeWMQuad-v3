from copy import deepcopy
from collections import defaultdict, namedtuple
import numpy as np
import pytest
from lewm.receipt_copy_development import copy_receipt


def test_shared_mutable_substructure_remains_shared_but_independent_of_source():
    row = {'points': [1., 2., 3.], 'flag': True}
    source = {'before': row, 'after': row, 'pair': (row, None)}
    result = copy_receipt(source)
    assert result == deepcopy(source)
    assert result['before'] is result['after'] is result['pair'][0]
    assert result['before'] is not row
    result['before']['points'][0] = 9.
    assert source['before']['points'][0] == 1.


@pytest.mark.parametrize('container', ['list', 'dict', 'tuple_list'])
def test_cycles_preserve_identity_without_referencing_source(container):
    if container == 'list':
        source = []; source.append(source)
        result = copy_receipt(source)
        assert result is result[0] and result is not source
    elif container == 'dict':
        source = {}; source['self'] = source
        result = copy_receipt(source)
        assert result is result['self'] and result is not source
    else:
        inner = []; source = (inner,); inner.append(source)
        result = copy_receipt(source)
        assert result is result[0][0] and result[0] is not inner


def test_numpy_fallback_preserves_aliasing_and_copies_array_data():
    a = np.arange(12.).reshape(3, 4)
    source = [a, {'same': a}, a[:, ::2], np.float64(1.25)]
    old, new = deepcopy(source), copy_receipt(source)
    assert new[0] is new[1]['same'] and new[0] is not a
    for i in (0, 2, 3): np.testing.assert_array_equal(old[i], new[i])
    new[0][0, 0] = 10.; assert a[0, 0] == 0.


def test_container_subclasses_and_namedtuples_keep_standard_type_semantics():
    Pair = namedtuple('Pair', ['left', 'right'])
    a = []
    source = defaultdict(list, item=Pair(a, a))
    result = copy_receipt(source)
    assert type(result) is defaultdict and result.default_factory is list
    assert type(result['item']) is Pair
    assert result['item'].left is result['item'].right and result['item'].left is not a


def test_custom_copy_can_reference_outer_container_through_shared_memo():
    class Witness:
        def __deepcopy__(self, memo):
            return {'outer': memo[id(source)]}
    source = [Witness()]
    result = copy_receipt(source)
    assert result[0]['outer'] is result and result is not source


def test_custom_memo_atomic_override_and_value_before_key_evaluation_match():
    marker = 'noninterned marker '+str(object())
    class Witness:
        def __deepcopy__(self, memo):
            memo[id(marker)] = 'replacement'
            return 5
    source = {marker: Witness(), 'later': marker}
    assert copy_receipt(source) == deepcopy(source) == {'replacement': 5, 'later': 'replacement'}


def test_fallback_exceptions_and_immutable_tuple_identity_match():
    class Reject:
        def __deepcopy__(self, memo): raise ValueError('cannot copy witness')
    for fn in (deepcopy, copy_receipt):
        with pytest.raises(ValueError, match='cannot copy witness'): fn([Reject()])
    value = ('identity', 2, None, (False, 1.2))
    assert copy_receipt(value) is value and deepcopy(value) is value

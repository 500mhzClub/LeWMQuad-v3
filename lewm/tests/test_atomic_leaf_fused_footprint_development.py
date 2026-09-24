"""Graph ownership, unsupported fallbacks and original cache/recovery semantics."""
from copy import deepcopy
import json
from types import FunctionType

import numpy as np
import pytest

from lewm import atomic_leaf_fused_footprint_development as candidate
from lewm import fused_scoped_footprint_development as baseline
from lewm.frozen_footprint_receipts_development import _ReceiptDict, _ReceiptList, detach_receipts
from lewm.tests import test_scoped_footprint_reuse_development as original_cache_tests


# Reuse cache-only tests, including actual recovery functions. Controller and
# selector integration are separate and are not claimed by this component suite.
for _name, _function in vars(original_cache_tests).items():
    if (not _name.startswith('test_') or not isinstance(_function, FunctionType)
            or 'ScopedFootprintReuse' not in _function.__code__.co_names
            or 'ScopedFootprintAnchoredSelector' in _function.__code__.co_names):
        continue
    _clone = FunctionType(_function.__code__, _function.__globals__ | dict(
        ScopedFootprintReuse=candidate.AtomicLeafFusedScopedFootprintReuse), _name, _function.__defaults__)
    _clone.__kwdefaults__ = _function.__kwdefaults__
    _clone.__dict__.update(_function.__dict__)
    globals()[_name] = _clone


def graph_signature(value):
    seen = {}
    def visit(node):
        if type(node) not in (dict, list, _ReceiptDict, _ReceiptList):
            if type(node) is float: return ('float', node.hex())
            return (type(node).__name__, node)
        if id(node) in seen: return ('alias', seen[id(node)])
        index = len(seen); seen[id(node)] = index
        children = ([(key, visit(child)) for key, child in node.items()]
            if isinstance(node, dict) else [visit(child) for child in node])
        return (type(node).__name__, index, children)
    return visit(value)


@pytest.mark.parametrize('size', [0, 1, 7, 1000])
def test_primitive_leaves_aliases_freeze_clone_and_detach_exact(size):
    leaves = [None, False, True, 0, -123, 2**80, 0., -0., float('inf'), float('nan'), 'text', '\u00e9']*size
    shared = {'leaves':leaves}; source = {'left':shared, 'right':[shared, shared]}
    expected = baseline.freeze_ordinary_footprint(source)
    actual = candidate.freeze_ordinary_footprint(source)
    assert graph_signature(actual) == graph_signature(expected)
    assert actual is not source and actual['left'] is actual['right'][0] is actual['right'][1]
    assert deepcopy(actual) is actual
    for value in (actual, candidate._clone_cached_receipt(actual)):
        assert graph_signature(value) == graph_signature(expected)
        assert graph_signature(detach_receipts(value)) == graph_signature(detach_receipts(expected))
        with pytest.raises(TypeError): value['left']['leaves'].append('bad')
        with pytest.raises(TypeError): value['new'] = 1
    copied = candidate._clone_cached_receipt(actual)
    assert copied is not actual and copied['left'] is not actual['left']
    for a,b in zip(actual['left']['leaves'], copied['left']['leaves']): assert a is b
    source['left']['leaves'].append('source mutation')
    assert graph_signature(actual) == graph_signature(expected)


@pytest.mark.parametrize('fault', ['tuple', 'bytes', 'complex', 'array', 'numpy_scalar', 'key',
    'dict_subclass', 'list_subclass', 'float_subclass', 'str_subclass', 'root_cycle', 'list_cycle', 'custom'])
def test_partial_invalid_graph_returns_entire_original(fault):
    source = {'good': {'nested':[1, 2]}, 'bad':None}
    values = dict(tuple=(1, 2), bytes=b'x', complex=1j, array=np.zeros(2), numpy_scalar=np.float64(1),
        key={1:'bad'}, dict_subclass=type('D',(dict,),{})(), list_subclass=type('L',(list,),{})(),
        float_subclass=type('F',(float,),{})(1), str_subclass=type('S',(str,),{})('x'), custom=object())
    if fault == 'root_cycle': source['bad'] = source
    elif fault == 'list_cycle':
        cycle = []; cycle.append(cycle); source['bad'] = cycle
    else: source['bad'] = values[fault]
    before = source['good']; child = before['nested']
    assert baseline.freeze_ordinary_footprint(source) is source
    assert candidate.freeze_ordinary_footprint(source) is source
    assert source['good'] is before and before['nested'] is child and child == [1, 2]


@pytest.mark.parametrize('source', [None, True, 3, 1., 'x', [], (), object()])
def test_non_dict_root_and_uncached_leaf_clone_retain_identity(source):
    assert candidate.freeze_ordinary_footprint(source) is source
    assert candidate._clone_cached_receipt(source) is source


def test_shared_dag_not_expanded_and_clone_containers_are_separately_owned():
    source = {'leaf':[1, 2]}
    for _ in range(30): source = {'left':source, 'right':source}
    actual = candidate.freeze_ordinary_footprint(source)
    expected = baseline.freeze_ordinary_footprint(source)
    assert graph_signature(actual) == graph_signature(expected)
    cloned = candidate._clone_cached_receipt(actual)
    for _ in range(30):
        assert actual['left'] is actual['right'] and cloned['left'] is cloned['right']
        assert actual is not cloned
        actual, cloned = actual['left'], cloned['left']


def test_cache_body_lifetime_keys_validation_and_counters_are_unchanged():
    old = baseline.FusedScopedFootprintReuse.footprint
    new = candidate.AtomicLeafFusedScopedFootprintReuse.footprint
    assert new.__code__ is old.__code__ and new.__defaults__ is old.__defaults__
    assert new.__kwdefaults__ is old.__kwdefaults__ and new.__globals__ is not old.__globals__
    replacements = dict(freeze_ordinary_footprint=candidate.freeze_ordinary_footprint,
        _clone_cached_receipt=candidate._clone_cached_receipt)
    for key, value in old.__globals__.items(): assert new.__globals__[key] is replacements.get(key, value)
    assert old.__globals__['freeze_ordinary_footprint'] is baseline.freeze_ordinary_footprint


def test_nested_mixed_graphs_match_reference_for_fixed_generated_population():
    rng = np.random.default_rng(904)
    for _ in range(60):
        shared = {'numbers':[float(v) for v in rng.normal(size=20)], 'flags':[None, True, False]}
        source = {'shared':shared, 'rows':[{'index':i, 'shared':shared, 'leaf':str(i)} for i in range(12)]}
        expected = baseline.freeze_ordinary_footprint(source)
        actual = candidate.freeze_ordinary_footprint(source)
        assert graph_signature(actual) == graph_signature(expected)
        assert json.dumps(actual, allow_nan=False) == json.dumps(expected, allow_nan=False)
        assert graph_signature(candidate._clone_cached_receipt(actual)) == graph_signature(baseline._clone_cached_receipt(expected))

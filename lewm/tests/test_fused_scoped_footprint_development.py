"""Reapply the original cache contracts and exercise fused graph ownership."""
from copy import deepcopy
from types import FunctionType

import numpy as np
import pytest

from lewm.frozen_footprint_receipts_development import freeze_footprint, detach_receipts
from lewm.fused_scoped_footprint_development import freeze_ordinary_footprint, FusedScopedFootprintReuse
from lewm.fused_scoped_batched_controller_development import (
    FusedScopedBatchedController, FusedScopedBatchedSelector, normalize_to_combined)
from lewm.scoped_batched_footprint_controller_development import ScopedBatchedFootprintController
from lewm.tests import test_scoped_footprint_reuse_development as original


# Run the complete existing suite against the new classes without modifying
# its module globals or its frozen source. Preserve parametrization marks.
for _name, _function in vars(original).items():
    if not _name.startswith('test_') or not isinstance(_function, FunctionType):
        continue
    _globals = dict(_function.__globals__, ScopedFootprintReuse=FusedScopedFootprintReuse,
                    ScopedFootprintAnchoredSelector=FusedScopedBatchedSelector,
                    ScopedFootprintAnchoredController=FusedScopedBatchedController)
    _clone = FunctionType(_function.__code__, _globals, _name, _function.__defaults__)
    _clone.__kwdefaults__ = _function.__kwdefaults__
    _clone.__dict__.update(_function.__dict__)
    globals()[_name] = _clone


def test_fused_freeze_preserves_dag_aliases_and_detaches_source():
    shared = {'leaf': [None, True, 8, -0., 'value']}
    source = {'a': [shared, shared], 'b': shared}
    reference, result = freeze_footprint(source), freeze_ordinary_footprint(source)
    assert result == reference
    assert result['a'][0] is result['a'][1] is result['b']
    assert deepcopy(result) is result
    source['a'][0]['leaf'].append('changed')
    assert result == reference
    with pytest.raises(TypeError): result['b']['leaf'].append(1)
    with pytest.raises(TypeError): result['b']['new'] = 1
    public = detach_receipts(result)
    assert type(public) is dict and type(public['a']) is list
    public['b']['leaf'].append('public')
    assert public['a'][0]['leaf'][-1] == 'public'
    assert result == reference


@pytest.mark.parametrize('kind', ['tuple', 'key', 'array', 'dict_subclass', 'list_subclass', 'cycle', 'nested_cycle'])
def test_partial_freeze_rejection_returns_entire_original_unchanged(kind):
    source = {'already_visited': {'x': [1, 2]}, 'last': None}
    if kind == 'tuple': source['last'] = (1, 2)
    elif kind == 'key': source['last'] = {1: 'non-string key'}
    elif kind == 'array': source['last'] = np.zeros(2)
    elif kind == 'dict_subclass': source['last'] = type('D', (dict,), {})()
    elif kind == 'list_subclass': source['last'] = type('L', (list,), {})()
    elif kind == 'cycle': source['last'] = source
    else:
        child = []
        child.append(child)
        source['last'] = child
    before = source['already_visited']
    assert freeze_footprint(source) is source
    assert freeze_ordinary_footprint(source) is source
    assert source['already_visited'] is before and before == {'x': [1, 2]}


def test_repeated_shared_subgraphs_keep_single_owned_container():
    source = {'leaf': [0, 1]}
    for _ in range(18): source = {'left': source, 'right': source}
    result = freeze_ordinary_footprint(source)
    for _ in range(18):
        assert result['left'] is result['right']
        result = result['left']
    assert result == {'leaf': [0, 1]}


def test_controller_metadata_is_only_normalized_difference_on_failure():
    kwargs = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.],
                  return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
                  navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    baseline = ScopedBatchedFootprintController(None, None, **kwargs)
    candidate = FusedScopedBatchedController(None, None, **kwargs)
    assert isinstance(candidate.selector, FusedScopedBatchedSelector)
    assert candidate.selector.residual is candidate.residual
    expected = baseline.observe({}, {}, {}, now_ns=1)
    actual = candidate.observe({}, {}, {}, now_ns=1)
    assert normalize_to_combined(actual) == expected
    with pytest.raises(ValueError): normalize_to_combined(expected)

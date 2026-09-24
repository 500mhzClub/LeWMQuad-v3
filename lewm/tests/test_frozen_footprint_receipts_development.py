"""Receipt ownership, live query forwarding and unchanged recovery decisions."""
from copy import copy, deepcopy
import json
import operator
from types import SimpleNamespace
import pytest
from lewm import frozen_footprint_receipts_development as receipts
from lewm.frozen_footprint_anchored_controller_development import (
    FrozenFootprintAnchoredController, FrozenFootprintAnchoredSelector, FLAG)
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)
from lewm.residual_anchored_continuation_development import reconsider_anchored_continuation
from lewm.tests.test_residual_anchored_continuation_development import fixture
from lewm.tests.test_residual_hold_feasibility_development import fixture as hold_fixture
from lewm.tests.test_residual_first_interval_feasibility_development import NOW


def test_deepcopy_shares_only_owned_read_only_graphs_and_detaches_public_result():
    witness = {'values': [1., 2., None, True]}
    original = {'possible_intersection': False, 'shapes': [witness, witness]}
    frozen = receipts.freeze_footprint(original)
    assert frozen == original and frozen is not original
    assert frozen['shapes'][0] is frozen['shapes'][1]
    assert copy(frozen) is frozen and deepcopy(frozen) is frozen
    original['shapes'][0]['values'][0] = 7.
    assert frozen['shapes'][0]['values'][0] == 1.
    selection = {'surface_checks': [frozen], 'other': frozen, 'candidates': [{'utility_m': 1.}]}
    intermediate = deepcopy(selection)
    assert intermediate['surface_checks'][0] is frozen
    assert intermediate['candidates'] is not selection['candidates']
    public = receipts.detach_receipts(intermediate)
    assert type(public) is dict and type(public['surface_checks']) is list
    assert type(public['surface_checks'][0]) is dict
    assert public['surface_checks'][0] is public['other']
    assert json.dumps(public, sort_keys=True) == json.dumps(intermediate, sort_keys=True)
    public['other']['shapes'][0]['values'][0] = 9.
    assert frozen['shapes'][0]['values'][0] == 1.
    assert public['other']['shapes'][1]['values'][0] == 9.


@pytest.mark.parametrize('mutate', [
    lambda x: operator.setitem(x, 'a', 2), lambda x: operator.delitem(x, 'a'),
    lambda x: x.clear(), lambda x: x.pop('a'), lambda x: x.popitem(),
    lambda x: x.setdefault('b', 2), lambda x: x.update(a=2),
    lambda x: operator.ior(x, {'b': 2}), lambda x: x.__init__({'b': 2}),
])
def test_dictionary_mutations_rejected(mutate):
    value = receipts.freeze_footprint({'a': 1})
    with pytest.raises(TypeError, match='read-only'):
        mutate(value)
    assert value == {'a': 1}


@pytest.mark.parametrize('mutate', [
    lambda x: operator.setitem(x, 0, 2), lambda x: operator.delitem(x, 0),
    lambda x: x.append(2), lambda x: x.clear(), lambda x: x.extend([2]),
    lambda x: x.insert(0, 2), lambda x: x.pop(), lambda x: x.remove(1),
    lambda x: x.reverse(), lambda x: x.sort(), lambda x: operator.iadd(x, [2]),
    lambda x: operator.imul(x, 2), lambda x: x.__init__([2]),
])
def test_nested_list_mutations_rejected(mutate):
    value = receipts.freeze_footprint({'a': [1]})['a']
    with pytest.raises(TypeError, match='read-only'):
        mutate(value)
    assert value == [1]


@pytest.mark.parametrize('kind', ['cycle', 'custom', 'tuple', 'non_string_key'])
def test_unsupported_receipts_retain_original_copy_behavior(kind):
    class Custom:
        calls = 0
        def __deepcopy__(self, memo):
            type(self).calls += 1
            return Custom()
    value = {'a': []}
    if kind == 'cycle': value['a'].append(value)
    elif kind == 'custom': value['a'].append(Custom())
    elif kind == 'tuple': value['a'].append((1, 2))
    else: value[1] = 'value'
    assert receipts.freeze_footprint(value) is value
    copied = deepcopy(value)
    assert copied is not value
    if kind == 'cycle': assert copied['a'][0] is copied
    if kind == 'custom': assert Custom.calls == 1


def test_proxy_forwards_each_actual_query_and_bound_map_method():
    class Memory:
        def __init__(self): self.calls = []
        def footprint(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {'possible_intersection': len(self.calls) == 2}
    class Map:
        def __init__(self): self.surface = Memory(); self.floor = {}; self.calls = []
        def waypoint(self, *args, **kwargs):
            self.calls.append((args, kwargs)); return self.floor
    original = Map(); view = receipts.FootprintReceiptMap(original)
    assert view.floor is original.floor
    assert view.waypoint([1, 2], now_ns=NOW) is original.floor
    assert original.calls == [(([1, 2],), {'now_ns': NOW})]
    first = view.surface.footprint('geometry', [0., 0.], 0., now_ns=NOW)
    second = view.surface.footprint('geometry', [0., 0.], 0., now_ns=NOW)
    assert first['possible_intersection'] is False and second['possible_intersection'] is True
    assert len(original.surface.calls) == 2
    assert original.surface.calls[0] == original.surface.calls[1]


def test_detachment_covers_tuple_wrappers_and_container_cycles():
    frozen = receipts.freeze_footprint({'a': [1]})
    loop = []
    wrapper = (frozen, loop)
    loop.append(wrapper)
    public = receipts.detach_receipts({'tuple': wrapper, 'alias': frozen})
    assert type(public['tuple']) is tuple
    assert type(public['tuple'][0]) is dict
    assert public['tuple'][0] is public['alias']
    assert public['tuple'][1][0] is public['tuple']
    public['alias']['a'].append(2)
    assert frozen['a'] == [1]


@pytest.mark.parametrize('make_fixture', [fixture, hold_fixture])
@pytest.mark.parametrize('blocked', [False, True])
def test_complete_original_recovery_results_with_frozen_footprints(make_fixture, blocked):
    selection, residual, mapper = make_fixture()
    mapper.surface.block = blocked
    expected = reconsider_anchored_continuation(selection, residual, mapper, object(), now_ns=NOW)
    internal = deepcopy(selection)
    internal['surface_checks'] = [receipts.freeze_footprint(v) for v in internal['surface_checks']]
    result = reconsider_anchored_continuation(internal, residual,
        receipts.FootprintReceiptMap(mapper), object(), now_ns=NOW)
    public = receipts.detach_receipts(result)
    assert public == expected
    public['surface_checks'][0]['possible_intersection'] = True
    assert selection['surface_checks'][0]['possible_intersection'] is False


def test_selector_calls_original_whole_chain_then_returns_plain_containers(monkeypatch):
    original_mapper = SimpleNamespace(surface=SimpleNamespace(
        footprint=lambda: {'possible_intersection': False, 'nested': [1]}))
    calls = []
    def choose(self, model, history, mapper, geometry, *, now_ns):
        calls.append((self, model, history, mapper, geometry, now_ns))
        result = mapper.surface.footprint()
        assert deepcopy(result) is result
        return {'surface_checks': [result]}
    monkeypatch.setattr(ResidualAnchoredContinuationSelector, 'choose', choose)
    selector = FrozenFootprintAnchoredSelector(residual=object(), condition='jepa',
        variant='full', goal_initial_body_xy_m=[1., 0.])
    model, history, geometry = object(), object(), object()
    result = selector.choose(model, history, original_mapper, geometry, now_ns=NOW)
    assert len(calls) == 1 and calls[0][:3] == (selector, model, history)
    assert calls[0][3]._mapper is original_mapper and calls[0][4:] == (geometry, NOW)
    assert type(result['surface_checks'][0]) is dict
    assert type(result['surface_checks'][0]['nested']) is list
    assert not any(isinstance(v, receipts.FootprintReceiptMap) for v in vars(selector).values())


def test_observation_state_and_failure_behavior_are_inherited():
    for name in ('observe', 'advance'):
        assert getattr(FrozenFootprintAnchoredController, name) is getattr(ResidualAnchoredContinuationController, name)
    controller = FrozenFootprintAnchoredController(object(), object(),
        public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=True), navigation_ticks=40, condition='jepa', variant='full', persistent=True)
    assert controller.selector.residual is controller.residual
    assert controller.memory is controller.mapper.surface
    failed = controller.observe({}, {}, {}, now_ns=1)
    assert failed['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
    assert failed['requested_command'] == [0., 0., 0.] and failed[FLAG] is True

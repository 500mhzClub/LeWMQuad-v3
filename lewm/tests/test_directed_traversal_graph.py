import pytest

from lewm.memory.directed_traversal_graph import DirectedTraversalGraph, Traversal


def graph():
    result = DirectedTraversalGraph()
    for node in ('A', 'B', 'C', 'D'):
        result.observe_place(node)
    return result


def event(identity='1', source='A', target='B', duration_s=1, **changes):
    fields = dict(reached=True, viable_arrival=True, association_confirmed=True)
    fields.update(changes)
    return Traversal(identity, source, target, duration_s, **fields)


def test_observation_alone_and_forward_edge_do_not_invent_return_route():
    g = graph()
    assert g.route('A', 'B') is None
    g.record(event())
    assert g.route('A', 'B') == ['A', 'B']
    assert g.route('B', 'A') is None
    g.record(event('2', 'B', 'A', 3))
    assert g.route('B', 'A') == ['B', 'A']
    assert g.edge_summary('A', 'B')['mean_success_duration_s'] == 1
    assert g.edge_summary('B', 'A')['mean_success_duration_s'] == 3


@pytest.mark.parametrize('failed', ['reached', 'viable_arrival', 'association_confirmed'])
def test_failed_or_unconfirmed_traversal_cannot_create_route(failed):
    g = graph()
    g.record(event(**{failed: False}))
    assert g.route('A', 'B') is None
    assert g.edge_summary('A', 'B')['attempts'] == 1


def test_later_failure_disables_direction_until_reverified_without_erasing_history():
    g = graph()
    g.record(event('1'))
    g.record(event('2', reached=False))
    assert g.route('A', 'B') is None
    g.record(event('3', duration_s=3))
    assert g.route('A', 'B') == ['A', 'B']
    assert g.edge_summary('A', 'B') == {
        'attempts': 3, 'qualified_traversals': 2,
        'usable_now': True, 'mean_success_duration_s': 2.0,
    }


def test_composed_route_respects_measured_direction_cost_and_avoidance():
    g = graph()
    for index, (a, b, duration) in enumerate((('A', 'B', 5), ('B', 'D', 5),
                                             ('A', 'C', 1), ('C', 'D', 1))):
        g.record(event(str(index), a, b, duration))
    assert g.route('A', 'D') == ['A', 'C', 'D']
    assert g.route('A', 'D', avoid_edges={('C', 'D')}) == ['A', 'B', 'D']
    assert g.route('D', 'A') is None
    assert g.route('A', 'A') == ['A']


def test_duplicate_receipt_is_idempotent_but_conflicting_evidence_is_rejected():
    g = graph()
    assert g.record(event())
    assert not g.record(event())
    assert g.edge_summary('A', 'B')['attempts'] == 1
    with pytest.raises(ValueError):
        g.record(event(reached=False))


def test_unobserved_places_cannot_be_smuggled_into_graph():
    g = graph()
    with pytest.raises(ValueError):
        g.record(event(target='oracle-unseen'))
    with pytest.raises(ValueError):
        g.route('A', 'oracle-unseen')


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf'), True])
def test_nonsensical_traversal_cost_is_rejected(bad):
    with pytest.raises(ValueError):
        event(duration_s=bad)


def test_identity_and_explicit_boolean_requirements():
    with pytest.raises(ValueError):
        event(identity='')
    with pytest.raises(ValueError):
        event(source='A', target='A')
    with pytest.raises(ValueError):
        event(reached=1)


def test_existing_navigator_reverse_and_blocked_routes_are_explicit_policy_assumptions():
    from types import SimpleNamespace
    from lewm.memory.topological_navigator import TopologicalNavigator

    legacy = TopologicalNavigator.__new__(TopologicalNavigator)
    legacy.memory = SimpleNamespace(edges={(0, 1): 1})
    assert legacy._weighted_path(1, 0) == [1, 0]
    assert legacy._weighted_path(0, 1, avoid_edges={(0, 1)}) == [0, 1]
    # The prospective committed-route layer makes neither assumption.
    g = graph()
    g.record(event())
    assert g.route('B', 'A') is None
    assert g.route('A', 'B', avoid_edges={('A', 'B')}) is None

import pickle

import pytest

from lewm.current_pair_routing_memory_development import (
    CapturedUpdateSet, CurrentPairRoutingSnapshot, RoutingMemoryScopeMixin)


def snapshot():
    return CurrentPairRoutingSnapshot(frame=4, measured_ns=400_000_000,
        floor=frozenset({(0,0),(1,0)}), occupied=frozenset({(2,0),(3,0)}),
        fine_occupied=frozenset({(10,0),(15,0)}),
        position_map=(0.,0.,0.), map_from_initial=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
        floor_height=-.3, primary_current_floor_cells=1, auxiliary_current_floor_cells=1,
        current_floor=frozenset({(1,0)}), current_occupied=frozenset({(3,0)}),
        current_fine_occupied=frozenset({(15,0)}))


def test_reobserved_cells_are_current_and_both_camera_updates_are_unioned():
    cells = CapturedUpdateSet({(1,0)})
    cells.update(c for c in [(1,0),(2,0)])
    cells.update({(3,0)})
    assert cells.observed == {(1,0),(2,0),(3,0)}
    cells.begin_observation()
    cells.update({(2,0)})
    assert cells.observed == {(2,0)}
    assert cells == {(1,0),(2,0),(3,0)}
    cells.begin_observation()
    cells.update([])
    assert not cells.observed and len(cells) == 3


def test_current_view_survives_process_transport_without_mutating_full_evidence():
    full = pickle.loads(pickle.dumps(snapshot()))
    view = full.current_pair_view()
    assert view.floor == {(1,0)} and view.occupied == {(3,0)}
    assert view.fine_occupied == {(15,0)}
    assert full.floor == {(0,0),(1,0)} and full.fine_occupied == {(10,0),(15,0)}
    assert view.frame == full.frame and view.measured_ns == full.measured_ns


class Consumer:
    def _route(self, snapshot, *args, **kwargs):
        self.route_snapshot = snapshot
        return {'route_cells': []}

    def _route_target(self, route, snapshot, position):
        self.target_snapshot = snapshot
        return position

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        self.clearance_snapshot = snapshot
        return {'action': 'hold'}, evidence


class ScopedConsumer(RoutingMemoryScopeMixin, Consumer):
    pass


@pytest.mark.parametrize('scope', ['persistent','latest_mapped_pair'])
def test_routing_scope_changes_graph_and_lookahead_but_preserves_action_clearance(scope):
    controller = ScopedConsumer(); controller.routing_memory_scope = scope
    full = snapshot(); correction = object()
    controller._route(full)
    controller._route_target({}, full, (0,0))
    selected, returned = controller._select_action(None, correction, None, None, None, full, None, None)
    expected = full.floor if scope == 'persistent' else full.current_floor
    assert controller.route_snapshot.floor == expected
    assert controller.target_snapshot.floor == expected
    assert controller.clearance_snapshot is full and returned is correction
    assert selected['routing_memory_scope']['accumulated_action_clearance_preserved']
    assert selected['routing_memory_scope']['retained_floor_cells'] == 2


def test_no_implicit_fallback_to_persistent_cells_for_missing_current_evidence():
    controller = ScopedConsumer(); controller.routing_memory_scope = 'latest_mapped_pair'
    with pytest.raises(ValueError, match='captured'):
        controller._routing_view(object())

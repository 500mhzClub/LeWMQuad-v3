from dataclasses import replace

import pytest

from lewm.current_pair_routing_memory_development import CurrentPairRoutingSnapshot
from lewm.return_routing_memory_development import RUNTIMES


@pytest.mark.parametrize('arm', RUNTIMES)
def test_only_return_routing_changes_and_original_snapshot_remains(arm):
    runtime = RUNTIMES[arm].__new__(RUNTIMES[arm])
    snapshot = CurrentPairRoutingSnapshot(frame=40, measured_ns=5_500_000_000,
        floor=frozenset({(0, 0), (1, 0)}), occupied=frozenset({(20, 20), (21, 20)}),
        fine_occupied=frozenset({(100, 100), (105, 100)}),
        current_floor=frozenset({(1, 0)}), current_occupied=frozenset({(21, 20)}),
        current_fine_occupied=frozenset({(105, 100)}), position_map=(0., 0., 0.),
        map_from_initial=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
        floor_height=-.32, primary_current_floor_cells=1, auxiliary_current_floor_cells=0)
    original = replace(snapshot)
    runtime.planning_generation = 0
    runtime.mission_generation = 0
    assert runtime._routing_view(snapshot) is snapshot
    # A concurrent mission publication cannot change scope midway through a plan.
    runtime.mission_generation = 1
    assert runtime._routing_view(snapshot) is snapshot
    runtime.planning_generation = 1
    routed = runtime._routing_view(snapshot)
    if arm == 'current_pair_return':
        assert routed.floor == snapshot.current_floor
        assert routed.occupied == snapshot.current_occupied
        assert routed.fine_occupied == snapshot.current_fine_occupied
        assert runtime.routing_memory_scope == 'latest_mapped_pair'
    else:
        assert routed is snapshot and runtime.routing_memory_scope == 'persistent'
    assert snapshot == original  # Accumulated predictive-clearance input stays intact.
    assert routed.frame == snapshot.frame and routed.measured_ns == snapshot.measured_ns

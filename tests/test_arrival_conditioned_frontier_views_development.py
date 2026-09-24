from types import SimpleNamespace

import numpy as np

from lewm.arrival_conditioned_frontier_views_development import ArrivalConditionedFrontierVisits
from lewm.nearby_panorama_directed_view_development import NearbyPanoramaFrontierVisits


def test_distant_completed_view_preserves_frontier_then_allows_closer_observation():
    floor = frozenset((x, 0) for x in range(8))
    snapshot = SimpleNamespace(floor=floor, occupied=frozenset(), measured_ns=10, frame=10)
    route = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER', target_map_xy_m=[.325, .025],
        route_cells=[[x, 0] for x in range(7)])
    original = NearbyPanoramaFrontierVisits(standoff_m=.5)
    revised = ArrivalConditionedFrontierVisits(standoff_m=.5)
    for visits in (original, revised):
        visits.excluded = {(-1, 0)}
        visits.visit = dict(target_xy_m=[.325, .025], heading_rad=0., aligned_ns=1,
            started_ns=0, standoff_view=True)
        assert visits.advance(snapshot, np.array([0., .025]), 0., [2., 0.], 10, route)[0] == 'replan'
    assert (6, 0) in original.excluded  # Reproduces the premature retirement.
    assert revised.excluded == {(-1, 0)}
    assert revised.events[-1]['exclusion_deferred_until_arrival']
    assert revised.advance(snapshot, np.array([0., .025]), 0., [2., 0.], 11, route)[0] == 'route'
    action, receipt = revised.advance(snapshot, np.array([.27, .025]), 0., [2., 0.], 12, route)
    assert action == 'view'
    assert 'standoff_view' not in receipt
    snapshot.measured_ns = 14
    assert revised.advance(snapshot, np.array([.27, .025]), receipt['heading_rad'],
        [2., 0.], 14, route)[0] == 'replan'
    assert (6, 0) in revised.excluded
    assert not revised.events[-1]['exclusion_deferred_until_arrival']
    assert revised.standoff_m == .5


def test_completed_arrival_panorama_can_support_a_nearby_directed_view():
    visits = ArrivalConditionedFrontierVisits(standoff_m=.5, panorama=True)
    visits.approach_after_view.add((6, 0))
    snapshot = SimpleNamespace(floor=frozenset((x, 0) for x in range(12)),
        occupied=frozenset(), measured_ns=10, frame=10)
    position = np.array([.27, .025])
    route = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        target_map_xy_m=[.325, .025], route_cells=[[5,0],[6,0]])
    action, receipt = visits.advance(snapshot, position, 0., [2.,0.], 10, route)
    assert action == 'view' and 'standoff_view' not in receipt
    for index, heading in enumerate(receipt['panorama_headings_rad']):
        snapshot.measured_ns = 11+index
        action, _ = visits.advance(snapshot, position, heading, [2.,0.], 11+index, route)
    assert action == 'replan'
    assert not visits.events[-1]['exclusion_deferred_until_arrival']
    # This next query reproduced the native KeyError: a completed non-standoff
    # panorama was missing the start position required by nearby-view selection.
    nearby = dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',
        target_map_xy_m=[.475,.025], route_cells=[[5,0],[6,0],[7,0],[8,0],[9,0]])
    action, receipt = visits.advance(snapshot, position, 0., [2.,0.], 20, nearby)
    assert action == 'view'
    assert receipt['nearby_completed_panorama']['fresh_directed_view_required']
    assert visits.events[-1]['view_start_map_xy_m'] == position.tolist()

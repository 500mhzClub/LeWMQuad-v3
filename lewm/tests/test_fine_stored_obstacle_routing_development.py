from types import SimpleNamespace
import pytest
import numpy as np
from lewm.fine_stored_obstacle_routing_development import proposer


def test_same_samples_at_finer_resolution_and_missing_evidence_rejected():
    floor={(x,y) for x in range(-3,4) for y in range(-4,2)}
    snapshot=SimpleNamespace(occupied={(0,11)},fine_occupied={(2,58)})
    route=proposer(snapshot)(floor,snapshot.occupied,[.025,.11],[.025,-.15])
    assert route['route_cells']
    assert route['nominal_radius_m']==.45
    assert not route['motion_permitted']
    assert not proposer(snapshot)(floor,snapshot.occupied,[.025,.13],[.025,-.15])['route_cells']
    with pytest.raises(ValueError,match='coverage'):
        proposer(SimpleNamespace(occupied={(0,11)},fine_occupied=set()))


def test_spatial_query_matches_all_cells_for_crossing_distant_and_degenerate_segments():
    from lewm.fine_stored_obstacle_routing_development import FineCellClearance,fine_distances
    rng=np.random.default_rng(71)
    cells={tuple(map(int,row)) for row in rng.integers(-450,450,size=(400,2))}
    geometry=FineCellClearance(cells)
    segments=[(np.array([0.,0.]),np.array([0.,0.])),
        (np.array([-4.9,-4.9]),np.array([4.9,4.9]))]
    segments += [(rng.uniform(-4.9,4.9,2),rng.uniform(-4.9,4.9,2)) for _ in range(100)]
    for a,b in segments:
        assert geometry.minimum(a,b)==float(fine_distances(a,b,geometry.cells).min())

import numpy as np
from lewm.fine_stored_obstacle_routing_development import FineCellClearance,fine_distances


def test_pruned_minimum_matches_full_geometry_including_crossing_and_stationary_segments():
    rng=np.random.default_rng(20260913)
    cells=np.unique(rng.integers(-200,200,(2500,2)),axis=0)
    geometry=FineCellClearance(cells.tolist())
    for i in range(200):
        a,b=rng.uniform(-2.5,2.5,(2,2))
        if i%4==0:b=a.copy()
        if i%4==1:b[0]=a[0]
        expected=float(fine_distances(a,b,cells).min())
        np.testing.assert_allclose(geometry.minimum(a,b),expected,rtol=0,atol=1e-12)
    assert FineCellClearance([]).minimum([0.,0.],[1.,1.]) is None
    tangent=FineCellClearance([(0,0)])
    assert tangent.minimum([-.1,0.],[.1,0.])==0.

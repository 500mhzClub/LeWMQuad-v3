import numpy as np
from lewm.observed_floor_waypoint_development import segment_cells as reference
from lewm.vectorized_connector_routing_development import segment_cells


def test_supercover_preserves_corners_parallel_axes_and_roundoff_boundaries():
    pairs=[([0.,0.],[0.,0.]),([-.2,.1],[.3,.1]),([.1,-.2],[.1,.3]),
        ([-.25,-.25],[.25,.25]),([4.8,-4.8],[4.9,-4.9])]
    for eps in (0.,1e-16,-1e-16,1e-13,-1e-13,1e-12,-1e-12):
        pairs.extend([([.1+eps,.1],[.3,.1+eps]),([.1+eps,.1],[.1+eps,.1])])
    rng=np.random.default_rng(20260913)
    for _ in range(100):
        a=rng.uniform(-1.,1.,2);pairs.append((a,a+rng.uniform(-.4,.4,2)))
    for a,b in pairs:assert segment_cells(a,b)==reference(a,b)

"""Keep 1 cm stored obstacle cells for continuous entry into the coarse floor route."""
import numpy as np
from functools import lru_cache
from scipy.spatial import cKDTree
from lewm.eligible_floor_registration_development import bind
from lewm.partial_height_round_trip_development import PartialHeightMap
from lewm.continuous_start_connector_development import ContinuousStartConnectorRuntime
from lewm.vectorized_connector_routing_development import propose as grid_propose
from lewm.observed_geometry_refinement_development import segment_cell_distances
from lewm import process_mapped_runtime_development as mapping_process

fine_distances=bind(segment_cell_distances,CELL_M=.01)


class FineCellClearance:
    """Exact segment/cell distance after a conservative spatial-index query."""
    def __init__(self,cells):
        self.cells=np.asarray(sorted(cells),dtype=int).reshape(-1,2)
        if len(self.cells)>40000 or np.any(self.cells < -500) or np.any(self.cells>=500):
            raise ValueError('bounded integer observed cells required')
        self.low=self.cells*.01;self.high=self.low+.01
        self.tree=cKDTree((self.low+self.high)*.5) if len(self.cells) else None
        self.coarse_cells=frozenset(tuple(map(int,k)) for k in self.cells//5)

    def minimum(self,start,end):
        a,b=np.asarray(start,float),np.asarray(end,float)
        if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([a,b]).all() or np.max(np.abs([a,b]))>5.:
            raise ValueError('bounded finite segment required')
        if not len(self.cells):return None
        midpoint=(a+b)*.5
        _,nearest=self.tree.query(midpoint)
        gap=np.maximum(np.maximum(self.low[nearest]-midpoint,midpoint-self.high[nearest]),0.)
        upper=float(np.linalg.norm(gap))
        # The midpoint-to-square distance bounds the segment minimum above.
        # Every closer square's centre is within half the segment length,
        # that upper bound, and the square's half diagonal of the midpoint.
        radius=float(np.linalg.norm(b-a))*.5+upper+np.sqrt(2.)*.005+1e-12
        indices=self.tree.query_ball_point(midpoint,radius)
        return float(fine_distances(a,b,self.cells[indices]).min())


@lru_cache(maxsize=2)
def _cached_clearance(cells):
    return FineCellClearance(cells)


def cached_clearance(cells):
    key=cells if isinstance(cells,frozenset) else frozenset(tuple(row) for row in cells)
    return _cached_clearance(key)


class FineStoredMap(PartialHeightMap):
    retain_fine_obstacles=True


def proposer(snapshot):
    clearance=cached_clearance(snapshot.fine_occupied)
    fine=clearance.cells
    # Every coarse occupied cell must still have its original fine observations.
    if clearance.coarse_cells != snapshot.occupied:
        raise ValueError('fine and coarse stored obstacle coverage must agree')
    def clear(start,end,occupied,radius):
        distance=clearance.minimum(start,end)
        return distance is None or distance>radius+1e-12
    def propose(floor,occupied,position,goal,*,radius_m=.45,excluded_frontiers=()):
        result=grid_propose(floor,occupied,position,goal,radius_m=radius_m,connector_clear=clear,
            excluded_frontiers=excluded_frontiers)
        return result|dict(start_connector_geometry='continuous_disk_against_stored_1cm_cells',
            stored_obstacle_cell_m=.01,coarse_route_inflation_unchanged=True)
    return propose


class FineStoredObstacleRuntime(ContinuousStartConnectorRuntime):
    def _routing_proposer(self,snapshot):return proposer(snapshot)


def initialize_fine_mapping():
    mapping_process.initialize_mapping()
    mapping_process._mapper=FineStoredMap()

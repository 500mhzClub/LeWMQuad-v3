"""One-centimetre current obstacle cells; unchanged 45-cm nominal footprint."""
from dataclasses import replace
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.independent_depth_obstacle_development import IndependentDepthObstacles
from lewm import independent_depth_process_development as process
from lewm import observed_geometry_refinement_development as geometry
from lewm import fresh_obstacle_dispatch_development as fresh
from lewm import stopping_margin_dispatch_development as stopping
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.veto_view_round_trip_development import VetoViewRoundTripRuntime

CELL_M=.01
FRAME='current_body_1cm_grid'


class FineDepthObstacles(IndependentDepthObstacles):
    _observe=bind(IndependentDepthObstacles._observe,CELL_M=CELL_M)

    def observe(self,*args,**kwargs):
        current=super().observe(*args,**kwargs)
        if current is None:return None
        # Cells outside this two-metre square cannot intersect the supported
        # origin-to-endpoint (<=0.5 m) connector plus its 0.45 m radius.
        cells=frozenset(k for k in current.occupied if all(-100<=v<100 for v in k))
        self.receipts[-1].update(obstacle_cell_m=CELL_M,obstacle_crop_half_extent_m=1.,
            retained_obstacle_cells=len(cells))
        return replace(current,occupied=cells,coordinate_frame=FRAME)


def initialize_fine_obstacles():
    process.initialize_obstacles()
    process._observer=FineDepthObstacles()


_distances=bind(geometry.segment_cell_distances,CELL_M=CELL_M)
_connector=bind(geometry.nominal_connector,segment_cell_distances=_distances)


def connector(start,end,cells,*,radius_m):
    if not np.array_equal(start,[0.,0.]) or np.linalg.norm(end)>.5 or radius_m!=.45:
        raise ValueError('cropped fine grid requires body-origin connector <=0.5 m and radius 0.45 m')
    return _connector(start,end,cells,radius_m=radius_m)|dict(obstacle_cell_m=CELL_M)


_original=bind(fresh.dispatch_request,nominal_connector=connector)


def original(plan,current,*,now_ns):
    if current is not None and current.coordinate_frame!=FRAME:
        raise ValueError('explicit one-centimetre body-frame obstacle grid required')
    return _original(plan,current,now_ns=now_ns)


dispatch_request=bind(stopping.dispatch_request,original=original,nominal_connector=connector)


class _FineDispatch(stopping._StoppingDispatch):
    request=bind(PacedMultirateController.request,dispatch_request=dispatch_request)


class FineObstacleRoundTripRuntime(VetoViewRoundTripRuntime,_FineDispatch):
    pass

"""Current paired plane orientation plus complete raw-pixel floor-height coverage.

This changes the routing floor classifier. It does not establish foot support,
fill invalid rays or change the independently retained raw obstacle geometry.
"""
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.floor_pose_registration_development import unit
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.multirate_routing_map_development import Geometry
from lewm.local_floor_routing_map_development import LocalCoverageGeometry
from lewm.partial_height_round_trip_development import PartialHeightMap
from lewm.partial_floor_height_development import read_pose
from lewm.current_pair_routing_memory_development import CapturedCurrentPairMap


def current_paired_plane(primary,auxiliary,up):
    selector=PairedHeightCandidates(primary,auxiliary)
    clouds=[selector(p['depth_m'],p['valid'],mount,up)[0]
        for p,mount in zip(selector.packets,selector.mounts,strict=True)]
    return fit_joint_plane(*clouds,up)|dict(candidate_selection=selector.receipt)


class _RawValidQuadGeometry(Geometry):
    def index(self,depth,valid,up):
        return dict(ground_cells=valid[:-1,:-1]&valid[1:,:-1]&valid[:-1,1:]&valid[1:,1:])


class CurrentPlaneCoverageGeometry(LocalCoverageGeometry):
    def __init__(self,plane_available):
        super().__init__();self.plane_available=bool(plane_available)

    def floor_coverage(self,depth,valid,*args,**kwargs):
        if not self.plane_available:
            return super().floor_coverage(depth,valid,*args,**kwargs)|dict(
                current_paired_plane_coverage=False,unavailable_plane_uses_original_local_coverage=True)
        geometry=_RawValidQuadGeometry()
        try:result=geometry.floor_coverage(depth,valid,*args,**kwargs)
        finally:geometry.close()
        return result|dict(current_paired_plane_coverage=True,
            current_plane_replaces_pixel_mesh_orientation=True,coverage_depth_is_raw=True,
            complete_valid_pixel_rectangle_required=True,all_pixel_height_band_m=.01)


class CurrentPlaneCoverageMap(PartialHeightMap):
    _read_pose=staticmethod(read_pose)

    def update(self,policy,depth,evidence,*,auxiliary_depth,measured_ns):
        if self.failed:raise ValueError('routing-map failure latched')
        try:
            p,R,pose=self._read_pose(evidence,identity=self.identity,now_ns=measured_ns)
            if self.B is None:
                initial_up=np.asarray(policy['sensor_state']['sensed']['specific_force']['values']).mean(0)
                magnitude=np.linalg.norm(initial_up)
                if not 8<=magnitude<=12:raise ValueError('initial gravity magnitude inconsistent')
                initial_up=initial_up/magnitude
            else:initial_up=self.B[2]
            self.last_floor_plane=current_paired_plane(depth,auxiliary_depth,unit(R.T@initial_up))
            update=bind(PartialHeightMap.update,current_measured_floor_pose=self._read_pose,
                Geometry=lambda:CurrentPlaneCoverageGeometry(self.last_floor_plane['available']))
            return update(self,policy,depth,evidence,auxiliary_depth=auxiliary_depth,measured_ns=measured_ns)
        except Exception:
            self.failed=True;raise


class CurrentPlaneFloorRoutingMap(CapturedCurrentPairMap,CurrentPlaneCoverageMap):
    pass


def initialize_mapping():
    from lewm import process_mapped_runtime_development as process
    from lewm.two_cm_floor_extent_development import configure
    configure();process.initialize_mapping();process._mapper=CurrentPlaneFloorRoutingMap()

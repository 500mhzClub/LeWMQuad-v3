"""Require measured floor on every image quad overlapping the projected square.

The original raw-depth, validity, height-band and visibility checks are retained.
Pixels wholly outside the projected square cannot veto that square's coverage.
The unavailable-current-plane fallback and obstacle mapping remain unchanged.
"""
import numpy as np
from numba import njit

from lewm.body_projected_floor_geometry_development import T, FOCAL
from lewm.current_plane_floor_coverage_development import (
    CurrentPlaneCoverageGeometry, CurrentPlaneCoverageMap, _RawValidQuadGeometry)
from lewm.current_pair_routing_memory_development import CapturedCurrentPairMap
from lewm.eligible_floor_registration_development import bind
from lewm.joint_visual_floor_map_development import GRID, CELL_M


def quad_intersection(uv, x, y):
    """Convex separating-axis test; boundary-touching quads are included."""
    corners = np.stack((np.stack((x,y),-1), np.stack((x+1,y),-1),
        np.stack((x+1,y+1),-1), np.stack((x,y+1),-1)), axis=-2)
    edges = np.roll(uv,-1,axis=0)-uv
    axes = np.concatenate((np.array([[1.,0.],[0.,1.]]),
        np.stack((-edges[:,1],edges[:,0]),-1)))
    polygon = uv@axes.T; pixels = corners@axes.T
    return ((pixels.max(-2) >= polygon.min(0)-1e-9)
        & (pixels.min(-2) <= polygon.max(0)+1e-9)).all(-1)


@njit
def polygon_coverage(good,uv,visible,a,b,original):
    """Same separating-axis predicate, with an early exit at the first bad quad."""
    covered=original.copy()
    axes=np.empty((6,2));lower=np.empty(6);upper=np.empty(6)
    for i in range(len(covered)):
        if covered[i] or not visible[i]:continue
        axes[0,0]=1.;axes[0,1]=0.;axes[1,0]=0.;axes[1,1]=1.
        for edge in range(4):
            following=(edge+1)%4
            axes[edge+2,0]=-(uv[i,following,1]-uv[i,edge,1])
            axes[edge+2,1]=uv[i,following,0]-uv[i,edge,0]
        for axis in range(6):
            lo=1e300;hi=-1e300
            for corner in range(4):
                value=uv[i,corner,0]*axes[axis,0]+uv[i,corner,1]*axes[axis,1]
                lo=min(lo,value);hi=max(hi,value)
            lower[axis]=lo;upper[axis]=hi
        rejected=False
        for y in range(a[i,1],b[i,1]+1):
            for x in range(a[i,0],b[i,0]+1):
                if good[y,x]:continue
                intersects=True
                for axis in range(6):
                    nx,ny=axes[axis,0],axes[axis,1]
                    lo=(x if nx>=0 else x+1)*nx+(y if ny>=0 else y+1)*ny
                    hi=lo+abs(nx)+abs(ny)
                    if hi<lower[axis]-1e-9 or lo>upper[axis]+1e-9:
                        intersects=False
                        break
                if intersects:
                    rejected=True
                    break
            if rejected:break
        covered[i]=not rejected
    return covered


def warmup():
    polygon_coverage(np.ones((1,1),np.bool_),np.zeros((0,4,2)),
        np.zeros(0,np.bool_),np.zeros((0,2),np.int64),np.zeros((0,2),np.int64),np.zeros(0,np.bool_))


class ProjectedPolygonRawGeometry(_RawValidQuadGeometry):
    def floor_coverage(self, depth, valid, map_from_body, translation_map, floor_height, cells=GRID):
        original = super().floor_coverage(depth,valid,map_from_body,translation_map,floor_height,cells)
        covered = original['covered'].copy()
        R,p = np.asarray(map_from_body,float),np.asarray(translation_map,float)
        cells = np.asarray(cells)
        xy = (cells[:,None,:]+np.array([[0,0],[1,0],[1,1],[0,1]]))*CELL_M
        world = np.concatenate((xy,np.full((*xy.shape[:-1],1),floor_height)),axis=-1)
        camera = ((world-p)@R-T[:3,3])@T[:3,:3]
        z = camera[...,2]
        uv = camera[...,:2]/np.maximum(z[...,None],1e-12)*FOCAL+[319.5,239.5]
        visible = ((z>=.2)&(z<=5.)).all(1)&(uv.min(1)-1e-9>=0).all(1)&(uv.max(1)+1e-9<[639,479]).all(1)
        near = np.abs(self.body_projection(depth)@R[2]+p[2]-floor_height)<=.01
        good = self.index(depth,valid,R[2])['ground_cells'] & near[:-1,:-1]&near[:-1,1:]&near[1:,:-1]&near[1:,1:]
        a,b = original['projected_lower_xy'],original['projected_upper_xy']
        covered=polygon_coverage(good,uv,visible,a,b,covered)
        return original | dict(covered=covered,
            projected_polygon_coverage=True, complete_overlapping_pixel_quads_required=True,
            rectangle_only_rejections_removed=int((covered&~original['covered']).sum()),
            complete_valid_pixel_rectangle_required=False,
            all_pixel_height_band_m=.01, ground_support_approved=False)


class ProjectedPolygonCoverageGeometry(CurrentPlaneCoverageGeometry):
    def floor_coverage(self,depth,valid,*args,**kwargs):
        if not self.plane_available:
            return super().floor_coverage(depth,valid,*args,**kwargs)
        geometry = ProjectedPolygonRawGeometry()
        try:
            result = geometry.floor_coverage(depth,valid,*args,**kwargs)
        finally:
            geometry.close()
        return result | dict(current_paired_plane_coverage=True,
            current_plane_replaces_pixel_mesh_orientation=True,coverage_depth_is_raw=True)


class ProjectedPolygonCoverageMap(CurrentPlaneCoverageMap):
    update = bind(CurrentPlaneCoverageMap.update, CurrentPlaneCoverageGeometry=ProjectedPolygonCoverageGeometry)


class ProjectedPolygonFloorRoutingMap(CapturedCurrentPairMap,ProjectedPolygonCoverageMap):
    pass


def initialize_mapping():
    from lewm import process_mapped_runtime_development as process
    from lewm.two_cm_floor_extent_development import configure
    configure();warmup();process.initialize_mapping();process._mapper=ProjectedPolygonFloorRoutingMap()

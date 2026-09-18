import numpy as np

from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.body_projected_floor_geometry_development import T, FOCAL
from lewm.current_plane_floor_coverage_development import _RawValidQuadGeometry
from lewm.projected_polygon_floor_coverage_development import (
    ProjectedPolygonRawGeometry,quad_intersection,polygon_coverage)


def test_intersection_includes_crossings_and_touch_but_not_empty_rectangle_corners():
    uv=np.array([[3.,0.],[6.,3.],[3.,6.],[0.,3.]])
    x=np.array([0,2,6,0]);y=np.array([0,2,3,2])
    expected=[False,True,True,True]
    np.testing.assert_array_equal(quad_intersection(uv,x,y),expected)
    np.testing.assert_array_equal(quad_intersection(uv[::-1],x,y),expected)


def test_compiled_coverage_matches_independent_predicate_for_each_bad_quad():
    polygons=(np.array([[3.,0.],[6.,3.],[3.,6.],[0.,3.]]),
        np.array([[1.,1.],[5.,0.],[6.,5.],[2.,6.]]))
    for uv in polygons:
        for polygon in (uv,uv[::-1]):
            a=np.floor(polygon.min(0)-1e-9).astype(int)
            # This test covers bounded visible polygons; shift away from zero.
            polygon=polygon+1
            a=np.floor(polygon.min(0)-1e-9).astype(int)
            b=np.floor(polygon.max(0)+1e-9).astype(int)
            for y in range(a[1],b[1]+1):
                for x in range(a[0],b[0]+1):
                    good=np.ones((9,9),bool);good[y,x]=False
                    got=polygon_coverage(good,polygon[None],np.array([True]),a[None],b[None],np.array([False]))[0]
                    expected=not quad_intersection(polygon,np.array([x]),np.array([y]))[0]
                    assert got==expected


def test_outside_defect_removed_but_inside_obstacle_and_missing_depth_rejected():
    c=2**-.5
    body_R=np.array([[c,-c,0],[c,c,0],[0,0,1.]])
    R,p=reference_pose(body_R,np.array([0.,0.,.3]))
    yy,xx=np.mgrid[:480,:640]
    rays=np.stack(((xx-319.5)/FOCAL,(yy-239.5)/FOCAL,np.ones_like(xx)),axis=-1)
    direction=rays@T[:3,:3].T@R.T;origin=p+R@T[:3,3]
    depth=np.divide(-origin[2],direction[...,2],out=np.zeros_like(xx,dtype=float),where=direction[...,2]<0)
    valid=(depth>=.2)&(depth<=5.)
    depth=np.where(valid,depth,0).astype(np.float32)
    cell=np.array([[12,12]])
    world=np.column_stack(((cell[0]+np.array([[0,0],[1,0],[1,1],[0,1]]))*.05,np.zeros(4)))
    camera=((world-p)@R-T[:3,3])@T[:3,:3]
    uv=camera[:,:2]/camera[:,2:]*FOCAL+[319.5,239.5]
    low=np.floor(uv.min(0)).astype(int);high=np.floor(uv.max(0)).astype(int)
    y,x=np.mgrid[low[1]+1:high[1],low[0]+1:high[0]]
    touches=np.zeros(x.shape,bool)
    for dx in (-1,0):
        for dy in (-1,0):
            touches|=quad_intersection(uv,x+dx,y+dy)
    outside=np.argwhere(~touches)[0];outside=(y[tuple(outside)],x[tuple(outside)])
    centre=np.floor(uv.mean(0)).astype(int);inside=(centre[1],centre[0])

    def classified(d,v):
        answers=[]
        for cls in (_RawValidQuadGeometry,ProjectedPolygonRawGeometry):
            g=cls()
            try:answers.append(bool(g.floor_coverage(d,v,R,p,0.,cell)['covered'][0]))
            finally:g.close()
        return answers

    assert classified(depth,valid)==[True,True]
    changed=depth.copy();changed[outside]-=.03
    assert classified(changed,valid)==[False,True]
    changed=depth.copy();changed[inside]-=.03
    assert classified(changed,valid)==[False,False]
    changed=depth.copy();v=valid.copy();changed[inside]=0;v[inside]=False
    assert classified(changed,v)==[False,False]

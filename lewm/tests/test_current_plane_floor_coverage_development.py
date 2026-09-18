import numpy as np
from lewm.current_plane_floor_coverage_development import CurrentPlaneCoverageGeometry,current_paired_plane
from lewm.local_floor_routing_map_development import LocalCoverageGeometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical,reference_pose


def packets():
    rng=np.random.default_rng(2026091502);yy,xx=np.indices((480,640))
    rays=np.stack(((xx+.5-320)/FOCAL,(yy+.5-240)/FOCAL,np.ones_like(xx)),axis=-1)
    result=[]
    for E in (np.asarray(BODY_FROM_OPTICAL),body_from_optical()):
        z=rays@E[2,:3]
        depth=np.divide(-.32-E[2,3],z,out=np.zeros_like(z),where=z<0)
        valid=(depth>=.2)&(depth<=5);depth[valid]+=rng.normal(0,.002,valid.sum())
        valid&=(depth>=.2)&(depth<=5);depth[~valid]=0
        result.append(dict(depth_m=depth.astype(np.float32),valid=valid))
    return result


def test_noisy_floor_covered_but_invalid_ray_height_outlier_and_frustum_rejected():
    primary,aux=packets();plane=current_paired_plane(primary,aux,np.array([0.,0.,1.]))
    assert plane['available']
    R,p=reference_pose(np.eye(3),np.zeros(3));cells=np.array([[9,-2],[18,-3],[-50,0]])
    before={k:v.copy() for k,v in aux.items()}
    def cover(packet):
        g=CurrentPlaneCoverageGeometry(True)
        try:return g.floor_coverage(packet['depth_m'],packet['valid'],R,p,-.32,cells=cells)
        finally:g.close()
    result=cover(aux);assert result['covered'].tolist()==[True,True,False]
    for k in aux:np.testing.assert_array_equal(aux[k],before[k])
    xy=(result['projected_lower_xy'][0]+result['projected_upper_xy'][0])//2;x,y=xy
    invalid={k:v.copy() for k,v in aux.items()};invalid['valid'][y,x]=False;invalid['depth_m'][y,x]=0
    assert not cover(invalid)['covered'][0]
    raised={k:v.copy() for k,v in aux.items()};raised['depth_m'][y,x]+=.04
    assert not cover(raised)['covered'][0]


def test_unavailable_plane_preserves_original_local_coverage():
    primary,aux=packets();R,p=reference_pose(np.eye(3),np.zeros(3));cells=np.array([[9,-2],[18,-3]])
    a=CurrentPlaneCoverageGeometry(False);b=LocalCoverageGeometry()
    try:
        x=a.floor_coverage(aux['depth_m'],aux['valid'],R,p,-.32,cells=cells)
        y=b.floor_coverage(aux['depth_m'],aux['valid'],R,p,-.32,cells=cells)
        np.testing.assert_array_equal(x['covered'],y['covered'])
        assert x['unavailable_plane_uses_original_local_coverage']
    finally:a.close();b.close()

import numpy as np
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates, select_clouds
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.joint_measured_floor_plane_development import fit_joint_plane


def test_noisy_plane_with_invalid_rays_and_unchanged_inputs():
    rng = np.random.default_rng(2026091501)
    yy,xx = np.indices((480,640))
    rays = np.stack(((xx+.5-320)/FOCAL,(yy+.5-240)/FOCAL,np.ones_like(xx)),axis=-1)
    mounts=(np.asarray(BODY_FROM_OPTICAL),body_from_optical());packets=[]
    for E in mounts:
        z = rays@E[2,:3]
        depth = np.divide(-.32-E[2,3],z,out=np.zeros_like(z),where=z<0)
        valid=(depth>=.2)&(depth<=5)
        depth[valid]+=rng.normal(0,.002,valid.sum())
        valid &= (depth>=.2)&(depth<=5)
        valid[250,322]=False;depth[~valid]=0
        packets.append(dict(depth_m=depth.astype(np.float32),valid=valid))
    copies=[(p['depth_m'].copy(),p['valid'].copy()) for p in packets]
    selector=PairedHeightCandidates(*packets)
    pairs=[selector(p['depth_m'],p['valid'],E,np.array([0.,0.,1.])) for p,E in zip(packets,mounts)]
    plane=fit_joint_plane(*(p[0] for p in pairs),np.array([0.,0.,1.]))
    assert plane['available'] and abs(plane['offset_body_m']-.32)<.001
    assert selector.receipt['selected_count']>1000
    for packet,copy,(points,mask) in zip(packets,copies,pairs):
        assert np.array_equal(packet['depth_m'],copy[0]) and np.array_equal(packet['valid'],copy[1])
        assert not mask[62,80]  # sampled pixel (250,322) is invalid
        assert len(points)==mask.sum()


def test_diffuse_vertical_wall_is_not_a_dominant_floor_cluster():
    y,z=np.meshgrid(np.linspace(-1,1,80),np.linspace(-.6,-.16,80))
    points=np.column_stack((np.ones(y.size),y.ravel(),z.ravel()))
    masks,receipt=select_clouds((points,np.empty((0,3))),[0.,0.,1.])
    assert not masks[0].any() and receipt['initial_height_cluster_fraction']<.25


def test_small_coherent_population_remains_visible_to_downstream_count_gate():
    x,y=np.meshgrid(np.linspace(.5,1,7),np.linspace(-.3,.3,7))
    points=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,-.32)))
    masks,receipt=select_clouds((points,np.empty((0,3))),[0.,0.,1.])
    assert masks[0].all() and receipt['selected_count']==49
    plane=fit_joint_plane(points[masks[0]],np.empty((0,3)),[0.,0.,1.])
    assert not plane['available'] and plane['candidate_count']==49

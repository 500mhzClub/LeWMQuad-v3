"""Dense-reference comparisons, threshold boundaries and unchanged contracts."""
import numpy as np
import pytest

from lewm import eligible_floor_cell_index_development as candidate
from lewm.floor_footprint_bounds_development import observed_floor_cell_index as original
from lewm.tests.test_floor_footprint_bounds_development import scene
from lewm.causal_sensor_state import SensorContractError


def same(depth,valid,up):
    before = (depth.tobytes(),valid.tobytes(),np.asarray(up).tobytes())
    old = original(depth,valid,up); new = candidate.observed_floor_cell_index(depth,valid,up)
    assert set(old) == set(new)
    for key in old:
        assert old[key].dtype == new[key].dtype
        assert old[key].shape == new[key].shape
        assert old[key].tobytes() == new[key].tobytes()
        assert not new[key].flags.writeable
    assert before == (depth.tobytes(),valid.tobytes(),np.asarray(up).tobytes())
    return new


@pytest.mark.parametrize('kind',['room','missing','all_missing','random','step','tilted','float32','strided'])
def test_complete_masks_and_prefixes_equal_dense_reference(kind):
    depth, valid = scene(); up = np.array([0.,0.,1.]); rng=np.random.default_rng(90213)
    if kind == 'missing': valid[rng.random(valid.shape)<.3] = False; depth[~valid]=0.
    elif kind == 'all_missing': valid[:]=False; depth[:]=0.
    elif kind == 'random': depth=rng.uniform(.2,5.,depth.shape); valid[:]=True
    elif kind == 'step': depth[:,320:]=np.where(valid[:,320:],np.clip(depth[:,320:]+.2,.2,5.),0.)
    elif kind == 'tilted': up=np.array([.15,-.10,1.]);up/=np.linalg.norm(up)
    elif kind == 'float32': depth=depth.astype(np.float32)
    elif kind == 'strided': depth=depth[:,::-1];valid=valid[:,::-1]
    same(depth,valid,up)


@pytest.mark.parametrize('seed',range(6))
def test_noisy_floor_and_near_limit_depths(seed):
    depth,valid=scene();rng=np.random.default_rng(seed)
    depth[valid]=np.clip(depth[valid]+rng.normal(0,.002,valid.sum()),.2,5.)
    depth[350:360,100:110]=np.nextafter(.2,np.inf)
    valid[350:360,100:110]=True
    depth[380:390,400:410]=np.nextafter(5.,0.)
    valid[380:390,400:410]=True
    same(depth,valid,[0.,0.,1.])


@pytest.mark.parametrize('fault',['shape','mask_dtype','invalid_nonzero','nan','near','far','up'])
def test_original_input_rejections_preserved(fault):
    depth,valid=scene();up=[0.,0.,1.]
    if fault=='shape': depth=depth[:-1]
    elif fault=='mask_dtype': valid=valid.astype(np.int32)
    elif fault=='invalid_nonzero': valid[400,300]=False;depth[400,300]=1.
    elif fault=='nan': depth[400,300]=np.nan
    elif fault=='near': valid[400,300]=True;depth[400,300]=np.nextafter(depth.dtype.type(.2),depth.dtype.type(0.))
    elif fault=='far': valid[400,300]=True;depth[400,300]=np.nextafter(depth.dtype.type(5.),depth.dtype.type(np.inf))
    else:up=[0.,0.,2.]
    for fn in (original,candidate.observed_floor_cell_index):
        with pytest.raises(SensorContractError):fn(depth,valid,up)


def test_no_normal_calculation_for_already_rejected_cells(monkeypatch):
    depth=np.zeros((480,640));valid=np.zeros_like(depth,dtype=bool)
    monkeypatch.setattr(candidate.np,'cross',lambda *a,**k:pytest.fail('no eligible cell'))
    result=candidate.observed_floor_cell_index(depth,valid,[0.,0.,1.])
    assert not result['ground_cells'].any()
    assert result['invalid_cell_prefix'][-1,-1]==479*639


def test_uncertain_alignment_uses_original_dense_path(monkeypatch):
    depth,valid=scene();calls=[]
    real_norm=np.linalg.norm
    def norm(value,*args,**kwargs):
        result=real_norm(value,*args,**kwargs)
        if kwargs.get('axis')==1:
            # Force the first gathered normal comparison to its exact threshold.
            result=result.copy();result[0]=abs(value[0,2])/.97
        return result
    marker=object()
    monkeypatch.setattr(candidate.np.linalg,'norm',norm)
    monkeypatch.setattr(candidate,'dense_index',lambda *a:calls.append(a) or marker)
    assert candidate.observed_floor_cell_index(depth,valid,[0.,0.,1.]) is marker
    assert len(calls)==1


def test_output_ownership_matches_original():
    depth,valid=scene();up=np.array([0.,0.,1.])
    result=same(depth,valid,up);before={k:v.tobytes() for k,v in result.items()}
    depth[:]=0.;valid[:]=False;up[:]=0.
    assert before=={k:v.tobytes() for k,v in result.items()}
    with pytest.raises(TypeError):result['new']=1


@pytest.mark.parametrize('offset',[-1e-10,-np.finfo(float).eps,0.,np.finfo(float).eps,1e-10])
def test_alignment_threshold_neighbours_match(offset):
    depth,valid=scene();z=.97+offset
    same(depth,valid,np.array([np.sqrt(1-z*z),0.,z]))


@pytest.mark.parametrize('direction',[-np.inf,0.,np.inf])
def test_fourth_point_planarity_threshold_neighbours_match(direction):
    from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
    depth,valid=scene();depth=depth.astype(float)
    r,c=400,300;transform=np.asarray(BODY_FROM_OPTICAL)
    def ray(rr,cc):return np.array([(cc+.5-320)/FOCAL,(rr+.5-240)/FOCAL,1.])@transform[:3,:3].T
    def point(rr,cc):return depth[rr,cc]*ray(rr,cc)+transform[:3,3]
    a,b,e=point(r,c),point(r,c+1),point(r+1,c)
    normal=np.cross(b-a,e-a);length=np.linalg.norm(normal)
    z=(.003*length-(transform[:3,3]-a)@normal)/(ray(r+1,c+1)@normal)
    if direction: z=np.nextafter(z,direction)
    depth[r+1,c+1]=z
    valid[:]=False;valid[r:r+2,c:c+2]=True;depth[~valid]=0.
    same(depth,valid,[0.,0.,1.])

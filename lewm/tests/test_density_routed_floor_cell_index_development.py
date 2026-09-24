"""Routing never supplies a floor classification or bypasses either validator."""
import numpy as np
import pytest

from lewm import density_routed_floor_cell_index_development as routed
from lewm.floor_footprint_bounds_development import observed_floor_cell_index as original
from lewm.tests.test_floor_footprint_bounds_development import scene
from lewm.causal_sensor_state import SensorContractError


@pytest.mark.parametrize('kind',['room','missing','random','tilted','dense','strided','float64'])
def test_complete_outputs_match_dense_reference(kind):
    d,v=scene();up=np.array([0.,0.,1.]);rng=np.random.default_rng(941)
    if kind=='missing':v[rng.random(v.shape)<.5]=False;d[~v]=0.
    elif kind=='random':d=rng.uniform(.2,5.,d.shape);v[:]=True
    elif kind=='tilted':up=np.array([.4,-.1,1.]);up/=np.linalg.norm(up)
    elif kind=='dense':up=np.array([-1.,0.,0.]);d=np.ones_like(d);v[:]=True
    elif kind=='strided':d=d[:,::-1];v=v[:,::-1]
    elif kind=='float64':d=d.astype(float)
    before=(d.tobytes(),v.tobytes(),up.tobytes())
    a=original(d,v,up);b=routed.observed_floor_cell_index(d,v,up)
    for key in a:
        assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape
        assert a[key].tobytes()==b[key].tobytes()
        assert not b[key].flags.writeable
    assert before==(d.tobytes(),v.tobytes(),up.tobytes())


@pytest.mark.parametrize('route',[False,True])
def test_kernel_choice_does_not_change_mathematical_output(monkeypatch,route):
    d,v=scene();up=[0.,0.,1.]
    monkeypatch.setattr(routed,'prefer_dense',lambda *a:route)
    a=original(d,v,up);b=routed.observed_floor_cell_index(d,v,up)
    assert all(a[k].tobytes()==b[k].tobytes() for k in a)


def test_sparse_and_dense_workloads_select_different_kernels():
    d,v=scene()
    assert routed.prefer_dense(d,v,[0.,0.,1.]) is False
    assert routed.prefer_dense(np.ones_like(d),np.ones_like(v),[-1.,0.,0.]) is True


@pytest.mark.parametrize('route',[False,True])
@pytest.mark.parametrize('fault',['unsampled_nan','unsampled_invalid','unsampled_far','shape','up'])
def test_both_routes_reject_bad_input_even_outside_estimation_grid(monkeypatch,route,fault):
    d,v=scene();up=[0.,0.,1.]
    if fault=='unsampled_nan':d[401,301]=np.nan
    elif fault=='unsampled_invalid':v[401,301]=False;d[401,301]=1.
    elif fault=='unsampled_far':v[401,301]=True;d[401,301]=6.
    elif fault=='shape':d=d[:-1]
    else:up=[0.,0.,2.]
    monkeypatch.setattr(routed,'prefer_dense',lambda *a:route)
    with pytest.raises(SensorContractError):routed.observed_floor_cell_index(d,v,up)


def test_full_grid_is_passed_unchanged_to_selected_kernel(monkeypatch):
    d,v=scene();up=[0.,0.,1.];marker=object()
    def kernel(depth,valid,normal):
        assert depth is d and valid is v and normal is up
        return marker
    monkeypatch.setattr(routed,'eligible_index',kernel)
    assert routed.observed_floor_cell_index(d,v,up) is marker

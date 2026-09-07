from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from lewm.bounded_rotation_geometry_adapter_development import rotation_for_geometry,shape_corrections,adapt_geometry_query


@pytest.mark.parametrize('scale',[0.,1e-12,1e-10])
def test_projection_meets_unchanged_strict_contract_and_bounds_every_box_corner(scale):
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])@np.diag([1+scale,1-scale,1.])
    before=R.copy(); Q,d=rotation_for_geometry(R)
    np.testing.assert_array_equal(R,before)
    np.testing.assert_allclose(Q.T@Q,np.eye(3),rtol=0,atol=1e-12)
    assert abs(np.linalg.det(Q)-1)<=1e-12
    lo=np.array([-.4,-.2,-.5]); hi=np.array([.3,.3,.1])
    bounds=shape_corrections([dict(shape_id='a',lower=lo,upper=hi)],d['matrix_difference_frobenius_upper'])
    for choice in product((False,True),repeat=3):
        point=np.where(choice,lo,hi).astype(np.longdouble)
        difference=(R.astype(np.longdouble)-Q.astype(np.longdouble))@point
        assert np.sqrt(difference@difference)<=bounds['a']


@pytest.mark.parametrize('rotation',[np.diag([1.,1.,-1.]),np.zeros((3,3)),np.eye(3)*1.01,np.full((3,3),np.nan)])
def test_projection_does_not_repair_gross_invalid_pose(rotation):
    with pytest.raises(ValueError): rotation_for_geometry(rotation)


class Geometry:
    def supports(self,joints,directions):
        return {'shapes':[dict(shape_id='a',lower=[-.3,-.2,-.4],upper=[.3,.2,.1])]}


def test_physical_error_hypotheses_and_pose_are_retained_without_history_reset():
    R=np.eye(3); R[0,0]+=1e-10
    s=dict(rotation_initial_body_from_current_body=R.tolist(),position_initial_body_m=[1.,2.,3.],reference_frame=42)
    before=deepcopy(s); errors={'a':.02}
    query,evidence=adapt_geometry_query(Geometry(),None,s,errors)
    assert s==before and errors=={'a':.02}
    assert query['translation_observation_from_body']==[1.,2.,3.]
    assert query['point_error_by_shape']['a']>=.02+evidence['numerical_shape_correction_m']['a']
    assert not evidence['navigation_qualified'] and not evidence['physical_uncertainty_calibrated']


@pytest.mark.parametrize('errors',[None,{}, {'a':-1.},{'a':float('nan')}])
def test_unknown_or_invalid_physical_error_not_silently_zeroed(errors):
    s=dict(rotation_initial_body_from_current_body=np.eye(3),position_initial_body_m=[0.,0.,0.])
    with pytest.raises(ValueError): adapt_geometry_query(Geometry(),None,s,errors)


def test_shape_roster_rejects_reversed_bounds_and_duplicates():
    s=dict(shape_id='a',lower=[0.,0.,0.],upper=[1.,1.,1.])
    with pytest.raises(ValueError): shape_corrections([s,s],1e-12)
    with pytest.raises(ValueError): shape_corrections([s|dict(lower=[2.,0.,0.])],1e-12)


@pytest.mark.parametrize('forward_offset,visible',[(0.,False),(1.5,True)])
def test_startup_projection_reports_fixed_camera_blind_region(forward_offset,visible):
    from scripts.probe_go2_bounded_rotation_geometry_adapter_v1 import startup_projection
    class Surface:
        status='BOUNDED_MEASURED_SURFACE_AVAILABLE'
        anchor=np.array([0.,0.,-.3]); normal=np.array([0.,0.,1.]); up_error=0.; surface_tube_m=.001
        class frame:
            _up=np.array([0.,0.,1.])
    class ShiftedGeometry(Geometry):
        def supports(self,joints,directions):
            result=super().supports(joints,directions)
            for s in result['shapes']:
                s['lower'][0]+=forward_offset; s['upper'][0]+=forward_offset
            return result
    p={'sensor_state':{'sensed':dict(gyro={},specific_force={},joints={})}}
    result=startup_projection(Surface(),ShiftedGeometry(),None,p)
    assert result['complete_frustum_shapes']==int(visible)
    assert result['shapes'][0]['entire_enclosure_before_minimum_range']==(not visible)
    assert not result['contact_modality_present'] and not result['unobserved_floor_support_assumed']

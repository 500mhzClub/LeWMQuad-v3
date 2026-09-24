import numpy as np
from lewm.partial_floor_height_development import fit_gyro_height


def test_narrow_strip_provides_height_without_claiming_measured_normal():
    x=np.linspace(.3,1.5,200);p=np.column_stack((x,np.zeros_like(x),np.full_like(x,-.31)))
    result=fit_gyro_height(p,np.empty((0,3)),np.array([0.,0.,1.]))
    assert result['available'] and not result['normal_measured_from_current_points']
    assert not result['full_plane_qualification'] and abs(result['offset_body_m']-.31)<1e-12
    p[100,2]+=.02
    assert not fit_gyro_height(p,np.empty((0,3)),np.array([0.,0.,1.]))['available']

from types import SimpleNamespace
import numpy as np
import pytest
from lewm.optional_plane_refinement_development import refine_if_supported,original


def test_optional_refinement_preserves_fit_but_rejects_plane_conflict(monkeypatch):
    def lost(*a,**kw):
        raise original.SensorContractError('plane refinement loses an original image inlier')
    monkeypatch.setattr(original,'refine',lost)
    monkeypatch.setattr(original,'fit',lambda *a: (np.eye(3),np.zeros(3),{}))
    candidate=dict(reference=SimpleNamespace(rotation=np.eye(3),position=np.zeros(3)),
        R=np.eye(3),p=np.zeros(3),local_R=np.eye(3),t=np.zeros(3),
        registration=dict(reference_inlier_points_body_m=[[0,0,0]],current_inlier_points_body_m=[[0,0,0]]))
    plane=dict(normal_body=[0,0,1],offset_body_m=.3)
    kwargs=dict(camera='primary',gyro=np.eye(3),last_R=np.eye(3),last_p=np.zeros(3))
    result=refine_if_supported(candidate,plane,plane,**kwargs)
    assert result['p'] is candidate['p']
    assert result['R'] is candidate['R']
    assert result['registration']['measured_plane_refinement']['applied'] is False
    assert 'measured_plane_refinement' not in candidate['registration']
    with pytest.raises(original.PlaneImageConflict):
        refine_if_supported(candidate,plane,plane|dict(offset_body_m=.304),**kwargs)
    monkeypatch.setattr(original,'fit',lambda *a: (np.eye(3),np.array([.03,0,0]),{}))
    with pytest.raises(original.PlaneImageConflict):
        refine_if_supported(candidate,plane,plane,**kwargs)


def test_other_rejections_propagate(monkeypatch):
    def reject(*a,**kw):raise original.SensorContractError('other invalid measurement')
    monkeypatch.setattr(original,'refine',reject)
    with pytest.raises(original.SensorContractError,match='other invalid measurement'):
        refine_if_supported({}, {}, {})

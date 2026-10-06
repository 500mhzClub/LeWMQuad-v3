"""The floor-init change must measure a plane and preserve its frame convention."""
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.navigation_capability_paired_floor_start_development import measured_map_floor_height


def test_qualified_plane_avoids_observed_wall_height_counterexample():
    # C3 dev01: old mesh-candidate median was -0.085249 m, whereas the
    # already-qualified paired depth plane measured floor near -0.319183 m.
    plane=dict(available=True,normal_body=[0.,0.,1.],offset_body_m=.319183)
    assert measured_map_floor_height(plane,np.eye(3),np.zeros(3)) == -.319183
    assert abs(measured_map_floor_height(plane,np.eye(3),np.zeros(3))-(-.085249)) > .23


def test_body_rotation_and_translation_do_not_reanchor_the_floor():
    angle=.3
    R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    p=np.array([1.2,-.7,.12]);height=-.32
    normal=R.T@np.array([0.,0.,1.])
    plane=dict(available=True,normal_body=normal,offset_body_m=p[2]-height)
    assert measured_map_floor_height(plane,R,p) == pytest.approx(height,abs=1e-14)


def test_unqualified_plane_must_use_existing_startup_recovery():
    with pytest.raises(SensorContractError,match='initial measured floor unavailable'):
        measured_map_floor_height(dict(available=False),np.eye(3),np.zeros(3))


def test_sloped_measured_plane_uses_map_origin_not_current_body_xy():
    normal=np.array([.01,.02,1.]);normal/=np.linalg.norm(normal)
    p=np.array([.1,-.2,.03]);height=-.32
    plane=dict(available=True,normal_body=normal,offset_body_m=float(normal@p-normal[2]*height))
    assert measured_map_floor_height(plane,np.eye(3),p) == pytest.approx(height,abs=1e-14)


def test_missing_first_plane_recovers_with_original_start_frame(monkeypatch):
    from types import SimpleNamespace
    from lewm import navigation_capability_paired_floor_start_development as module
    mapper=module.PairedFloorStartupMap()
    monkeypatch.setattr(module.process,'_mapper',mapper)
    monkeypatch.setattr(module,'validate_depth',lambda *a,**k:None)
    monkeypatch.setattr(module,'validate_auxiliary',lambda *a,**k:None)
    planes=iter([dict(available=False),dict(available=True,normal_body=[0.,0.,1.],offset_body_m=.35)])
    monkeypatch.setattr(module,'current_paired_plane',lambda *a:next(planes))
    mapper._read_pose=lambda evidence,**k:(np.array([0.,0.,.03]),np.eye(3),dict(frame=evidence['frame']))
    def ordinary_update(self,*a,**k):
        self.latest=SimpleNamespace(floor_height=self.floor_height)
        return self.latest
    monkeypatch.setattr(module.RecoverableStartupMap,'update',ordinary_update)
    state=dict(valid=np.ones((1,3),bool),values=np.array([[0.,0.,9.81]]))
    command=dict(valid=np.ones((1,3),bool),values=np.zeros((1,3)))
    policy=dict(sensor_state=dict(sensed=dict(specific_force=state),control=dict(applied_command=command)))
    packet=SimpleNamespace(frame=0,measured_ns=0,policy=policy,depth={},auxiliary_depth={})
    snapshot,receipt=module.mapping_update(packet,dict(frame=0))
    assert snapshot is None and receipt['status']=='WAITING_FOR_MEASURED_INITIAL_FLOOR'
    assert mapper.failed is False and mapper.floor_height is None
    original_basis=mapper.B.copy()
    packet.frame=4;packet.measured_ns=400_000_000
    snapshot,receipt=module.mapping_update(packet,dict(frame=4))
    assert snapshot.floor_height == pytest.approx(-.32)
    np.testing.assert_array_equal(mapper.B,original_basis)
    assert receipt['initial_floor_measurement']['frame']==4

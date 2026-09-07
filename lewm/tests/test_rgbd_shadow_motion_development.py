from types import SimpleNamespace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_shadow_motion_development import schedule,priors,ShadowObserver
from scripts.rgbd_shadow_motion_session_development import AppearanceRGBDSession,ShadowMotionSession
from scripts.rgbd_shadow_motion_physical_init_development import AppearancePhysicalInit
from scripts.rgbd_session_development import RGBDSession
from scripts.whole_task_physics_session_development import GeometryFreePhysicalSample
from scripts.run_go2_rgbd_shadow_motion_development_v1 import specification,artifact_names
from scripts.probe_go2_rgbd_motion_scene_development_v1 import pack,ARMS


def test_fixed_bidirectional_tape_and_tail_expiry():
    commands=np.asarray(schedule())
    assert commands.shape==(50,3)
    assert set(commands[:,0])=={0.,.08,-.06,.10}
    assert set(commands[:,2])=={0.,.3,-.3}
    assert np.all(commands[:,1]==0)
    assert sum(np.all(commands==0,axis=1))==20
    altered=schedule();altered[0][0]=99;assert schedule()[0][0]==.08
    velocity,region=priors('a'*64)
    assert velocity.anchor_ns==1_500_000_000
    assert velocity.anchor_ns+(len(commands)+3)*100_000_000<region.valid_until_ns
    assert region.lower_initial_body_m==(-1.25,)*3


def test_shadow_terminal_failure_is_recorded_without_reinvocation_or_command_output():
    velocity,_=priors('a'*64);shadow=ShadowObserver(velocity)
    class Broken:
        calls=0
        def observe(self,*args,**kw):
            self.calls+=1
            raise SensorContractError('deliberate missing observation')
    model=Broken();shadow.model=model
    a=shadow.observe(None,None,None,now_ns=1_500_000_000)
    b=shadow.observe(None,None,None,now_ns=1_600_000_000)
    assert model.calls==1 and shadow.successes==0
    assert a['status']=='TERMINAL_SHADOW_FAILURE'
    assert b['status']=='NOT_REINVOKED_AFTER_SHADOW_FAILURE'
    assert a['state'] is b['state'] is None and a['selects_command'] is b['selects_command'] is False
    b['failure']['chain'].append('external mutation')
    assert shadow.failure['chain']==['deliberate missing observation']


def test_mro_keeps_raw_sensor_and_physical_wrappers_without_old_scene_init():
    mro=ShadowMotionSession.__mro__
    assert RGBDSession in mro and AppearancePhysicalInit in mro
    assert mro.index(GeometryFreePhysicalSample)<mro.index(AppearancePhysicalInit)
    assert 'BoundedFloorPhysicalInit' not in [c.__name__ for c in mro]
    assert 'MovingRGBDSession' not in [c.__name__ for c in mro]
    assert AppearanceRGBDSession.execute_requested_ticks is GeometryFreePhysicalSample.execute_requested_ticks


def test_physical_identity_excludes_every_visual_mesh(monkeypatch):
    import scripts.rgbd_shadow_motion_session_development as module
    def entity(name,ids):return SimpleNamespace(name=name,links=[SimpleNamespace(idx=i,name=f'{name}_{i}') for i in ids])
    floor=entity('ground',[1]);robot=entity('go2',[2,3]);wall=entity('wall',[4]);vfloor=entity('floor_visual',[5]);vwall=entity('wall_visual',[6])
    scene=SimpleNamespace(entities=[floor,robot,wall,vfloor,vwall])
    build=SimpleNamespace(scene=scene,robot=robot,collision_floor=floor,visual_surfaces=[vfloor,vwall])
    session=SimpleNamespace(ctx=SimpleNamespace(build=build),geometry={'wall_boxes':[{'wall_id':'wall'}]},
        _contact_topology={'ground':{1},'robot':{2,3},'support':{3}})
    monkeypatch.setattr(module,'appearance_environment_identity',lambda _: {})
    AppearanceRGBDSession.install_contact_identity(session)
    assert session.object_ids=={1:'ground_plane',4:'wall'}
    assert set(session.link_names)=={1,2,3,4}
    scene.entities.append(entity('undeclared_physical',[7]))
    with pytest.raises(ValueError,match='unresolved'):AppearanceRGBDSession.install_contact_identity(session)


@pytest.mark.parametrize('arm',ARMS)
def test_specification_and_explicit_artifacts_match_fresh_scene(arm):
    spec=specification(arm);definition=pack()
    assert spec['procedural_seed']==definition.physics_seed
    assert spec['appearance_seed']==definition.visual_seed and spec['appearance_arm']==arm
    assert spec['geometry']['spawn_se2_world']==[-.25,-.2,.27]
    assert [r['wall_id'] for r in spec['geometry']['wall_boxes']]==[o.object_id for o in definition.static_objects]
    names=artifact_names({'rgbd_frames':54})
    assert len(names)==len(set(names))==194
    assert all('sealed' not in n and not n.startswith('/') and '..' not in n for n in names)
    assert sum(n.startswith('visual_meshes/') for n in names)==6

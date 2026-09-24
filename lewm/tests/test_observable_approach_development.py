from types import SimpleNamespace

import numpy as np
import pytest

from lewm.observable_approach_development import floor_view_standoff, blocking_limit, ObservableApproachRegions, ObservableApproachTraversal
from lewm.causal_sensor_state import SensorContractError
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.tests.test_observed_traversal_controller_development import Stream
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def floor(height=.4):
    x,y=np.meshgrid(np.linspace(.6,2.5,30),np.linspace(-.5,.5,20))
    p=np.stack((x.ravel(),y.ravel(),np.full(x.size,-height)),axis=1)
    return p,np.tile([0.,0.,1.],(len(p),1))


def surface(distance=3.5,y=(-.5,.5)):
    return {'surface_segments':[{'normal_body_xy':[1.,0.],'offset_body_m':distance,
        'endpoints_body_xy_m':[[distance,y[0]],[distance,y[1]]]}]}


def test_floor_view_margin_depends_on_measured_height_and_fov():
    p,n=floor(); a=floor_view_standoff(p,n,[0.,0.,1.])
    p,n=floor(.5); b=floor_view_standoff(p,n,[0.,0.,1.])
    assert a['floor_points']==600 and a['probe_row']==400
    assert a['minimum_front_standoff_m']>.6
    assert b['minimum_front_standoff_m']>a['minimum_front_standoff_m']
    assert not a['hardware_calibrated']


def test_missing_and_degenerate_floor_support_are_not_fixed_height():
    p,n=floor()
    assert floor_view_standoff(p[:50],n[:50],[0.,0.,1.]) is None
    p[:,1]=0.
    assert floor_view_standoff(p,n,[0.,0.,1.]) is None
    with pytest.raises(SensorContractError): floor_view_standoff(p,n,[0.,0.,2.])


def test_observed_blocker_caps_target_but_remote_support_does_not_invent_blocker():
    p,n=floor(); f=floor_view_standoff(p,n,[0.,0.,1.])
    limit=blocking_limit(surface(),.46,f)
    assert limit['maximum_approach_m']==pytest.approx(3.5-f['minimum_front_standoff_m'])
    assert blocking_limit(surface(y=(2.,3.)),.46,f)['maximum_approach_m'] is None
    assert not limit['free_volume_qualified']
    with pytest.raises(SensorContractError): blocking_limit(surface(),.46,None)


def test_near_corner_cannot_override_visible_front_wall():
    obs=ObservableApproachRegions(ArticulatedCollisionGeometry(URDF)); now=100
    obs.memory.position=np.zeros(3); obs.memory.rotation=np.eye(3); obs.memory.last_ns=now
    obs.memory.latest_surface=surface(); obs.volume=lambda p:{'maximum_radius_m':.46}
    p,n=floor(); obs.floor_view=floor_view_standoff(p,n,[0.,0.,1.])
    obs.corners=[{'point_initial_body_m':[3.5,.7,0.], 'approach_initial_body':[1.,0.,0.],
        'longitudinal_role':'near_boundary','observed_ns':now,'side':'left'}]
    target=obs.target({},now_ns=now)
    assert target['unconstrained_target_body_m'][0]>3.5
    assert target['target_body_m'][0]<3.5-obs.floor_view['minimum_front_standoff_m']+1e-9
    assert target['kind']=='OBSERVED_BLOCKER_LIMITED_REGION'
    obs.memory.latest_surface=surface(.8)
    assert obs.target({},now_ns=now) is None


def test_approach_rechecks_and_only_tightens_target():
    stream=Stream(); geometry=ArticulatedCollisionGeometry(URDF)
    obs=SimpleNamespace(memory=SimpleNamespace(last_ns=None,position=np.zeros(3),rotation=np.eye(3),up_initial=np.array([0.,0.,1.])),relative=None)
    obs.target=lambda *a,**k:{'target_initial_body_m':[2.,0.,0.],'kind':'SYNTHETIC'}
    obs.limit=lambda *a,**k:{'maximum_approach_m':1. if child.tick<6 else 3.}
    obs.clearance=lambda *a,**k:{'all_samples_supported':True,'unknown_samples':0}
    child=ObservableApproachTraversal(geometry,obs)
    updates=[]
    for tick in range(9):
        p,fast,now=stream.frame(tick); obs.memory.last_ns=now
        obs.relative={'motion':{'translation_previous_body_m':[0.,0.,0.]}}
        row=child.observe(p,fast,now_ns=now)
        if row['approach_constraint_update']: updates.append(row['approach_constraint_update'])
    assert len(updates)==1 and child.target['target_initial_body_m']==[1.,0.,0.]

from threading import Lock
from types import SimpleNamespace
import math
import numpy as np
from lewm import veto_view_round_trip_development as subject
from lewm.stopping_margin_dispatch_development import StoppingMarginRoundTripRuntime


def test_translation_veto_requests_view_and_only_measured_turn_releases_it(monkeypatch):
    runtime=subject.VetoViewRoundTripRuntime.__new__(subject.VetoViewRoundTripRuntime)
    runtime.lock=Lock();runtime.view_recovery=None;runtime.mission_generation=0
    runtime.plans=[SimpleNamespace(observed_ns=0,command=(.2,0.,0.))]
    monkeypatch.setattr(StoppingMarginRoundTripRuntime,'request',lambda self,now_ns:dict(
        reason='CURRENT_STOPPING_MARGIN_VETO',command_observation_ns=0,requested_command=[0.,0.,0.]))
    response=runtime.request(now_ns=300_000_000)
    assert response['requested_command']==[0.,0.,0.] and response['view_recovery']['trigger_ns']==300_000_000
    monkeypatch.setattr(StoppingMarginRoundTripRuntime,'_route',lambda *a,**k:dict(route_cells=[[1,1]],status='route'))
    heading=[0.]
    def pose(*args,**kwargs):
        c,s=math.cos(heading[0]),math.sin(heading[0]);return np.zeros(3),np.array([[c,-s,0],[s,c,0],[0,0,1]]),{}
    monkeypatch.setattr(runtime,'_pose',pose)
    snapshot=SimpleNamespace(map_from_initial=np.eye(3))
    assert runtime._route(snapshot,{},[0,1],measured_ns=200_000_000)['status']=='WAITING_FOR_POST_VETO_VIEW'
    assert runtime._route(snapshot,{},[0,1],measured_ns=400_000_000)['route_cells']==[]
    assert runtime.scan_target==math.pi/4
    assert response['view_recovery']['target_heading_rad'] is None
    heading[0]=math.pi/4
    assert runtime._route(snapshot,{},[0,1],measured_ns=500_000_000)['route_cells']==[[1,1]]
    assert runtime.view_recovery is None

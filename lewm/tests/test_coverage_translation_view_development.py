from types import SimpleNamespace

import numpy as np

from lewm.coverage_translation_view_development import CoverageTranslationViewMixin,footprint_extension,filter_translation
from lewm.geometry_progress_pilot_development import ACTIONS


def fixture():
    pred=np.zeros((6,8,5));pred[:,:,3]=1
    pred[1,3:,0]=np.linspace(.05,.20,5)
    floor=frozenset((x,y) for x in range(-20,21) for y in range(-20,21))
    snap=SimpleNamespace(floor=floor,occupied=frozenset())
    selected=dict(action='forward',candidates=[dict(action=a,utility_m=u) for a,u in zip(ACTIONS,[0,4,3,2,1,-1])],
        memory_forecast_candidates=[dict(action=a,nominal_predicted_path_clear=True,
            reserve_recovery_path_clear=False) for a in ACTIONS],
        planned_stopping_projection=dict(candidates=[dict(action=a,projection_clear=True) for a in ACTIONS]))
    return pred,snap,selected


def test_observed_floor_allows_translation_and_pure_turn_is_not_filtered():
    pred,snap,s=fixture()
    result,target=filter_translation(s,pred,snap,np.zeros(3),np.eye(3))
    assert result['action']=='forward' and target is None
    assert not result['translation_footprint_coverage']['rejected']
    s['action']='left_turn';snap.floor=frozenset()
    assert filter_translation(s,pred,snap,np.zeros(3),np.eye(3))==(s,None)


def test_unknown_extension_rejects_translation_but_keeps_original_clearance_rules():
    pred,snap,s=fixture();snap.floor=snap.floor-{(12,0)}
    s['memory_forecast_candidates'][4]['nominal_predicted_path_clear']=False
    result,target=filter_translation(s,pred,snap,np.zeros(3),np.eye(3))
    assert result['action']=='hold' and target==(12,0)
    assert s['action']=='forward'
    assert result['memory_forecast_candidates']==s['memory_forecast_candidates']
    assert result['planned_stopping_projection']==s['planned_stopping_projection']


def test_unknown_shared_hold_motion_is_not_misreported_as_new_action_extension():
    pred,snap,_=fixture();pred[:,:,:2]=pred[1,:,:2];snap.floor=frozenset()
    r=footprint_extension(pred,np.zeros(3),np.eye(3),snap.floor)
    assert r['baseline_unknown_cells']>0
    assert all(c['new_unknown_cells']==0 for c in r['candidates'])
    assert r['current_and_hold_unknown_not_declared_free']


def test_swept_path_checks_interior_even_when_endpoints_have_floor():
    pred,snap,_=fixture();pred[1,3:,0]=np.linspace(.2,1.5,5)
    snap.floor=frozenset((x,y) for x in range(-40,60) for y in range(-40,41))-{(15,0)}
    r=footprint_extension(pred,np.zeros(3),np.eye(3),snap.floor)
    assert [15,0] in r['candidates'][1]['added_cells']


class WeakViewParent:
    def __init__(self):self.mission_generation=0
    def _route(self,*args,**kwargs):
        return dict(status='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW',route_cells=[],view_heading_rad=.3)


class ViewRuntime(CoverageTranslationViewMixin,WeakViewParent):
    pass


def test_weak_visual_recovery_keeps_priority_without_resolving_unknown_patch():
    runtime=ViewRuntime();runtime.coverage_target=(12,0)
    snapshot=SimpleNamespace(floor=frozenset(),occupied=frozenset())
    result=runtime._route(snapshot,{},np.zeros(2),measured_ns=100)
    assert result['status']=='LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW'
    assert runtime.coverage_target==(12,0)
    assert runtime.coverage_view_status=='WEAK_VIEW_RECOVERY_HAS_PRIORITY'


def test_an_actual_obstacle_observation_resolves_the_coverage_request():
    runtime=ViewRuntime();runtime.coverage_target=(12,0)
    snapshot=SimpleNamespace(floor=frozenset(),occupied=frozenset({(12,0)}))
    runtime._route(snapshot,{},np.zeros(2),measured_ns=100)
    assert runtime.coverage_target is None
    assert runtime.coverage_view_status=='REQUESTED_PATCH_OBSERVED'

"""Feature budget/distribution and unchanged reference selection contracts."""
from collections import Counter
import cv2
import numpy as np
import pytest
from lewm.balanced_rgbd_features_development import select_keypoints,BalancedFeatureFrame
from lewm.balanced_multi_reference_rgbd_development import BalancedMultiReferenceRGBDPose,BalancedVisualLedMotion
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose
from lewm.causal_sensor_state import SensorContractError
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_coupled_room_return_v1 import OUTPUT


def keypoints():
    return [cv2.KeyPoint(160*x+2+(i%20)*7,160*y+2+(i//20)*7,3,response=float(10000*(x==0)+100-i))
            for x in range(4) for y in range(3) for i in range(100)]


def test_same_budget_but_spatial_balance_despite_one_dominant_cell():
    selected=select_keypoints(keypoints());counts=Counter((int(p.pt[0]//160),int(p.pt[1]//160)) for p in selected)
    assert len(selected)==600 and len(counts)==12 and set(counts.values())=={50}
    assert [p.pt for p in selected]==[p.pt for p in select_keypoints(keypoints())]


def test_duplicate_orientations_do_not_consume_distinct_point_budget():
    a=cv2.KeyPoint(20,30,3,angle=0,response=1)
    b=cv2.KeyPoint(20,30,3,angle=90,response=2)
    selected=select_keypoints([a,b]);assert len(selected)==1 and selected[0].angle==90


def test_unused_cell_quota_redistributed_and_empty_input_supported():
    points=[p for p in keypoints() if p.pt[0]<160]
    assert len(select_keypoints(points))==300 and select_keypoints([])==[]
    frame=BalancedFeatureFrame(np.zeros((480,640,3),np.uint8),{})
    assert not frame.keypoints and frame.descriptors is None


@pytest.mark.parametrize('x,y',[(640,1),(-1,2),(1,480),(float('nan'),0)])
def test_bad_feature_coordinates_rejected(x,y):
    with pytest.raises(SensorContractError):select_keypoints([cv2.KeyPoint(x,y,3)])


def test_feature_input_is_copied_and_native_contract_not_extended():
    rgb=np.zeros((480,640,3),np.uint8);depth={'x':np.array([1.])}
    f=BalancedFeatureFrame(rgb,depth);rgb[:]=255;depth['x'][:]=2
    assert not f.rgb.any() and f.depth['x'][0]==1
    with pytest.raises(SensorContractError):BalancedFeatureFrame(rgb.astype(float),{})
    assert BalancedMultiReferenceRGBDPose._choose is MultiReferenceRGBDPose._choose
    assert BalancedMultiReferenceRGBDPose._candidate is MultiReferenceRGBDPose._candidate


def test_real_short_stream_uses_only_current_packets_and_latches_fault():
    reader=IntentReturnRGBDReplay(OUTPUT/'nominal_left');model=BalancedVisualLedMotion()
    for i in range(4):
        p,d,f,now=reader.packet(i);r=model.observe(p,d,f,now_ns=now)
        assert r['current_pose'] is not None
        assert r['current_pose']['frame']==i and r['feature_selection']['descriptor_count']<=600
        assert r['reference_selection']['status'] in ('INITIAL_REFERENCE','PRIMARY_ACCEPTED','RECENT_REFERENCE_ACCEPTED')
    r=model.observe({}, {}, {},now_ns=now+100_000_000)
    assert r['current_pose'] is None and r['status']=='VISUAL_TERMINAL_FAILURE'
    p,d,f,now=reader.packet(4);r=model.observe(p,d,f,now_ns=now)
    assert r['current_pose'] is None

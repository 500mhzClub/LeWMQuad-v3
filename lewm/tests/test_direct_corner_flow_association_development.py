from copy import deepcopy
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.direct_corner_flow_association_development import tracked_points, patch_agrees
from lewm.joint_rgbd_rigid_pose_development import register
from lewm.causal_depth_observation_development import FOCAL


def frames(dx=3.):
    rng=np.random.default_rng(2026090917)
    gray=cv2.GaussianBlur(rng.integers(30,220,(480,640),dtype=np.uint8),(3,3),0)
    moved=cv2.warpAffine(gray,np.float32([[1,0,dx],[0,1,0]]),(640,480))
    points=[cv2.KeyPoint(float(x),float(y),8.) for y in (60,140,220,300,380,440) for x in (60,140,220,300,380,460,540,600)]
    depth=dict(depth_m=np.full((480,640),2.,np.float32),valid=np.ones((480,640),bool))
    return SimpleNamespace(gray=gray,keypoints=points,depth=deepcopy(depth)),SimpleNamespace(gray=moved,keypoints=[],depth=deepcopy(depth))


def test_direct_tracks_recover_known_translation_and_pass_original_geometry():
    ref,cur=frames(); before=(ref.gray.copy(),cur.gray.copy())
    values,receipt=tracked_points(ref,cur)
    assert receipt['counts']['valid_depth_pair']>=40
    Q,t,mask,reg=register(*values,gyro_rotation=np.eye(3),mode='joint',frame=1)
    np.testing.assert_allclose(Q,np.eye(3),atol=.001,rtol=0)
    np.testing.assert_allclose(t,[0.,6./FOCAL,0.],atol=.001,rtol=0)
    assert reg['inliers']>=40 and reg['reference_grid_cells']>=6
    assert not receipt['pose_admitted'] and not receipt['original_descriptor_gates_applied']
    np.testing.assert_array_equal(before[0],ref.gray); np.testing.assert_array_equal(before[1],cur.gray)


def test_empty_reference_and_invalid_depth_supply_no_geometry():
    ref,cur=frames(); ref.keypoints=[]
    assert len(tracked_points(ref,cur)[0][0])==0
    ref,cur=frames(); cur.depth['valid'][:]=False
    assert len(tracked_points(ref,cur)[0][0])==0


def test_reference_duplicates_do_not_inflate_correspondences():
    ref,cur=frames(); ref.keypoints=ref.keypoints+ref.keypoints
    values,receipt=tracked_points(ref,cur)
    assert receipt['counts']['distinct_reference_corners']==48 and len(values[0])<=48


def test_photometric_check_rejects_flat_inverted_and_border_patches():
    ref,_=frames(dx=0.)
    assert patch_agrees(ref.gray,ref.gray,[100.,100.],[100.,100.])
    assert not patch_agrees(ref.gray,255-ref.gray,[100.,100.],[100.,100.])
    assert not patch_agrees(np.zeros_like(ref.gray),np.zeros_like(ref.gray),[100.,100.],[100.,100.])
    assert not patch_agrees(ref.gray,ref.gray,[2.,2.],[2.,2.])


@pytest.mark.parametrize('fault',['too_many','wrong_image','nonfinite_corner','nonfinite_flow','reverse_mismatch'])
def test_bounded_inputs_and_flow_failures(monkeypatch,fault):
    ref,cur=frames()
    if fault=='too_many': ref.keypoints=[ref.keypoints[0]]*601
    elif fault=='wrong_image': cur.gray=cur.gray.astype(float)
    elif fault=='nonfinite_corner': ref.keypoints=[SimpleNamespace(pt=(float('nan'),1.))]
    else:
        calls=[]
        def flow(a,b,p,*args,**kwargs):
            calls.append(True); q=p.copy()
            if fault=='nonfinite_flow': q[:]=np.nan
            elif len(calls)==2: q[:,:,0]+=1.
            return q,np.ones((len(p),1),np.uint8),np.zeros((len(p),1))
        monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',flow)
    if fault in ('nonfinite_flow','reverse_mismatch'):
        assert len(tracked_points(ref,cur)[0][0])==0
    else:
        with pytest.raises(SensorContractError): tracked_points(ref,cur)

"""Unfrozen collection integration: unchanged public sensing, fail closed on optics."""
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.causal_depth_observation_development import INTRINSICS,from_native_depth
from lewm.longer_motion_collection_development import pack,specification
from lewm.physical_semantics import world_from_optical
from lewm.tests.test_independent_pulse_context_development import policy
from scripts.near_field_rgbd_capture_development import NearFieldCapture


def fake_session(monkeypatch,output,*,fault=None):
    import scripts.near_field_rgbd_capture_development as capture
    p=pack(specification('fit'));p=replace(p,camera=replace(p.camera,near_m=.005))
    runner=SimpleNamespace(_sim_time_ns=1_500_000_000,_extract_rgb=lambda output:output[0])
    camera=SimpleNamespace(_raytracer=None,_batch_renderer=None,res=(640,480),intrinsics=np.array(INTRINSICS),
        near=.05 if fault=='near' else .005,far=200.,fov=60.,transform=np.eye(4))
    def set_pose(*,pos,lookat,up):camera.transform=world_from_optical(pos,np.asarray(lookat)-pos,up)@np.diag([1.,-1.,-1.,1.])
    camera.set_pose=set_pose
    def render(*,rgb,depth,segmentation,normal):
        if depth and fault=='physics':runner._sim_time_ns+=2_000_000
        if depth and fault=='pose':camera.transform[0,3]+=.1
        return [np.full((480,640,3),89,np.uint8) if rgb else None,
            np.full((480,640),.02,np.float32) if depth else None,None,None]
    camera.render=render
    if fault=='intrinsics':camera.intrinsics[0,0]+=1.
    robot=SimpleNamespace(get_pos=lambda:np.array([[0.,0.,1.]]),get_quat=lambda:np.array([[1.,0.,0.,0.]]))
    build=SimpleNamespace(robot=robot,camera=camera)
    session=SimpleNamespace(ctx=SimpleNamespace(pack=p,build=build,runner=runner),samples=[dict(timestamp_s=1.5)],depth_manifest=[],depth_audit=[])
    monkeypatch.setattr(capture,'appearance_environment_identity',lambda s:dict(test_identity=True))
    monkeypatch.setattr(capture,'sampling_readback',lambda c:dict(draw_framebuffer_is_single_sample_target=True,
        draw_framebuffer_is_multisample_target=False,samples=0,sample_buffers=0,multisample_enabled=False,pixel_scale=1))
    return session


def test_corrected_capture_records_native_near_depth_but_public_pixels_remain_unknown(monkeypatch,tmp_path):
    session=fake_session(monkeypatch,tmp_path)
    row=NearFieldCapture.capture_fixed_rgb(session,tmp_path,'rgb_0000')
    assert session.depth_audit[0]['native_near_m']==.005
    assert row['rigid_mount_no_obstacle_adjustment']
    np.testing.assert_allclose(np.array(row['world_from_optical'])[:3,3],[.326,0.,1.043],rtol=0,atol=1e-14)
    assert (tmp_path/'rgb_0000.png').is_file() and (tmp_path/'native_depth_0000.npz').is_file()
    p=policy(0);d=from_native_depth(session.latest_native_depth,p,measured_ns=1_500_000_000,
        available_ns=1_500_000_000,now_ns=1_500_000_000)
    assert not d['valid'].any() and not d['depth_m'].any()
    assert session.ctx.runner._sim_time_ns==1_500_000_000


@pytest.mark.parametrize('fault',['near','intrinsics','physics','pose'])
def test_wrong_calibration_or_changed_capture_epoch_rejected(monkeypatch,tmp_path,fault):
    session=fake_session(monkeypatch,tmp_path,fault=fault)
    with pytest.raises(ValueError):NearFieldCapture.capture_fixed_rgb(session,tmp_path,'rgb_0000')
    assert not (tmp_path/'rgb_0000.png').exists() and not session.depth_audit


def test_terminal_audit_keeps_visibility_failure_and_contact_without_materializing_bad_images(monkeypatch,tmp_path):
    import scripts.audit_go2_independent_layout_collection_v1 as audit
    from lewm.independent_layout_inventory_development import build_inventory
    from lewm.independent_layout_collection_development import CollectionInventory
    inv=CollectionInventory(build_inventory());trial=inv.episode_ids('l00')[0]
    report=dict(physical_visibility_pass=False,recorded_sensor_reconstruction_pass=True,departure_present=True,
        setup_admitted=True,schedule_complete=True,target_motion_valid=5,target_contact_positive=1)
    launch=dict(source_sha256={audit.PROTOCOL:'0'*64},role='train')
    terminal=dict(status='TERMINAL_LAYOUT_COLLECTION_FAILURE',uncommitted_trial=None)
    commit={trial:dict(absent_expected_artifacts=[],result={})}
    monkeypatch.setattr(audit,'output_root',lambda b:tmp_path);monkeypatch.setattr(audit,'load_inventory',lambda:inv)
    monkeypatch.setattr(audit,'load_terminal_batch',lambda *args:(launch,terminal,commit,{}))
    monkeypatch.setattr(audit,'audit_condition',lambda *args:(report,{},dict(condition=trial),{}))
    monkeypatch.setattr(audit,'prefix_comparisons',lambda *args:{})
    monkeypatch.setattr(audit,'verify',lambda *args:None);monkeypatch.setattr(audit,'verify_artifacts',lambda *args:None)
    def no_dataset(*args):raise AssertionError('visibility-invalid images must not enter dataset')
    monkeypatch.setattr(audit,'PulseTimedDataset',no_dataset);monkeypatch.setattr(audit,'IntentReturnRGBDReplay',no_dataset)
    audit.run_audit('l00');result=json.loads((tmp_path/'layout_audit.json').read_text())
    assert result['visibility_failed_trials']==[trial] and result['contact_positive_targets']==1
    assert result['departures']==0 and result['materialized_samples']==[]
    assert result['conditions'][trial]['departure_present'] is True
    assert result['conditions'][trial]['status']=='RAW_SENSOR_VISIBILITY_FAILURE'

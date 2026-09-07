"""Quaternion-eigen pose reconstruction, raw lifting, chain and native-score audit."""
import json

import cv2
import numpy as np

from lewm.causal_depth_observation_development import FOCAL
from lewm.finite_rgbd_error_members_development import perturb_packets
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.joint_rgbd_rigid_pose_development import FeatureFrame,matched_points,register
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_keyframe_rgbd_pose_development_v1 import independent_lift
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_joint_rgbd_rigid_pose_development_v1 import OUTPUT,INPUT,MEMBERS,MODES
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json


def quaternion_rotation(wxyz):
    q=np.asarray(wxyz,float);q=q/np.linalg.norm(q);w=q[0];v=q[1:];x,y,z=v
    skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return (w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*skew


def horn_fit(a,b):
    """Largest eigenvector of the symmetric quaternion objective, not SVD."""
    A=a-a.mean(0);B=b-b.mean(0);H=B.T@A;xx,xy,xz=H[0];yx,yy,yz=H[1];zx,zy,zz=H[2]
    N=np.array([[xx+yy+zz,yz-zy,zx-xz,xy-yx],
        [yz-zy,xx-yy-zz,xy+yx,zx+xz],[zx-xz,xy+yx,-xx+yy-zz,yz+zy],
        [xy-yx,zx+xz,yz+zy,-xx-yy+zz]])
    _,vectors=np.linalg.eigh(N);R=quaternion_rotation(vectors[:,-1]);t=a.mean(0)-R@b.mean(0)
    return R,t


def angular_distance(A,B):
    return float(2*np.arcsin(np.clip(np.linalg.norm(A-B,ord=2)/2,0.,1.)))


def project_body(points):
    z=points[:,0]-.326
    return np.column_stack((
        -points[:,1]/z*FOCAL+319.5,-(points[:,2]-.043)/z*FOCAL+239.5))


def self_check():
    rng=np.random.default_rng(2026090632);largest=0.
    for planar in (False,True):
        for _ in range(20):
            b=rng.normal(size=(30,3))
            if planar:b[:,2]=2.
            R=quaternion_rotation(rng.normal(size=4));t=rng.normal(size=3);a=b@R.T+t
            Q,p=horn_fit(a,b);error=max(float(np.max(abs(Q-R))),float(np.max(abs(p-t))))
            assert error<1e-12;largest=max(largest,error)
    return dict(random_planar_and_volumetric_cases=40,maximum_transform_entry_error=largest)


def main():
    cv2.setNumThreads(1);target=OUTPUT/'rigid_pose_reference_audit.json'
    if target.exists():raise ValueError('exclusive reference audit only')
    launch=read_json(OUTPUT,'launch.json');result=read_json(OUTPUT,'result.json');verify(launch)
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    own='scripts/audit_go2_joint_rgbd_rigid_pose_development_v1.py';sources={own:digest(ROOT/own)}
    verify_bindings(inputs|sources);synthetic=self_check()
    predictions=read_json(OUTPUT,'predictions.json');evaluation=read_json(OUTPUT,'evaluation.json')
    cameras=read_json(INPUT/'fit','camera_audit.json')
    with np.load(INPUT/'fit'/'physics_trace.npz',allow_pickle=False) as raw:poses=raw['base_pose_world'].copy()
    native_R=[quaternion_rotation(p[[6,3,4,5]]) for p in poses];R0=native_R[749];p0=poses[749,:3];reports={}
    for mode in MODES:
        for member in MEMBERS:
            name=mode+'__'+member.name;gyro=FastRelativeOrientation();anchor=None;anchor_R=np.eye(3);anchor_G=np.eye(3)
            anchor_p=np.zeros(3);anchor_frame=0;anchor_ns=1_500_000_000;nodes=[];failure=None
            accepted=points=promotions=0;maximum_fit_difference=0.;maximum_score_difference=0.;terminal=None
            scores={r['frame']:r for r in evaluation['details'][name]}
            for frame,row in enumerate(predictions['rows']):
                item=row['members'][name]
                if failure is not None:
                    assert item['status']=='NOT_REINVOKED_AFTER_FAILURE' and item['state'] is None and item['failure']==failure
                    continue
                p,d=load_rgbd_observation(INPUT/'fit',frame);f=load_fast_packet(INPUT/'fit',frame)
                p,d,f=perturb_packets(member,p,d,f,anchor_ns=1_500_000_000);now=p['sensor_state']['decision_ns']
                assert row['frame']==frame and row['measured_ns']==now
                if item['status']=='TERMINAL_FAILURE':
                    failure=item['failure'];assert item['state'] is None
                    z=d['depth_m'];bad=d['valid']&((z<.2)|(z>5.))
                    if bad.any():
                        assert 'metric depth and explicit unknown-ray mask required' in failure['chain']
                        terminal=dict(kind='INVALID_DEPTH_PACKET',invalid_pixels=int(bad.sum()));continue
                attitude=gyro.begin(p,f,now_ns=now) if frame==0 else gyro.step(p,f,now_ns=now)
                G=np.asarray(attitude['rotation_initial_body_from_current_body'])
                if failure is not None:
                    a,b,ua,ub=matched_points(anchor,FeatureFrame(p['image']['rgb'],d))
                    try:register(a,b,ua,ub,gyro_rotation=anchor_G.T@G,mode=mode,frame=frame)
                    except Exception as error:
                        assert str(error) in failure['chain'];terminal=dict(kind='REGISTRATION_REJECTION',reason=str(error))
                    else:raise AssertionError('terminal outer-motion failure needs dedicated audit')
                    continue
                assert item['status']=='CONDITIONAL_RIGID_POSE';s=item['state'];accepted+=1
                assert s['position_error_bound'] is None and s['orientation_error_bound'] is None
                assert not s['global_history_reset'] and s['reference_frame']==anchor_frame
                np.testing.assert_allclose(s['gyro_rotation_initial_body_from_current_body'],G,atol=1e-12,rtol=0)
                R=np.asarray(s['rotation_initial_body_from_current_body']);position=np.asarray(s['position_initial_body_m'])
                if frame==0:
                    anchor=FeatureFrame(p['image']['rgb'],d)
                    nodes.append(dict(frame=0,measured_ns=now,parent_frame=None,position_initial_body_m=[0.,0.,0.],
                        rotation_initial_body_from_current_body=np.eye(3).tolist(),pose_error_bound=None))
                else:
                    reg=s['registration'];a,b,ua,ub=[np.asarray(reg[k]) for k in ('reference_inlier_points_body_m',
                        'current_inlier_points_body_m','reference_inlier_pixels','current_inlier_pixels')]
                    assert len(a)==reg['inliers'] and len(a)>=12 and len(a)/reg['lifted_matches']==reg['inlier_fraction']>=.6
                    np.testing.assert_allclose(a,independent_lift(anchor.depth,ua),atol=2e-12,rtol=0)
                    np.testing.assert_allclose(b,independent_lift(d,ub),atol=2e-12,rtol=0)
                    Q,t=horn_fit(a,b) if mode=='joint' else (anchor_G.T@G,a.mean(0)-(anchor_G.T@G)@b.mean(0))
                    delta=max(float(np.max(abs(Q-np.asarray(reg['relative_rotation'])))),float(np.max(abs(t-reg['translation_reference_body_m']))))
                    maximum_fit_difference=max(maximum_fit_difference,delta);assert delta<1e-9
                    np.testing.assert_allclose(R,anchor_R@Q,atol=1e-9,rtol=0)
                    np.testing.assert_allclose(position,anchor_p+anchor_R@t,atol=1e-9,rtol=0)
                    disagreement=angular_distance(anchor_G.T@G,Q)
                    assert disagreement<=.1 and abs(disagreement-reg['gyro_disagreement_rad'])<1e-9
                    before=predictions['rows'][frame-1]['members'][name]['state']
                    assert np.linalg.norm(position-np.asarray(before['position_initial_body_m']))<=.15+1e-10
                    assert angular_distance(np.asarray(before['rotation_initial_body_from_current_body']),R)<=.20+1e-10
                    assert np.all(np.linalg.norm(a-b@Q.T-t,axis=1)<=.02+1e-10)
                    assert np.all(np.linalg.norm(project_body((a-t)@Q)-ub,axis=1)<=1.+1e-7)
                    assert np.all(np.linalg.norm(project_body(b@Q.T+t)-ua,axis=1)<=1.+1e-7)
                    for cloud,key in ((a,'reference_scatter_rms_m'),(b,'current_scatter_rms_m')):
                        centered=cloud-cloud.mean(0);rms=np.sqrt(np.maximum(np.linalg.eigvalsh(centered.T@centered/len(cloud)),0))[::-1]
                        np.testing.assert_allclose(rms,reg[key],atol=1e-7,rtol=0)
                        assert rms[1]>=.02 and rms[1]>=.05*rms[0]
                    local_angle=angular_distance(np.eye(3),Q);motion=np.linalg.norm(t)>=.4 or local_angle>=.35
                    near=min(reg['reference_grid_cells'],reg['current_grid_cells'])<=7
                    assert min(reg['reference_grid_cells'],reg['current_grid_cells'])>=6
                    assert s['promoted_keyframe']==(motion or near)
                    if s['promoted_keyframe']:
                        nodes.append(dict(frame=frame,measured_ns=now,parent_frame=anchor_frame,position_initial_body_m=s['position_initial_body_m'],
                            rotation_initial_body_from_current_body=s['rotation_initial_body_from_current_body'],pose_error_bound=None))
                        anchor=FeatureFrame(p['image']['rgb'],d);anchor_R=R.copy();anchor_G=G.copy();anchor_p=position.copy()
                        anchor_frame=frame;anchor_ns=now;promotions+=1
                    points+=len(a)
                sample=cameras[frame]['physical_sample_index'];truth=R0.T@(poses[sample,:3]-p0);Rt=R0.T@native_R[sample]
                independent_position=float(np.linalg.norm(position-truth));independent_angle=angular_distance(R,Rt)
                difference=max(abs(independent_position-scores[frame]['position_error_m']),abs(independent_angle-scores[frame]['orientation_error_rad']))
                maximum_score_difference=max(maximum_score_difference,difference);assert difference<1e-9
            assert nodes==predictions['keyframes'][name] and failure==result['summaries'][name]['failure']
            assert accepted==result['summaries'][name]['admitted']
            reports[name]=dict(accepted_frames=accepted,accepted_points_checked=points,promotions_checked=promotions,
                maximum_independent_fit_difference=maximum_fit_difference,maximum_independent_native_score_difference=maximum_score_difference,
                terminal=terminal,unknown_error_bounds_preserved=True)
            print('RIGID_REFERENCE_AUDIT_PASS',name,flush=True)
    verify(launch);verify_bindings(inputs|sources)
    report=dict(status='RIGID_POSE_REFERENCE_AUDIT_PASS',models=reports,synthetic_quaternion_eigen_checks=synthetic,
        source_sha256=sources,input_sha256=inputs,independent_native_pose_scorer=True,
        feature_identity_or_uncertainty_proven=False,proposal_optimality_independently_audited=False,navigation_qualified=False)
    write_json(target,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

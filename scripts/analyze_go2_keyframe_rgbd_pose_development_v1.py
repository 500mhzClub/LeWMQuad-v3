"""Accepted-pair arithmetic and terminal-gate audit; no model resumed or fitted."""
from itertools import product
import json

import cv2
import numpy as np

from lewm.causal_depth_observation_development import FOCAL
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.finite_rgbd_error_members_development import perturb_packets
from lewm.keyframe_rgbd_pose_development import FeatureFrame,matched_points,KeyframeHypotheses
from lewm.rgbd_correspondence_motion_development import project,cells,RULES
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_keyframe_rgbd_pose_development_v1 import OUTPUT,INPUT,MEMBERS
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json


def corner_radius(points,pixels):
    """Independent long-double eight-corner lifting enclosure."""
    a=np.asarray(points,np.longdouble);uv=np.asarray(pixels,np.longdouble);z=a[:,0]-np.longdouble(.326)
    h=KeyframeHypotheses();largest=np.zeros(len(a),np.longdouble)
    for su,sv,sz in product((-1,1),repeat=3):
        q=uv+[su*h.pixel_coordinate_error,sv*h.pixel_coordinate_error];zz=z+sz*h.lifted_depth_error_m
        p=np.column_stack((zz+np.longdouble(.326),-zz*(q[:,0]+.5-320)/FOCAL,-zz*(q[:,1]+.5-240)/FOCAL+np.longdouble(.043)))
        largest=np.maximum(largest,np.sqrt(np.sum((p-a)**2,axis=1)))
    return largest


def independent_lift(depth,uv):
    uv=np.asarray(uv,float);ij=np.floor(uv).astype(int);w=uv-ij;x,y=ij.T
    z=(depth['depth_m'][y,x]*(1-w[:,0])*(1-w[:,1])+depth['depth_m'][y,x+1]*w[:,0]*(1-w[:,1])
       +depth['depth_m'][y+1,x]*(1-w[:,0])*w[:,1]+depth['depth_m'][y+1,x+1]*w[:,0]*w[:,1])
    return np.column_stack((z+.326,-z*(uv[:,0]+.5-320)/FOCAL,-z*(uv[:,1]+.5-240)/FOCAL+.043))


def terminal_gate(reference,current,R):
    a,b,ua,ub=matched_points(reference,current)
    if len(a)<12:return dict(lifted_matches=len(a),failed_requirements=['minimum_matches'])
    delta=a-b@R.T;t=np.median(delta,axis=0);mask=np.linalg.norm(delta-t,axis=1)<=RULES['residual_m']
    history=[];stable=False
    for _ in range(10):
        if mask.sum()<12:break
        t=delta[mask].mean(0);fp,fg=project((a-t)@R);rp,rg=project(b@R.T+t)
        use=(np.linalg.norm(delta-t,axis=1)<=RULES['residual_m'])&fg&rg
        use&=np.linalg.norm(fp-ub,axis=1)<=RULES['reprojection_pixels']
        use&=np.linalg.norm(rp-ua,axis=1)<=RULES['reprojection_pixels']
        stable=bool(np.array_equal(mask,use));history.append([int(mask.sum()),int(use.sum()),stable])
        if stable:break
        mask=use
    gates=dict(stable_inliers=stable,minimum_matches=int(mask.sum())>=12,
        inlier_fraction=float(mask.mean())>=RULES['minimum_inlier_fraction'],
        reference_grid_cells=cells(ua[mask])>=RULES['minimum_grid_cells'],
        current_grid_cells=cells(ub[mask])>=RULES['minimum_grid_cells'],reference_displacement=np.linalg.norm(t)<=3.)
    return dict(lifted_matches=len(a),inliers=int(mask.sum()),fraction=float(mask.mean()),
        reference_grid_cells=cells(ua[mask]),current_grid_cells=cells(ub[mask]),
        translation_reference_body_m=t.tolist(),iteration_counts=history,
        failed_requirements=[k for k,v in gates.items() if not v])


def main():
    cv2.setNumThreads(1);target=OUTPUT/'accepted_pair_and_terminal_analysis.json'
    if target.exists():raise ValueError('exclusive analysis output required')
    launch=read_json(OUTPUT,'launch.json');result=read_json(OUTPUT,'result.json');verify(launch)
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources={'scripts/analyze_go2_keyframe_rgbd_pose_development_v1.py':digest(ROOT/'scripts/analyze_go2_keyframe_rgbd_pose_development_v1.py')}
    verify_bindings(inputs|sources)
    predictions=read_json(OUTPUT,'predictions.json');evaluation=read_json(OUTPUT,'evaluation.json');out={}
    for member in MEMBERS:
        name=member.name;summary=result['summaries'][name];stop=summary['failure']['frame']
        orientation=FastRelativeOrientation();reference=None;paired=0;radius_error=0.;failure=None
        assert len(predictions['keyframes'][name])==1  # Actual V1 outcome, not an arbitrary chain audit.
        for frame,row in enumerate(predictions['rows']):
            item=row['members'][name]
            if frame>stop:
                assert item['status']=='NOT_REINVOKED_AFTER_FAILURE' and item['state'] is None and item['failure']==failure
                continue
            p,d=load_rgbd_observation(INPUT/'fit',frame);f=load_fast_packet(INPUT/'fit',frame)
            p,d,f=perturb_packets(member,p,d,f,anchor_ns=1_500_000_000);now=p['sensor_state']['decision_ns']
            attitude=orientation.begin(p,f,now_ns=now) if frame==0 else orientation.step(p,f,now_ns=now)
            if frame==0:reference=FeatureFrame(p['image']['rgb'],d)
            R=np.asarray(attitude['rotation_initial_body_from_current_body'])
            if frame==stop:
                assert item['status']=='TERMINAL_FAILURE' and item['state'] is None
                failure=item['failure'];gates=terminal_gate(reference,FeatureFrame(p['image']['rgb'],d),R)
                assert gates['failed_requirements'];continue
            assert item['status']=='CONDITIONAL_KEYFRAME_POSE'
            s=item['state'];np.testing.assert_allclose(s['rotation_initial_body_from_current_body'],R,atol=1e-12,rtol=0)
            assert not s['promoted_keyframe'] and s['reference_frame']==0 and s['keyframe_count']==1
            if frame==0:continue
            reg=s['registration'];a,b,ua,ub=[np.asarray(reg[k]) for k in ('reference_inlier_points_body_m',
                'current_inlier_points_body_m','reference_inlier_pixels','current_inlier_pixels')]
            np.testing.assert_allclose(a,independent_lift(reference.depth,ua),atol=2e-12,rtol=0)
            np.testing.assert_allclose(b,independent_lift(d,ub),atol=2e-12,rtol=0)
            t=(a-b@R.T).mean(0);np.testing.assert_allclose(t,s['position_initial_body_m'],atol=2e-12,rtol=0)
            assert len(a)==reg['inliers'] and min(cells(ua),cells(ub))>=6
            local_angle=(now-1_500_000_000)*1e-9*.001
            radius=float(np.mean(corner_radius(a,ua)+corner_radius(b,ub)
                +2*np.sin(np.longdouble(local_angle)/2)*np.linalg.norm(b.astype(np.longdouble),axis=1)))
            radius_error=max(radius_error,abs(radius-s['conditional_global_position_radius_m']))
            assert abs(radius-s['conditional_global_position_radius_m'])<1e-11
            paired+=len(a)
        records=evaluation['details'][name]
        assert len(records)==stop and paired==summary['inlier_static_point_checks']
        assert sum(r.get('inlier_static_point_hypothesis_violations',0) for r in records)==summary['inlier_static_point_violations']
        out[name]=dict(accepted_frames=stop,raw_depth_lifted_pairs_checked=paired,
            maximum_corner_radius_arithmetic_difference_m=radius_error,terminal_gate=gates,
            point_hypothesis_violations=summary['inlier_static_point_violations'],
            maximum_angle_radius_excess_rad=max(r['orientation_error_rad']-r['conditional_angle_radius_rad'] for r in records))
    verify(launch);verify_bindings(inputs|sources)
    report=dict(status='ACCEPTED_PAIR_ARITHMETIC_AND_TERMINAL_GATE_AUDIT_PASS',members=out,
        source_sha256=sources,input_sha256=inputs,all_runtime_keyframe_counts_one=True,
        actual_keyframe_promotion_validated=False,feature_id_correctness_proven=False,
        independent_native_scorer=False,navigation_qualified=False)
    write_json(target,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

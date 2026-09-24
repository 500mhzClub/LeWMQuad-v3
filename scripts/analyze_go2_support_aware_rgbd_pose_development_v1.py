"""Audit accepted reference chains and robust enclosures without resetting models."""
from itertools import combinations
import json

import cv2
import numpy as np

from lewm.finite_rgbd_error_members_development import perturb_packets
from lewm.keyframe_rgbd_pose_development import FeatureFrame
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.support_aware_rgbd_pose_development import contaminated_box
from scripts.analyze_go2_keyframe_rgbd_pose_development_v1 import corner_radius,independent_lift,terminal_gate
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_support_aware_rgbd_pose_development_v1 import OUTPUT,INPUT,MEMBERS
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json


def exhaustive_subset_check():
    rng=np.random.default_rng(2026090630);checked=0
    for n in (5,7,9):
        for f in range((n-1)//2+1):
            for _ in range(20):
                d=rng.normal(size=(n,3));r=rng.uniform(.2,3.,n);box=contaminated_box(d,r,maximum_outliers=f)
                for subset in combinations(range(n),n-f):
                    indices=list(subset);lo=(d[indices]-r[indices,None]).max(0);hi=(d[indices]+r[indices,None]).min(0)
                    if np.all(lo<=hi):
                        assert box['status']=='CONDITIONAL_CONTAMINATED_BOX'
                        assert np.all(lo>=np.asarray(box['lower'])-1e-12) and np.all(hi<=np.asarray(box['upper'])+1e-12)
                    checked+=1
    return checked


def main():
    cv2.setNumThreads(1);target=OUTPUT/'reference_chain_and_box_audit.json'
    if target.exists():raise ValueError('exclusive source-bound audit only')
    launch=read_json(OUTPUT,'launch.json');result=read_json(OUTPUT,'result.json');verify(launch)
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    own='scripts/analyze_go2_support_aware_rgbd_pose_development_v1.py';source={own:digest(ROOT/own)}
    verify_bindings(inputs|source);subset_checks=exhaustive_subset_check()
    predictions=read_json(OUTPUT,'predictions.json');evaluations=read_json(OUTPUT,'evaluation.json');summary={}
    for member in MEMBERS:
        name=member.name;orientation=FastRelativeOrientation();anchor=None;anchor_frame=0;anchor_R=np.eye(3);anchor_p=np.zeros(3)
        anchor_ns=1_500_000_000;old_radius=robust_radius=0.;nodes=[];pairs=promotions=accepted=0;failure=None;terminal=None
        max_box_difference=0.
        for frame,row in enumerate(predictions['rows']):
            item=row['members'][name];assert row['frame']==frame
            if failure is not None:
                assert item['status']=='NOT_REINVOKED_AFTER_FAILURE' and item['state'] is None and item['failure']==failure
                continue
            p,d=load_rgbd_observation(INPUT/'fit',frame);f=load_fast_packet(INPUT/'fit',frame)
            p,d,f=perturb_packets(member,p,d,f,anchor_ns=1_500_000_000);now=p['sensor_state']['decision_ns']
            if item['status']=='TERMINAL_FAILURE':
                z=d['depth_m'];valid=d['valid']
                bad=(valid&((z<.2)|(z>5.)))|(~valid&(z!=0.))
                if bad.any():
                    assert item['state'] is None;failure=item['failure']
                    assert 'metric depth and explicit unknown-ray mask required' in failure['chain']
                    terminal=dict(failed_requirements=['depth_packet_range_or_unknown_mask'],
                        invalid_pixels=int(bad.sum()),minimum_invalid_depth_m=float(z[bad].min()),
                        maximum_invalid_depth_m=float(z[bad].max()),registration_attempted=False)
                    continue
            attitude=orientation.begin(p,f,now_ns=now) if frame==0 else orientation.step(p,f,now_ns=now)
            R=np.asarray(attitude['rotation_initial_body_from_current_body']);assert row['measured_ns']==now
            if item['status']=='TERMINAL_FAILURE':
                assert item['state'] is None;failure=item['failure']
                terminal=terminal_gate(anchor,FeatureFrame(p['image']['rgb'],d),anchor_R.T@R)
                assert terminal['failed_requirements'];continue
            assert item['status']=='CONDITIONAL_KEYFRAME_POSE';s=item['state'];accepted+=1
            np.testing.assert_allclose(s['rotation_initial_body_from_current_body'],R,atol=1e-12,rtol=0)
            assert s['reference_frame']==anchor_frame and not s['global_history_reset']
            if frame==0:
                anchor=FeatureFrame(p['image']['rgb'],d)
                nodes.append(dict(frame=0,measured_ns=now,parent_frame=None,position_initial_body_m=[0.,0.,0.],conditional_global_position_radius_m=0.))
                continue
            reg=s['registration'];ev=s['robust_evidence'];relative=anchor_R.T@R
            a,b,ua,ub=[np.asarray(reg[k]) for k in ('reference_inlier_points_body_m','current_inlier_points_body_m',
                'reference_inlier_pixels','current_inlier_pixels')]
            np.testing.assert_allclose(a,independent_lift(anchor.depth,ua),atol=2e-12,rtol=0)
            np.testing.assert_allclose(b,independent_lift(d,ub),atol=2e-12,rtol=0)
            delta=a-b@relative.T;t=delta.mean(0);pglobal=anchor_p+anchor_R@t
            np.testing.assert_allclose(pglobal,s['position_initial_body_m'],atol=2e-12,rtol=0)
            local_angle=(now-anchor_ns)*1e-9*.001;anchor_angle=(anchor_ns-1_500_000_000)*1e-9*.001
            e=corner_radius(a,ua)+corner_radius(b,ub)+2*np.sin(np.longdouble(local_angle)/2)*np.linalg.norm(b.astype(np.longdouble),axis=1)
            reference_rotation_term=2*np.sin(anchor_angle/2)*np.linalg.norm(t)
            expected_legacy=float(old_radius+e.mean()+reference_rotation_term)
            np.testing.assert_allclose(expected_legacy,s['conditional_global_position_radius_m'],atol=1e-10,rtol=0)
            allowed=int(np.floor(.2*len(a)));assert ev['maximum_outliers']==allowed
            # Partition in long double, independent of the production sorted-float path.
            lows=delta.astype(np.longdouble)-e[:,None]-1e-12;highs=delta.astype(np.longdouble)+e[:,None]+1e-12
            lower=np.partition(lows,len(a)-allowed-1,axis=0)[len(a)-allowed-1]
            upper=np.partition(highs,allowed,axis=0)[allowed]
            error=max(float(np.max(np.abs(lower-ev['lower']))),float(np.max(np.abs(upper-ev['upper']))))
            max_box_difference=max(max_box_difference,error);assert error<1e-10
            available=bool(np.all(lower<=upper));assert available==(ev['status']=='CONDITIONAL_CONTAMINATED_BOX')
            local=global_radius=None
            if available:
                local=float(np.linalg.norm(np.maximum(np.abs(lower-t),np.abs(upper-t))))
                np.testing.assert_allclose(local,ev['local_radius_m'],atol=1e-10,rtol=0)
                if robust_radius is not None:global_radius=robust_radius+local+reference_rotation_term
            if global_radius is None:assert ev['global_radius_m'] is None
            else:np.testing.assert_allclose(global_radius,ev['global_radius_m'],atol=1e-9,rtol=0)
            angle=float(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1)))
            motion=bool(np.linalg.norm(t)>=.4 or angle>=.35)
            near=min(reg['reference_grid_cells'],reg['current_grid_cells'])<=7
            assert min(reg['reference_grid_cells'],reg['current_grid_cells'])>=6
            assert s['promoted_keyframe']==(motion or near)
            assert s['promotion_reason']==('motion_threshold' if motion else 'accepted_support_margin' if near else None)
            if s['promoted_keyframe']:
                nodes.append(dict(frame=frame,measured_ns=now,parent_frame=anchor_frame,
                    position_initial_body_m=s['position_initial_body_m'],conditional_global_position_radius_m=s['conditional_global_position_radius_m']))
                anchor=FeatureFrame(p['image']['rgb'],d);anchor_frame=frame;anchor_R=R.copy();anchor_p=pglobal.copy();anchor_ns=now
                old_radius=s['conditional_global_position_radius_m'];robust_radius=ev['global_radius_m'];promotions+=1
            assert s['keyframe_count']==len(nodes);pairs+=len(a)
        assert nodes==predictions['keyframes'][name]
        assert accepted==result['summaries'][name]['admitted'] and failure==result['summaries'][name]['failure']
        rows=evaluations['details'][name];assert len(rows)==accepted
        summary[name]=dict(accepted_frames=accepted,accepted_pairs_checked=pairs,promotions_checked=promotions,
            keyframe_parent_chain_exact=True,global_history_and_unknown_radii_preserved=True,
            maximum_independent_box_difference_m=max_box_difference,terminal_registration_gates=terminal)
    verify(launch);verify_bindings(inputs|source)
    report=dict(status='REFERENCE_CHAIN_AND_CONTAMINATED_BOX_AUDIT_PASS',members=summary,
        exhaustive_small_population_subset_checks=subset_checks,source_sha256=source,input_sha256=inputs,
        independent_native_scorer=False,correspondence_identity_proven=False,conditional_hypotheses_validated=False,navigation_qualified=False)
    write_json(target,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()

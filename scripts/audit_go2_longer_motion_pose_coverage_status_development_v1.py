"""Raw lifting, independent pose fitting, reference chains and coverage-cell audit."""
import json

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.analyze_go2_keyframe_rgbd_pose_development_v1 import independent_lift
from scripts.audit_go2_joint_rgbd_rigid_pose_development_v1 import horn_fit
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_bounded_depth_surface_development_v1 import HYPOTHESES
from scripts.probe_go2_longer_motion_pose_coverage_status_development_v1 import OUTPUT,INPUT,MODES,initial_up
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def defects(R):
    return max(float(np.max(abs(R.T@R-np.eye(3)))),abs(float(np.linalg.det(R))-1.))


def main():
    cv2.setNumThreads(1); target=OUTPUT/'pose_chain_coverage_audit_launch.json'
    if target.exists(): raise ValueError('exclusive audit only; no silent retry')
    launch=read_json(OUTPUT,'launch.json'); result=read_json(OUTPUT,'result.json'); verify(launch)
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources=discover_sources(('scripts/audit_go2_longer_motion_pose_coverage_status_development_v1.py',),launch['source_sha256'])
    verify_bindings(inputs|sources); write_json(target,dict(source_sha256=sources,input_sha256=inputs))
    reports={}
    try:
        predictions=read_json(OUTPUT,'predictions.json'); geometry=ArticulatedCollisionGeometry(URDF)
        for trial,record in predictions['trials'].items():
            gyro=FastRelativeOrientation(); anchors={}; failures={}; nodes={m:[] for m in MODES}
            stats={m:dict(accepted=0,raw_pairs_checked=0,promotions_checked=0,reference_coverage_queries=0,
                coverage_contract_rejections=0,maximum_direct_gyro_defect=0.,maximum_composed_pose_defect=0.,
                maximum_composed_vs_direct_gyro_difference=0.,maximum_transpose_vs_inverse_composition_difference=0.,
                maximum_independent_fit_difference=0.) for m in MODES}
            surface=None
            for frame,row in enumerate(record['rows']):
                p,d=load_rgbd_observation(INPUT/trial,frame); f=load_fast_packet(INPUT/trial,frame)
                now=p['sensor_state']['decision_ns']; assert row['measured_ns']==now and row['frame']==frame
                g=gyro.begin(p,f,now_ns=now) if frame==0 else gyro.step(p,f,now_ns=now)
                G=np.asarray(g['rotation_initial_body_from_current_body']); joints=p['sensor_state']['sensed']['joints']['values'][-1,:12]
                if frame==0:
                    surface=BoundedDepthSurface(d['depth_m'],d['valid'],initial_up(p),**HYPOTHESES)
                    errors=dict.fromkeys([s['shape_id'] for s in geometry.supports(joints,np.eye(3))['shapes']],0.)
                for mode in MODES:
                    item=row['members'][mode]; s=item['state']; evidence=item['coverage_evidence']; st=stats[mode]
                    if mode in failures:
                        assert item['status']=='NOT_REINVOKED_AFTER_FAILURE' and item['failure']==failures[mode] and s is None
                    elif item['status']=='TERMINAL_FAILURE':
                        assert s is None; failures[mode]=item['failure']
                    if s is None:
                        assert evidence['status']=='POSE_UNAVAILABLE' and item['floor_coverage'] is None; continue
                    assert item['status']=='CONDITIONAL_RIGID_POSE'; st['accepted']+=1
                    assert s['position_error_bound'] is None and s['orientation_error_bound'] is None and not s['global_history_reset']
                    R=np.asarray(s['rotation_initial_body_from_current_body']); position=np.asarray(s['position_initial_body_m'])
                    np.testing.assert_array_equal(G,s['gyro_rotation_initial_body_from_current_body'])
                    st['maximum_direct_gyro_defect']=max(st['maximum_direct_gyro_defect'],defects(G))
                    st['maximum_composed_pose_defect']=max(st['maximum_composed_pose_defect'],defects(R))
                    if frame==0:
                        anchors[mode]=(frame,d,R,position,G)
                        nodes[mode].append(dict(frame=0,measured_ns=now,parent_frame=None,position_initial_body_m=[0.,0.,0.],
                            rotation_initial_body_from_current_body=np.eye(3).tolist(),pose_error_bound=None))
                    else:
                        af,ad,ar,ap,ag=anchors[mode]; reg=s['registration']; assert s['reference_frame']==af
                        a,b,ua,ub=[np.asarray(reg[k]) for k in ('reference_inlier_points_body_m','current_inlier_points_body_m',
                            'reference_inlier_pixels','current_inlier_pixels')]
                        np.testing.assert_allclose(a,independent_lift(ad,ua),rtol=0,atol=2e-12)
                        np.testing.assert_allclose(b,independent_lift(d,ub),rtol=0,atol=2e-12)
                        Q,t=horn_fit(a,b) if mode=='joint' else (ag.T@G,a.mean(0)-(ag.T@G)@b.mean(0))
                        delta=max(float(np.max(abs(Q-reg['relative_rotation']))),float(np.max(abs(t-reg['translation_reference_body_m']))))
                        assert delta<1e-9; st['maximum_independent_fit_difference']=max(st['maximum_independent_fit_difference'],delta)
                        np.testing.assert_allclose(R,ar@Q,rtol=0,atol=1e-9)
                        np.testing.assert_allclose(position,ap+ar@t,rtol=0,atol=1e-9); st['raw_pairs_checked']+=len(a)
                        if mode=='gyro':
                            st['maximum_composed_vs_direct_gyro_difference']=max(st['maximum_composed_vs_direct_gyro_difference'],float(np.max(abs(R-G))))
                            inverse_composition=ar@np.linalg.solve(ag,G)
                            st['maximum_transpose_vs_inverse_composition_difference']=max(st['maximum_transpose_vs_inverse_composition_difference'],
                                float(np.max(abs(R-inverse_composition))))
                        if s['promoted_keyframe']:
                            nodes[mode].append(dict(frame=frame,measured_ns=now,parent_frame=af,
                                position_initial_body_m=s['position_initial_body_m'],
                                rotation_initial_body_from_current_body=s['rotation_initial_body_from_current_body'],pose_error_bound=None))
                            anchors[mode]=(frame,d,R,position,G); st['promotions_checked']+=1
                    if surface.status!='BOUNDED_MEASURED_SURFACE_AVAILABLE':
                        assert evidence['status']=='SURFACE_UNAVAILABLE' and item['floor_coverage'] is None; continue
                    assert evidence['floor_coverage']==item['floor_coverage']
                    assert evidence['orthogonality_max_abs']==float(np.max(abs(R.T@R-np.eye(3))))
                    assert evidence['determinant_abs_error']==abs(float(np.linalg.det(R))-1.)
                    try:
                        q=surface.query(geometry,joints,rotation_observation_from_body=R,translation_observation_from_body=position,
                            point_error_by_shape=errors,backend='reference')
                    except SensorContractError as error:
                        assert evidence['status']=='COVERAGE_CONTRACT_REJECTED' and evidence['reason']==str(error)
                        assert item['floor_coverage'] is None; st['coverage_contract_rejections']+=1
                    else:
                        assert evidence['status']=='CONDITIONAL_ZERO_ADDITIONAL_ERROR_COVERAGE'
                        assert q['floor_coverage']==item['floor_coverage']; st['reference_coverage_queries']+=1
            for mode in MODES:
                assert nodes[mode]==record['keyframes'][mode]
                name=trial+'__'+mode; assert failures.get(mode)==result['summaries'][name]['failure']
                assert stats[mode]['accepted']==result['summaries'][name]['admitted']; reports[name]=stats[mode]
            print('POSE_CHAIN_AND_COVERAGE_AUDIT_PASS',trial,json.dumps(stats),flush=True)
        verify(launch); verify_bindings(inputs|sources)
        write_json(OUTPUT/'pose_chain_coverage_audit.json',dict(status='POSE_CHAIN_AND_COVERAGE_AUDIT_PASS',models=reports,
            audit_launch_sha256=digest(target),pose_projection_or_model_rerun=False,
            geometric_surface_construction_shared=True,feature_identity_and_uncertainty_proven=False,navigation_qualified=False))
    except Exception as error:
        write_json(OUTPUT/'pose_chain_coverage_audit_failure.json',dict(status='TERMINAL_POSE_CHAIN_COVERAGE_AUDIT_FAILURE',
            reason=str(error),completed_models=reports)); raise


if __name__=='__main__': main()

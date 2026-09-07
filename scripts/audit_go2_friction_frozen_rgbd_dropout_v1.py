"""Independent accepted-pair pose arithmetic and dropout/native score audit.

No estimator restart, fitting, source modification or physical execution.
Feature identities and physical sensor uncertainty are not proved by this audit.
"""
import json
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.analyze_go2_keyframe_rgbd_pose_development_v1 import independent_lift
from scripts.audit_go2_joint_rgbd_rigid_pose_development_v1 import horn_fit
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_friction_frozen_rgbd_dropout_v1 import INPUT, OUTPUT, CONDITIONS, MODES
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json, read_npz
from scripts.startup_source_inventory_development import discover_sources


def check_stats(actual, values):
    assert actual['count']==len(values)
    if not values:
        assert all(actual[k] is None for k in ('mean','p95','maximum')); return
    ordered=sorted(values); offset=.95*(len(values)-1); lo=math.floor(offset); frac=offset-lo
    p95=ordered[lo]*(1-frac)+ordered[min(lo+1,len(values)-1)]*frac
    expected=[math.fsum(values)/len(values), p95, max(values)]
    np.testing.assert_allclose([actual[k] for k in ('mean','p95','maximum')],expected,rtol=0,atol=1e-10)


def audit_condition(condition, record, evaluated):
    directory=INPUT/condition; raw=read_npz(directory,'physics_trace.npz')
    support=read_json(directory,'support_predictions.json')['rows']
    assert [s['measured_ns'] for s in support]==list(range(1_500_000_000,24_000_000_001,20_000_000))
    missing={s['measured_ns'] for s in support if s['modes']['level_sphere_rolling']['consensus_velocity_body_m_s'] is None}
    assert len(missing)==evaluated['unavailable_support_observations']
    recovered=[]
    for span in evaluated['dropout_spans']:
        times=list(range(span['first_missing_ns'],span['last_missing_ns']+1,20_000_000))
        assert len(times)==span['missing_samples']; recovered.extend(times)
        covering=[j for j in range(1,226) if any(1_500_000_000+(j-1)*100_000_000 <= t <= 1_500_000_000+j*100_000_000 for t in times)]
        assert covering==span['camera_interval_end_frames']
        for mode in MODES:
            values=[evaluated['details'][mode]['windows'][j-1]['displacement_error_m'] for j in covering]
            s=span['models'][mode]; assert s['intervals']==len(values) and s['visual_unavailable']==sum(v is None for v in values)
            check_stats(s['displacement_error_m'],[v for v in values if v is not None])
    assert len(recovered)==len(set(recovered)) and set(recovered)==missing
    poses=raw['base_pose_world']; R0=Rotation.from_quat(poses[749,3:]).as_matrix(); p0=poses[749,:3]
    truth=[R0.T@(poses[749+j*50,:3]-p0) for j in range(226)]
    gyro=FastRelativeOrientation(); anchors={}; nodes={m:[] for m in MODES}; failures={}
    report={m:dict(accepted=0,raw_inlier_pairs=0,maximum_fit_coordinate_difference=0.,maximum_score_difference=0.) for m in MODES}
    for j,row in enumerate(record['rows']):
        p,d=load_rgbd_observation(directory,j); fast=load_fast_packet(directory,j); now=p['sensor_state']['decision_ns']
        assert row['frame']==j and now==row['measured_ns']==1_500_000_000+j*100_000_000
        assert row['depth_valid_pixels']==int(d['valid'].sum())
        attitude=gyro.begin(p,fast,now_ns=now) if j==0 else gyro.step(p,fast,now_ns=now)
        G=np.asarray(attitude['rotation_initial_body_from_current_body']); sample=749+j*50
        for mode in MODES:
            item=row['members'][mode]; state=item['state']; frame=evaluated['details'][mode]['frames'][j]
            assert frame['frame']==j and frame['measured_ns']==now and frame['phase']==int(raw['phase'][sample])
            assert frame['contact_unavailable_at_capture']==(now in missing) and frame['visual_available']==(state is not None)
            if mode in failures:
                assert item['status']=='NOT_REINVOKED_AFTER_FAILURE' and item['failure']==failures[mode] and state is None
            elif item['status']=='TERMINAL_FAILURE':
                assert state is None; failures[mode]=item['failure']
            if state is None:
                assert frame['position_error_m'] is None and frame['orientation_error_rad'] is None
            else:
                assert item['status']=='CONDITIONAL_RIGID_POSE'; report[mode]['accepted']+=1
                assert state['position_error_bound'] is None and state['orientation_error_bound'] is None and not state['global_history_reset']
                R=np.asarray(state['rotation_initial_body_from_current_body']); position=np.asarray(state['position_initial_body_m'])
                np.testing.assert_array_equal(G,state['gyro_rotation_initial_body_from_current_body'])
                assert state['rgb_sha256']==d['rgb_sha256']
                if j==0:
                    np.testing.assert_array_equal(R,np.eye(3)); np.testing.assert_array_equal(position,np.zeros(3))
                else:
                    af,ad,ar,ap,ag=anchors[mode]; reg=state['registration']
                    assert state['reference_frame']==af and reg['reference_frame']==af
                    a,b,ua,ub=[np.asarray(reg[k]) for k in ('reference_inlier_points_body_m','current_inlier_points_body_m',
                        'reference_inlier_pixels','current_inlier_pixels')]
                    np.testing.assert_allclose(a,independent_lift(ad,ua),rtol=0,atol=2e-12)
                    np.testing.assert_allclose(b,independent_lift(d,ub),rtol=0,atol=2e-12)
                    Q,t=horn_fit(a,b) if mode=='joint' else (ag.T@G,a.mean(0)-(ag.T@G)@b.mean(0))
                    delta=max(float(np.max(abs(Q-reg['relative_rotation']))),float(np.max(abs(t-reg['translation_reference_body_m']))))
                    assert delta<1e-9
                    report[mode]['maximum_fit_coordinate_difference']=max(report[mode]['maximum_fit_coordinate_difference'],delta)
                    np.testing.assert_allclose(R,ar@Q,rtol=0,atol=1e-9); np.testing.assert_allclose(position,ap+ar@t,rtol=0,atol=1e-9)
                    report[mode]['raw_inlier_pairs']+=len(a)
                if j==0 or state['promoted_keyframe']:
                    nodes[mode].append(dict(frame=j,measured_ns=now,parent_frame=None if j==0 else anchors[mode][0],
                        position_initial_body_m=state['position_initial_body_m'],rotation_initial_body_from_current_body=state['rotation_initial_body_from_current_body'],pose_error_bound=None))
                    anchors[mode]=(j,d,R,position,G)
                expected_position=math.dist(position,truth[j])
                expected_angle=float(Rotation.from_matrix(R.T@R0.T@Rotation.from_quat(poses[sample,3:]).as_matrix()).magnitude())
                delta=max(abs(expected_position-frame['position_error_m']),abs(expected_angle-frame['orientation_error_rad']))
                assert delta<1e-9; report[mode]['maximum_score_difference']=max(report[mode]['maximum_score_difference'],delta)
            if j:
                w=evaluated['details'][mode]['windows'][j-1]
                stamps=[1_500_000_000+(j-1)*100_000_000+k*20_000_000 for k in range(6)]
                absent=sorted(set(stamps)&missing)
                assert w['start_ns']==stamps[0] and w['end_ns']==stamps[-1] and w['support_samples']==6
                assert w['unavailable_samples']==len(absent) and w['unavailable_measured_ns']==absent and w['contact_complete']==(not absent)
                assert w['phases']==sorted(set(int(v) for v in raw['phase'][sample-49:sample+1]))
                previous=record['rows'][j-1]['members'][mode]['state']
                if state is None or previous is None: assert w['displacement_error_m'] is None
                else:
                    error=math.sqrt(math.fsum(((state['position_initial_body_m'][k]-previous['position_initial_body_m'][k])-(truth[j][k]-truth[j-1][k]))**2 for k in range(3)))
                    assert abs(error-w['displacement_error_m'])<1e-12
    for mode in MODES:
        summary=evaluated['summaries'][mode]; frames=evaluated['details'][mode]['frames']; windows=evaluated['details'][mode]['windows']
        assert nodes[mode]==record['keyframes'][mode] and failures.get(mode)==record['failures'][mode]==summary['failure']
        assert summary['visual_available']==report[mode]['accepted'] and summary['frames']==226
        assert summary['contact_unavailable_at_capture']==sum(r['contact_unavailable_at_capture'] for r in frames)
        assert summary['visual_unavailable_at_contact_dropout']==sum(r['contact_unavailable_at_capture'] and not r['visual_available'] for r in frames)
        for name in ('position_error_m','orientation_error_rad'): check_stats(summary[name],[r[name] for r in frames if r['visual_available']])
        for phase,group in [('all',windows)]+[(key,[r for r in windows if r['phases']==[int(key)]]) for key in summary['by_phase']]:
            s=summary['windows'] if phase=='all' else summary['by_phase'][phase]
            for label,selected in [('all',group),('contact_complete',[r for r in group if r['contact_complete']]),('contact_dropout',[r for r in group if not r['contact_complete']])]:
                values=[r['displacement_error_m'] for r in selected if r['displacement_error_m'] is not None]
                assert s[label]['windows']==len(selected) and s[label]['visual_unavailable']==len(selected)-len(values)
                check_stats(s[label]['displacement_error_m'],values)
    return dict(models=report,missing_support_samples=len(missing),dropout_spans=len(evaluated['dropout_spans']))


def main():
    target=OUTPUT/'pose_dropout_audit_launch.json'
    if target.exists(): raise ValueError('exclusive pose/dropout audit')
    cv2.setNumThreads(1); launch=read_json(OUTPUT,'launch.json'); verify(launch); result=read_json(OUTPUT,'result.json')
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources=discover_sources(('scripts/audit_go2_friction_frozen_rgbd_dropout_v1.py',),launch['source_sha256'])
    verify_bindings(inputs|sources); write_json(target,dict(source_sha256=sources,input_sha256=inputs))
    reports={}
    try:
        predictions=read_json(OUTPUT,'predictions.json')['conditions']; evaluated=read_json(OUTPUT,'evaluation.json')
        for c in CONDITIONS:
            reports[c]=audit_condition(c,predictions[c],evaluated[c])
            assert result['summaries'][c]==evaluated[c]['summaries']
            print('FRICTION_POSE_DROPOUT_AUDIT_PASS',c,json.dumps(reports[c]),flush=True)
        verify(launch); verify_bindings(inputs|sources)
        write_json(OUTPUT/'pose_dropout_audit.json',dict(status='POSE_DROPOUT_AUDIT_PASS',conditions=reports,
            audit_launch_sha256=digest(target),estimator_rerun=False,feature_identity_proven=False,
            physical_error_calibrated=False,navigation_qualified=False))
    except Exception as error:
        write_json(OUTPUT/'pose_dropout_audit_failure.json',dict(status='TERMINAL_POSE_DROPOUT_AUDIT_FAILURE',reason=repr(error),completed_conditions=reports))
        raise


if __name__=='__main__': main()

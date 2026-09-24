"""Actual sensor-stream integration replay; no native pose or controller run."""
import json
import cv2
import numpy as np

from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.visual_led_motion_development import VisualLedMotion, ContactMotionSample, POSE_FIELDS
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

PREVIOUS=ROOT/'.generated/go2_friction_frozen_rgbd_dropout_v1_attempt_001'
INPUT=ROOT/'.generated/go2_support_friction_collection_v1_attempt_001'
OUTPUT=ROOT/'.generated/go2_visual_led_motion_integration_v1_attempt_001'
PROTOCOL='docs/go2_visual_led_motion_integration_v1_2026-09-06.md'
CONDITIONS=('nominal','lower_friction')
MODES=('joint','gyro')
IDENTITIES={
    'launch.json':'3642c3e4ac6ccc6f11113d1306c4cf03bf36be1d2ec2b834e97fe387a6267647',
    'result.json':'fd3bb8615f0c5518369664de74220196dfa9ce716f91442d76489b469395f2a3',
    'pose_dropout_audit_launch.json':'9c0da125b3dc4c3095e0d3e5784ce85d42ba7d9564b5ca866e3f64b306154195',
    'pose_dropout_audit.json':'4ce1d13f19bd35016a4fc2166b197a9a3745365e021503669400d24531682bd8'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json'); verify(old)
    audit=read_json(PREVIOUS,'pose_dropout_audit_launch.json'); verify_bindings(audit['source_sha256']|audit['input_sha256'])
    result=read_json(PREVIOUS,'result.json')
    inputs=old['input_sha256']|audit['input_sha256']|ids|{
        str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_visual_led_motion_integration_v1.py',
        'lewm/tests/test_visual_led_motion_development.py'),old['source_sha256']|audit['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch|=dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,
        scope='visual-led optional-contact interface integration; no new estimator selection or physical execution',
        physical_execution=False,model_fitting=False,conditions=CONDITIONS,modes=MODES)
    verify(launch); return launch


def replay(condition):
    directory=INPUT/condition
    # Acquisition identities are stream metadata, not evaluator pose/terrain.
    specification=read_json(directory,'specification.json')
    acquisition=specification['ideal_foot_sensor_identity']
    support={r['measured_ns']:r for r in read_json(directory,'support_predictions.json')['rows']}
    models={m:VisualLedMotion(m,identity=(0,0,0),contact_acquisition_identity=acquisition) for m in MODES}
    rows=[]
    for frame in range(226):
        p,d=load_rgbd_observation(directory,frame); fast=load_fast_packet(directory,frame); now=p['sensor_state']['decision_ns']
        samples=None
        if frame:
            samples=[]
            for stamp in range(now-100_000_000,now+1,20_000_000):
                r=support[stamp]
                samples.append(ContactMotionSample((0,0,0),acquisition,stamp,stamp,
                    r['modes']['level_sphere_rolling']['consensus_velocity_body_m_s'],r['rotation_gyro_anchor_from_body']))
        members={m:model.observe(p,d,fast,now_ns=now,contact_samples=samples) for m,model in models.items()}
        # A 50Hz consumer must not mistake a retained 10Hz pose for a new one.
        intermediate=[]
        if frame<225:
            for offset in (20_000_000,40_000_000,60_000_000,80_000_000):
                snapshots={m:model.snapshot(now_ns=now+offset) for m,model in models.items()}
                intermediate.append(dict(decision_ns=now+offset,members={m:{k:s[k] for k in
                    ('status','current_pose','visual_age_ns','contact_diagnostic_current','motion_permission')} for m,s in snapshots.items()}))
        rows.append(dict(frame=frame,measured_ns=now,members=members,intermediate_queries=intermediate))
        if frame%50==0: print('VISUAL_LED_INTEGRATION',condition,frame,flush=True)
    return rows


def check(records):
    # Saved immutable visual-only outputs are an identity witness, not fresh truth.
    old=read_json(PREVIOUS,'predictions.json')['conditions']; reports={}
    for c,rows in records.items():
        modes={}
        for m in MODES:
            statuses={}; disagreement=[]; intermediate=0
            for r,witness in zip(rows,old[c]['rows'],strict=True):
                state=r['members'][m]; expected=witness['members'][m]['state']
                assert state['status']=='CURRENT_VISUAL_POSE'
                assert {k:state['current_pose'][k] for k in POSE_FIELDS}=={k:expected[k] for k in POSE_FIELDS}
                assert state['motion_permission']=='NOT_EVALUATED' and not state['pose_updated_from_contact']
                d=state['contact_diagnostic']; statuses[d['status']]=statuses.get(d['status'],0)+1
                if d['displacement_disagreement_m'] is not None: disagreement.append(d['displacement_disagreement_m'])
                for query in r['intermediate_queries']:
                    s=query['members'][m]; assert s['current_pose'] is None and not s['contact_diagnostic_current']
                    assert s['status']=='TRANSLATION_AND_ORIENTATION_UNOBSERVED_SINCE_LAST_VISUAL'
                    assert s['visual_age_ns']==query['decision_ns']-r['measured_ns']; intermediate+=1
            modes[m]=dict(visual_outputs_exact=len(rows),intermediate_unobserved_queries=intermediate,
                contact_status_counts=statuses,contact_displacement_disagreement_m=dict(count=len(disagreement),
                    mean=float(np.mean(disagreement)) if disagreement else None,
                    maximum=float(max(disagreement)) if disagreement else None))
        reports[c]=modes
    return reports


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive interface integration replay')
    cv2.setNumThreads(1); launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        records={c:replay(c) for c in CONDITIONS}
        write_json(OUTPUT/'motion_evidence.json',dict(conditions=records,native_pose_loaded=False,model_fitting=False))
        report=check(records); verify(launch)
        result=dict(status='VISUAL_LED_MOTION_INTEGRATION_PASS',conditions=report,
            artifact_sha256={'motion_evidence.json':digest(OUTPUT/'motion_evidence.json')},
            physical_execution=False,model_fitting=False,physical_error_calibrated=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result); print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VISUAL_LED_INTEGRATION_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()

"""All 229 adjacent pairs before the recorded terminal visual failure."""
import json
import time
import cv2
import numpy as np
from lewm.overlap_retention_joint_observer_development import OverlapRetentionVisualLedMotion
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.descriptor_pair_stage_diagnostic_development import counted_descriptor_matches
from lewm.keyframe_rgbd_pose_development import matched_points as original_matches
from lewm.gyro_seeded_rgbd_correspondence_development import matched_points as flow_matches
from lewm.joint_rgbd_rigid_pose_development import register, angle, RIGID_RULES
from lewm.causal_sensor_state import SensorContractError
from lewm.physical_execution_development import rotation_xyzw
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.read_go2_exact_terminal_target_v1 import (
    INPUT, READOUT, INPUT_SHA, READOUT_SHA, CASE, BASE, create_output,
    validate_root, verify_artifacts, digest, write_json, read_json, discover_sources, verify, hardware)

OUTPUT = BASE/'go2_adjacent_pair_correspondence_v1_attempt_001'
PROTOCOL = 'docs/go2_adjacent_pair_correspondence_v1_2026-09-08.md'


def fit_pair(points, relative_gyro, frame):
    a,b,ua,ub,counts = points
    report = dict(counts=counts, points_reference_body_m=a.tolist(), points_current_body_m=b.tolist(),
        pixels_reference=ua.tolist(), pixels_current=ub.tolist(), qualified=False, failure=None)
    try:
        R,t,mask,reg = register(a,b,ua,ub,gyro_rotation=relative_gyro,mode='joint',frame=frame)
        # The reference is the immediately previous accepted pose, so these are
        # exactly the existing current-increment translation/rotation envelopes.
        if (np.linalg.norm(t)>RIGID_RULES['maximum_increment_translation_m']
                or angle(R)>RIGID_RULES['maximum_increment_rotation_rad']):
            raise SensorContractError('consecutive rigid-pose displacement envelope rejected')
        report.update(qualified=True, relative_rotation=R.tolist(), translation_reference_body_m=t.tolist(),
            inlier_mask=mask.tolist(), registration=reg)
    except SensorContractError as error:
        report['failure'] = str(error)
    return report


def sensor_pairs():
    reader = IntentReturnRGBDReplay(INPUT/CASE)
    original_rows = read_json(INPUT/CASE, 'context_decisions.json')
    assert len(reader.frames)==240 and len(original_rows)==240
    motion = OverlapRetentionVisualLedMotion(); pairs=[]
    for frame in range(230):
        p,d,f,now = reader.packet(frame)
        previous = motion.model.previous
        evidence = motion.observe(p,d,f,now_ns=now)
        assert json.loads(json.dumps(evidence)) == original_rows[frame]['decision']['evidence']
        if frame==0:
            continue
        assert previous.frame==frame-1 and previous.measured_ns==now-100_000_000
        assert motion.model.gyro.last_ns==now
        G=motion.model.gyro.rotation.copy(); relative=previous.gyro.T@G
        current=CornerSupportFeatureFrame(p['image']['rgb'],d)
        start=time.perf_counter()
        descriptor=counted_descriptor_matches(previous.features,current)
        check=original_matches(previous.features,current)
        assert all(np.array_equal(a,b) for a,b in zip(descriptor[:4],check,strict=True))
        original=fit_pair(descriptor,relative,frame)
        descriptor_s=time.perf_counter()-start
        start=time.perf_counter()
        flow=fit_pair(flow_matches(previous.features,current,relative),relative,frame)
        flow_s=time.perf_counter()-start
        pairs.append(dict(frame=frame, reference_frame=previous.frame, measured_ns=now,
            current_rgb_sha256=d['rgb_sha256'], reference_feature_witness=previous.features.witness(),
            current_feature_witness=current.witness(), relative_gyro_rotation=relative.tolist(),
            descriptor=original, adjacent_flow=flow, descriptor_arrays_exact=True,
            descriptor_with_duplicate_array_check_wall_s=descriptor_s, flow_wall_s=flow_s,
            observer_pose_replaced=False, native_coordinates_parsed=False))
    assert len(pairs)==229 and motion.model.failed
    return pairs


def evaluate(pairs):
    # All pair fits have been persisted before this evaluator opens native pose.
    cameras=read_json(INPUT/CASE,'camera_audit.json')
    with np.load(INPUT/CASE/'physics_trace.npz',allow_pickle=False) as z:
        poses=z['base_pose_world']
    rows=[]
    for row in pairs:
        a=poses[cameras[row['reference_frame']]['physical_sample_index']]
        b=poses[cameras[row['frame']]['physical_sample_index']]
        Ra,Rb=rotation_xyzw(a[3:]),rotation_xyzw(b[3:]); actual_t=Ra.T@(b[:3]-a[:3]); actual_R=Ra.T@Rb
        errors={}
        for arm in ('descriptor','adjacent_flow'):
            fit=row[arm]
            errors[arm]=None if not fit['qualified'] else dict(
                translation_error_m=float(np.linalg.norm(np.asarray(fit['translation_reference_body_m'])-actual_t)),
                rotation_error_rad=angle(actual_R.T@np.asarray(fit['relative_rotation'])))
        rows.append(dict(frame=row['frame'], errors=errors, native_evaluator_only=True))
    return rows


def main():
    if not __debug__:raise ValueError('enabled assertions required')
    cv2.setNumThreads(1)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive adjacent-pair diagnostic')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA}); prior=read_json(INPUT,'result.json')
    assert prior['status']=='OVERLAP_RETENTION_GOAL_PROBE_COMPLETE'
    ids={'result.json':INPUT_SHA}|prior['artifact_sha256'];verify_artifacts(INPUT,ids)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    assert readout['probe_result_sha256']==INPUT_SHA
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']}
    verify_artifacts(READOUT,readout_ids); original=read_json(READOUT,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_adjacent_pair_correspondence_v1.py',
        'docs/go2_exact_terminal_target_result_2026-09-08.md',
        'docs/go2_measured_pose_correspondence_result_2026-09-07.md',
        'docs/go2_gyro_seeded_correspondence_result_2026-09-07.md'),original['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<4*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('diagnostic resource allowance unavailable')
    launch=original|dict(source_sha256=sources,output_root=str(OUTPUT),protocol=PROTOCOL,
        probe_artifact_sha256=ids,readout_artifact_sha256=readout_ids,hardware=resources,
        native_execution=False,workers=1,threads=1,pair_frames=list(range(1,230)),
        concurrency_reason='one causal original observer and 229 small adjacent-pair comparisons',
        earlier_flow_observers_remain_rejected=True,full_observer_candidate_defined=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('ADJACENT_PAIR_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        pairs=sensor_pairs()
        assert len(json.dumps(pairs,allow_nan=False).encode())<256*1024**2
        write_json(OUTPUT/'sensor_pairs.json',pairs)
        marker=digest(OUTPUT/'sensor_pairs.json')
        verify_artifacts(OUTPUT,{'sensor_pairs.json':marker})
        errors=evaluate(read_json(OUTPUT,'sensor_pairs.json'));write_json(OUTPUT/'evaluation.json',errors)
        reports={}
        for arm in ('descriptor','adjacent_flow'):
            values=[e['errors'][arm] for e in errors if e['errors'][arm] is not None]
            reports[arm]=dict(qualified_pairs=len(values),failed_frames=[r['frame'] for r in pairs if not r[arm]['qualified']],
                maximum_translation_error_m=max((v['translation_error_m'] for v in values),default=None),
                maximum_rotation_error_rad=max((v['rotation_error_rad'] for v in values),default=None))
        verify(launch);verify_artifacts(INPUT,ids);verify_artifacts(READOUT,readout_ids)
        products={f:digest(OUTPUT/f) for f in ('launch.json','sensor_pairs.json','evaluation.json')}
        write_json(OUTPUT/'result.json',dict(status='ADJACENT_PAIR_CORRESPONDENCE_DIAGNOSTIC_COMPLETE',
            source_sha256=sources,artifact_sha256=products,pairs=229,original_evidence_rows_exact=230,
            reports=reports,final_pair=pairs[-1],wall_s=time.perf_counter()-started,hardware_after=hardware(),
            full_observer_evaluated=False,observer_adopted=False,original_outcome_changed=False,
            native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('ADJACENT_PAIR_COMPLETE',digest(OUTPUT/'result.json'),json.dumps(reports),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ADJACENT_PAIR_DIAGNOSTIC_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()

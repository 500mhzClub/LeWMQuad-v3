"""Reserved nominal estimator transfer; predictions precede all native scoring."""
import json
import time

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import RigidRGBDKeyframePose
from lewm.pose_coverage_diagnostic_development import query_coverage
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_joint_rgbd_rigid_pose_development_v1 import quaternion_rotation, angular_distance
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_bounded_depth_surface_development_v1 import HYPOTHESES
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_longer_observed_floor_motion_development_v1 import OUTPUT as INPUT, TRIALS, PROTOCOL
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

PREDECESSOR = ROOT/'.generated/go2_longer_motion_frozen_pose_development_v1_attempt_001'
OUTPUT = ROOT/'.generated/go2_longer_motion_frozen_pose_coverage_status_v1_attempt_001'
STATUS_PROTOCOL = 'docs/go2_longer_motion_frozen_pose_coverage_status_v1_2026-09-06.md'
MODES = ('joint', 'gyro')
IDENTITIES = {
    'launch.json': '1f762003db74c10d13f04af2e5164d5a00cb596f586cdc3e23a14885fa11ee04',
    'result.json': '87d5ab645887ac86475de8a26850fda0fd85b0eaac5e999a858c77db1a8b57d3',
}


def coverage_summary(rows):
    full = [r['frame'] for r in rows if r['floor_coverage'] and all(r['floor_coverage'].values())]
    return dict(queries=len(rows), maximum_covered_shapes=max((sum(r['floor_coverage'].values()) for r in rows), default=0),
        first_full_coverage_frame=full[0] if full else None, full_coverage_frames=len(full))


def initial_up(policy):
    force = policy['sensor_state']['sensed']['specific_force']
    mean = np.asarray(force['values']).mean(0); norm = float(np.linalg.norm(mean))
    if not force['valid'].all() or not np.isfinite(mean).all() or abs(norm-9.81)>.75:
        raise SensorContractError('initial gravity hypothesis not quiet')
    return mean/norm


def preflight():
    identities = {'launch.json': 'dda0037e6cef3f990492b866ca0705ad768e76f2ba89d8cf95440455aa7229c7',
                  'failure.json': '64f7578718d02f2ca85d76b420d04657287553bc0ab815354f0152d36cfb2548'}
    inputs = {str((PREDECESSOR/n).relative_to(ROOT)):h for n,h in identities.items()}
    verify_bindings(inputs)
    prior=read_json(PREDECESSOR,'launch.json'); verify(prior)
    assert read_json(PREDECESSOR,'failure.json')['status']=='TERMINAL_FROZEN_TRANSFER_FAILURE'
    sources=discover_sources((STATUS_PROTOCOL,'scripts/probe_go2_longer_motion_pose_coverage_status_development_v1.py',
        'lewm/tests/test_pose_coverage_diagnostic_development.py'),prior['source_sha256'])
    bound=prior | dict(source_sha256=sources,input_sha256=prior['input_sha256'] | inputs,
        diagnostic_protocol=STATUS_PROTOCOL,
        scope='explicit coverage unavailability without modifying frozen pose models, coverage thresholds, trials or scoring')
    verify(bound); return bound


def replay():
    geometry = ArticulatedCollisionGeometry(URDF); records = {}
    for trial in TRIALS:
        directory = INPUT/trial; models = {m:RigidRGBDKeyframePose(m) for m in MODES}; failures = {}; rows = []
        surface = None; errors = None
        for frame in range(586):
            p,d = load_rgbd_observation(directory,frame); fast = load_fast_packet(directory,frame)
            now = p['sensor_state']['decision_ns']; items = {}
            for mode,model in models.items():
                start = time.perf_counter(); query = None
                if mode in failures:
                    item = dict(status='NOT_REINVOKED_AFTER_FAILURE', state=None, failure=failures[mode])
                else:
                    try:
                        state = model.observe(p,d,fast,now_ns=now)
                    except SensorContractError as error:
                        chain=[]; cause=error
                        while cause is not None: chain.append(str(cause)); cause=cause.__cause__
                        failures[mode] = dict(frame=frame,measured_ns=now,chain=chain)
                        item = dict(status='TERMINAL_FAILURE',state=None,failure=failures[mode])
                    else:
                        item = dict(status='CONDITIONAL_RIGID_POSE',state=state)
                        joints = p['sensor_state']['sensed']['joints']['values'][-1,:12]
                        if frame==0 and surface is None:
                            surface = BoundedDepthSurface(d['depth_m'],d['valid'],initial_up(p),**HYPOTHESES)
                            errors = dict.fromkeys([s['shape_id'] for s in geometry.supports(joints,np.eye(3))['shapes']],0.)
                evidence=query_coverage(surface,geometry,p['sensor_state']['sensed']['joints']['values'][-1,:12],
                    item['state'],errors)
                item['floor_coverage']=evidence['floor_coverage']
                item['coverage_evidence']=evidence
                item['wall_ms'] = 1000*(time.perf_counter()-start); items[mode] = item
            rows.append(dict(frame=frame,measured_ns=now,members=items))
            if frame%50==0: print('FROZEN_TRANSFER',trial,frame,flush=True)
        records[trial] = dict(rows=rows,keyframes={m:v.nodes for m,v in models.items()},
            surface_status=None if surface is None else surface.status,zero_additional_shape_error_diagnostic_only=True)
    return plain(dict(trials=records,native_pose_loaded=False,parameter_fitting=False,navigation_qualified=False))


def score(predictions):
    summaries={}; details={}; comparisons={}
    for trial,record in predictions['trials'].items():
        directory=INPUT/trial; cameras=read_json(directory,'camera_audit.json')
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as raw: poses=raw['base_pose_world'].copy()
        native=[quaternion_rotation(p[[6,3,4,5]]) for p in poses]; R0=native[749]; p0=poses[749,:3]
        per_mode={}
        evaluator=read_json(INPUT,trial+'_raw_acquisition_audit_details.json')['evaluator_only_initial_surface_queries']
        truth_coverage={r['frame']:r['floor_coverage'] for r in evaluator}
        for mode in MODES:
            values=[]; queries=[]; failure=None; false_covered=missed_covered=compared_shapes=0
            for row in record['rows']:
                item=row['members'][mode]; s=item['state']; frame=row['frame']
                if item['status']=='TERMINAL_FAILURE': failure=item['failure']
                if s is None: continue
                sample=cameras[frame]['physical_sample_index']; truth=R0.T@(poses[sample,:3]-p0); Rt=R0.T@native[sample]
                values.append(dict(frame=frame,position_error_m=float(np.linalg.norm(np.asarray(s['position_initial_body_m'])-truth)),
                    orientation_error_rad=angular_distance(np.asarray(s['rotation_initial_body_from_current_body']),Rt)))
                if item['floor_coverage'] is not None:
                    coverage=item['floor_coverage']; queries.append(dict(frame=frame,floor_coverage=coverage))
                    if frame in truth_coverage:
                        expected=truth_coverage[frame]; assert set(coverage)==set(expected)
                        false_covered+=sum(coverage[k] and not expected[k] for k in expected)
                        missed_covered+=sum(not coverage[k] and expected[k] for k in expected); compared_shapes+=len(expected)
            name=trial+'__'+mode; details[name]=values; per_mode[mode]={v['frame']:v for v in values}
            summaries[name]=dict(admitted=len(values),failure=failure,keyframes=len(record['keyframes'][mode]),
                maximum_position_error_m=max((v['position_error_m'] for v in values),default=None),
                maximum_orientation_error_rad=max((v['orientation_error_rad'] for v in values),default=None),
                causal_initial_surface_coverage=coverage_summary(queries),
                compared_shape_queries=compared_shapes,estimated_covered_native_uncovered=false_covered,
                estimated_uncovered_native_covered=missed_covered,error_bounds_calibrated=False)
        shared=sorted(set(per_mode['joint'])&set(per_mode['gyro']))
        comparisons[trial]=dict(common_admitted_frames=len(shared), **{mode+'_maximum_position_error_m':
            max((per_mode[mode][i]['position_error_m'] for i in shared),default=None) for mode in MODES})
    return dict(summaries=summaries,details=details,comparisons=comparisons,
        native_evaluator_only=True,parameter_fitting=False,validation_used_for_selection=False,navigation_qualified=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh exclusive frozen transfer only')
    cv2.setNumThreads(1); launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        predictions=replay(); write_json(OUTPUT/'predictions.json',predictions)
        evaluated=score(predictions); write_json(OUTPUT/'evaluation.json',evaluated); verify(launch)
        coverage_outcomes={}
        for trial,record in predictions['trials'].items():
            for mode in MODES:
                evidence=[r['members'][mode]['coverage_evidence'] | dict(frame=r['frame']) for r in record['rows']]
                statuses={s:sum(r['status']==s for r in evidence) for s in sorted(set(r['status'] for r in evidence))}
                rejected=[r for r in evidence if r['status']=='COVERAGE_CONTRACT_REJECTED']
                coverage_outcomes[trial+'__'+mode]=dict(counts=statuses,first_rejection=rejected[0] if rejected else None,
                    maximum_orthogonality_defect=max((r.get('orthogonality_max_abs',0.) for r in evidence),default=0.),
                    maximum_determinant_defect=max((r.get('determinant_abs_error',0.) for r in evidence),default=0.))
        result=dict(status='FROZEN_POSE_EXPLICIT_COVERAGE_STATUS_COMPLETE',coverage_outcomes=coverage_outcomes,summaries=evaluated['summaries'],comparisons=evaluated['comparisons'],
            artifact_sha256={n:digest(OUTPUT/n) for n in ('predictions.json','evaluation.json')},
            physics_executed=False,parameter_fitting=False,validation_used_for_selection=False,
            error_bounds_calibrated=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result); print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FROZEN_TRANSFER_FAILURE',reason=str(error))); raise


if __name__=='__main__': main()

"""Existing poses -> explicitly enlarged geometry queries; no model replay."""
import json
from itertools import product

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.bounded_rotation_geometry_adapter_development import adapt_geometry_query
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.causal_sensor_state import SensorContractError
from lewm.coupled_floor_enclosure_development import coupled_footprint_enclosure
from lewm.floor_footprint_bounds_development import projected_footprint_rectangles
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_bounded_depth_surface_development_v1 import HYPOTHESES
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_longer_motion_pose_coverage_status_development_v1 import OUTPUT as PREVIOUS,INPUT,MODES,initial_up,coverage_summary
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_bounded_rotation_geometry_adapter_v1_attempt_001'
PROTOCOL='docs/go2_bounded_rotation_geometry_adapter_v1_2026-09-06.md'
IDENTITIES={'launch.json':'b022ac51531226205e09b9dd42ef631d761aec2149955589c4960051c77ba7e6',
    'result.json':'23b3861c028e8207c748dc55f520f84a8cf941b67ac94d62fd05d735ba289c34',
    'pose_chain_coverage_audit_launch.json':'2c3b87e93f0077e2db8ec7d120d83dee711964be4404c07f48d3eabbeccd0a7f',
    'pose_chain_coverage_audit.json':'82bc0e7278a704359151fa04a6fe4b056dffd86e2065f75012284789d96466e9'}


def preflight():
    inputs={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(inputs)
    previous=read_json(PREVIOUS,'launch.json'); verify(previous)
    audit=read_json(PREVIOUS,'pose_chain_coverage_audit_launch.json'); verify_bindings(audit['source_sha256']|audit['input_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_bounded_rotation_geometry_adapter_v1.py',
        'lewm/tests/test_bounded_rotation_geometry_adapter_development.py'),previous['source_sha256']|audit['source_sha256'])
    launch=previous|dict(source_sha256=sources,input_sha256=previous['input_sha256']|inputs|audit['input_sha256'],
        diagnostic_protocol=PROTOCOL,scope='saved-pose bounded numerical geometry adapter and startup projection; no model rerun or physics')
    verify(launch); return launch


def startup_projection(surface,geometry,joints,policy):
    if surface.status!='BOUNDED_MEASURED_SURFACE_AVAILABLE': return dict(surface_status=surface.status)
    shapes=geometry.supports(joints,np.eye(3))['shapes']; low,high=[np.array([s[k] for s in shapes]) for k in ('lower','upper')]
    enclosure=coupled_footprint_enclosure(low,high,np.broadcast_to(surface.anchor,low.shape),
        np.broadcast_to(surface.normal,low.shape),surface.frame._up,normal_error=0.,
        up_error=surface.up_error,plane_offset_error=surface.surface_tube_m)
    lo,hi=enclosure['footprint_lower_m'],enclosure['footprint_upper_m']
    projected=projected_footprint_rectangles(lo,hi,np.zeros(len(shapes)),np.zeros(len(shapes)),surface.frame._up)
    corners=np.array(list(product((False,True),repeat=3)))
    points=np.where(corners[None],hi[:,None],lo[:,None]); T=np.asarray(BODY_FROM_OPTICAL)
    optical=(points-T[:3,3])@T[:3,:3]; z=optical[:,:,2]
    return dict(surface_status=surface.status,sensed_channels=sorted(policy['sensor_state']['sensed']),
        shapes=[dict(shape_id=s['shape_id'],minimum_optical_depth_m=float(z[i].min()),
            maximum_optical_depth_m=float(z[i].max()),entire_enclosure_before_minimum_range=bool(z[i].max()<.2),
            complete_frustum_projection=bool(projected['projection_within_observed_camera'][i])) for i,s in enumerate(shapes)],
        complete_frustum_shapes=int(projected['projection_within_observed_camera'].sum()),
        contact_modality_present=False,unobserved_floor_support_assumed=False,hardware_sensing_qualified=False)


def predict():
    previous=read_json(PREVIOUS,'predictions.json'); geometry=ArticulatedCollisionGeometry(URDF); trials={}
    for trial,record in previous['trials'].items():
        p,d=load_rgbd_observation(INPUT/trial,0); joints=p['sensor_state']['sensed']['joints']['values'][-1,:12]
        surface=BoundedDepthSurface(d['depth_m'],d['valid'],initial_up(p),**HYPOTHESES)
        errors=dict.fromkeys([s['shape_id'] for s in geometry.supports(joints,np.eye(3))['shapes']],0.)
        startup=startup_projection(surface,geometry,joints,p); rows=[]
        for row in record['rows']:
            p,_=load_rgbd_observation(INPUT/trial,row['frame']); joints=p['sensor_state']['sensed']['joints']['values'][-1,:12]; items={}
            for mode in MODES:
                original=row['members'][mode]; state=original['state']
                item=dict(original_coverage_status=original['coverage_evidence']['status'],
                    original_floor_coverage=original['floor_coverage'],floor_coverage=None,correction=None)
                if state is None: item['status']='POSE_UNAVAILABLE'
                elif surface.status!='BOUNDED_MEASURED_SURFACE_AVAILABLE': item['status']='SURFACE_UNAVAILABLE'
                else:
                    query,correction=adapt_geometry_query(geometry,joints,state,errors); item['correction']=correction
                    item['geometry_rotation']=query['rotation_observation_from_body'].tolist()
                    item['total_point_error_by_shape']=query['point_error_by_shape']
                    try: result=surface.query(geometry,joints,**query)
                    except SensorContractError as error: item|=dict(status='GEOMETRY_QUERY_REJECTED',reason=str(error))
                    else: item|=dict(status='CONDITIONAL_NUMERICALLY_ENCLOSED_COVERAGE',floor_coverage=result['floor_coverage'])
                items[mode]=item
            rows.append(dict(frame=row['frame'],measured_ns=row['measured_ns'],members=items))
        trials[trial]=dict(rows=rows,startup_projection=startup)
        print('BOUNDED_ROTATION_QUERIES_COMPLETE',trial,flush=True)
    return plain(dict(trials=trials,models_rerun=False,physical_error_hypotheses_zero_not_validated=True,native_pose_loaded=False))


def score(predictions):
    summaries={}
    for trial,record in predictions['trials'].items():
        truth={r['frame']:r['floor_coverage'] for r in read_json(INPUT,trial+'_raw_acquisition_audit_details.json')['evaluator_only_initial_surface_queries']}
        for mode in MODES:
            rows=[]; repaired=changed=failures=false_covered=missed=0; maximum=0.
            for row in record['rows']:
                item=row['members'][mode]; coverage=item['floor_coverage']; old=item['original_floor_coverage']
                if item['correction']: maximum=max(maximum,max(item['correction']['numerical_shape_correction_m'].values()))
                if coverage is None: failures+=1; continue
                repaired+=int(item['original_coverage_status']=='COVERAGE_CONTRACT_REJECTED')
                if old is not None: changed+=sum(old[k]!=coverage[k] for k in coverage)
                expected=truth[row['frame']]; assert set(expected)==set(coverage)
                false_covered+=sum(coverage[k] and not expected[k] for k in coverage)
                missed+=sum(not coverage[k] and expected[k] for k in coverage)
                rows.append(dict(frame=row['frame'],floor_coverage=coverage))
            summaries[trial+'__'+mode]=dict(coverage=coverage_summary(rows),original_rejections_now_with_conditional_query=repaired,
                changed_previously_available_shape_queries=changed,unavailable_queries=failures,
                estimated_covered_native_uncovered=false_covered,estimated_uncovered_native_covered=missed,
                maximum_numerical_shape_correction_m=maximum,physical_uncertainty_calibrated=False)
    return summaries


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive bounded-rotation diagnostic only')
    cv2.setNumThreads(1); launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        predictions=predict(); write_json(OUTPUT/'predictions.json',predictions)
        summaries=score(predictions); verify(launch)
        write_json(OUTPUT/'result.json',dict(status='BOUNDED_ROTATION_GEOMETRY_DIAGNOSTIC_COMPLETE',summaries=summaries,
            startup={t:r['startup_projection'] for t,r in predictions['trials'].items()},
            artifact_sha256={'predictions.json':digest(OUTPUT/'predictions.json')},models_rerun=False,physics_executed=False,
            error_bounds_calibrated=False,navigation_qualified=False,goal_achieved=False))
        print(json.dumps(summaries),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_BOUNDED_ROTATION_DIAGNOSTIC_FAILURE',reason=str(error))); raise


if __name__=='__main__': main()

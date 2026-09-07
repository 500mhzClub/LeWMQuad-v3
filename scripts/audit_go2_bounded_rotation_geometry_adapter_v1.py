"""All-box-corner containment in extended precision and direct-cell coverage."""
import json
from itertools import product

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_bounded_depth_surface_development_v1 import HYPOTHESES
from scripts.probe_go2_bounded_rotation_geometry_adapter_v1 import OUTPUT,PREVIOUS,INPUT,MODES,initial_up
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def main():
    cv2.setNumThreads(1); target=OUTPUT/'corner_and_cell_audit_launch.json'
    if target.exists(): raise ValueError('exclusive audit only')
    launch=read_json(OUTPUT,'launch.json'); result=read_json(OUTPUT,'result.json'); verify(launch)
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    inputs|={str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('launch.json','result.json')}
    sources=discover_sources(('scripts/audit_go2_bounded_rotation_geometry_adapter_v1.py',),launch['source_sha256'])
    verify_bindings(inputs|sources); write_json(target,dict(source_sha256=sources,input_sha256=inputs))
    reports={}
    try:
        old=read_json(PREVIOUS,'predictions.json'); new=read_json(OUTPUT,'predictions.json')
        geometry=ArticulatedCollisionGeometry(URDF); corners=np.array(list(product((False,True),repeat=3)))
        for trial,record in new['trials'].items():
            p,d=load_rgbd_observation(INPUT/trial,0); surface=BoundedDepthSurface(d['depth_m'],d['valid'],initial_up(p),**HYPOTHESES)
            count=queries=0; actual_max=0.; bound_max=0.; minimum_margin=float('inf')
            for row,original in zip(record['rows'],old['trials'][trial]['rows'],strict=True):
                assert row['frame']==original['frame'] and row['measured_ns']==original['measured_ns']
                p,_=load_rgbd_observation(INPUT/trial,row['frame']); joints=p['sensor_state']['sensed']['joints']['values'][-1,:12]
                shapes=geometry.supports(joints,np.eye(3))['shapes']
                lo,hi=[np.array([s[k] for s in shapes],dtype=np.longdouble) for k in ('lower','upper')]
                vertices=np.where(corners[None],hi[:,None],lo[:,None])
                for mode in MODES:
                    item=row['members'][mode]; prior=original['members'][mode]; state=prior['state']
                    assert item['original_coverage_status']==prior['coverage_evidence']['status']
                    assert item['original_floor_coverage']==prior['floor_coverage']
                    if state is None:
                        assert item['status']=='POSE_UNAVAILABLE' and item['floor_coverage'] is None; continue
                    assert item['status']=='CONDITIONAL_NUMERICALLY_ENCLOSED_COVERAGE'
                    R=np.asarray(state['rotation_initial_body_from_current_body'],dtype=np.longdouble)
                    Q=np.asarray(item['geometry_rotation']); correction=item['correction']
                    np.testing.assert_allclose(Q.T@Q,np.eye(3),rtol=0,atol=1e-12)
                    assert abs(np.linalg.det(Q)-1)<=1e-12
                    delta=R-Q.astype(np.longdouble)
                    assert np.sqrt(np.sum(delta*delta))<=correction['matrix_difference_frobenius_upper']
                    movement=vertices@delta.T; distances=np.sqrt(np.sum(movement*movement,axis=2))
                    bounds=np.array([correction['numerical_shape_correction_m'][s['shape_id']] for s in shapes],dtype=np.longdouble)
                    assert (distances<=bounds[:,None]).all()
                    for s in shapes:
                        k=s['shape_id']; assert correction['physical_error_hypotheses'][k]==0.
                        assert item['total_point_error_by_shape'][k]>=correction['numerical_shape_correction_m'][k]
                    assert correction['estimator_rotation_unchanged'] and not correction['physical_uncertainty_calibrated']
                    count+=distances.size; actual_max=max(actual_max,float(distances.max())); bound_max=max(bound_max,float(bounds.max()))
                    minimum_margin=min(minimum_margin,float((bounds[:,None]-distances).min()))
                    query=surface.query(geometry,joints,rotation_observation_from_body=Q,
                        translation_observation_from_body=state['position_initial_body_m'],
                        point_error_by_shape=item['total_point_error_by_shape'],backend='reference')
                    assert query['floor_coverage']==item['floor_coverage']; queries+=1
            reports[trial]=dict(extended_precision_box_corners_checked=count,reference_cell_queries=queries,
                maximum_actual_corner_displacement_m=actual_max,maximum_supplied_correction_m=bound_max,
                minimum_corner_containment_margin_m=minimum_margin)
            print('ROTATION_CORNER_AND_CELL_AUDIT_PASS',trial,json.dumps(reports[trial]),flush=True)
        verify(launch); verify_bindings(inputs|sources)
        write_json(OUTPUT/'corner_and_cell_audit.json',dict(status='ROTATION_CORNER_AND_CELL_AUDIT_PASS',trials=reports,
            audit_launch_sha256=digest(target),physical_uncertainty_calibrated=False,
            shared_surface_and_kinematic_construction=True,navigation_qualified=False))
    except Exception as error:
        write_json(OUTPUT/'corner_and_cell_audit_failure.json',dict(status='TERMINAL_CORNER_CELL_AUDIT_FAILURE',reason=str(error),completed_trials=reports)); raise


if __name__=='__main__': main()

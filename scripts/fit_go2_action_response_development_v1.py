"""One A-only sensor-target fit. No B data or native trajectory labels are read."""
import json

import numpy as np
from scipy.spatial.transform import Rotation

from lewm.action_motion_identification_development import MotionState,MotionIdentificationController,motion_priors
from lewm.action_response_model_development import features,fit_response,model_identity,policy_state
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.run_go2_action_motion_identification_development_v1 import OUTPUT as DATA,PROTOCOL as DATA_PROTOCOL
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_startup_observation_turn_development_v1 import verify_extensions
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_action_response_fit_development_v1_attempt_001'
PROTOCOL='docs/go2_action_response_fit_development_v1_2026-09-06.md'
IDENTITIES={'launch.json':'874ae72a44e99dee15fbaf0a6c5c826ec1f9d427b23bd852780453d645e3491f',
 'result.json':'c39ddd478788e3053f0484928a8ff06ca736cbb86a26f9552200711d0063171f',
 'raw_artifact_audit.json':'2f6a6d8c257b20990e5a587a57d97ff3321b44c1c8a370fdba2c27fa01f82b46'}
SEEDS=('scripts/fit_go2_action_response_development_v1.py','lewm/tests/test_action_response_model_development.py',PROTOCOL)


def preflight():
    identities={str((DATA/name).relative_to(ROOT)):h for name,h in IDENTITIES.items()};verify_bindings(identities)
    launch=json.loads((DATA/'launch.json').read_text()); result=json.loads((DATA/'result.json').read_text())
    audit=json.loads((DATA/'raw_artifact_audit.json').read_text())
    if not audit['bounded_identification_execution_complete']: raise ValueError('completed verified A required')
    inputs=launch['input_sha256']|identities|{str((DATA/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(SEEDS,launch['source_sha256'])
    verify_bindings(sources|inputs);verify_native_bindings(launch['native_sha256']);verify_extensions(launch['native_geometry_sha256'])
    return dict(source_sha256=sources,input_sha256=inputs,native_sha256=launch['native_sha256'],
        native_geometry_sha256=launch['native_geometry_sha256'],A_protocol_sha256=launch['source_sha256'][DATA_PROTOCOL])


def sensor_training_rows(launch):
    velocity,region=motion_priors(1_500_000_000,launch['A_protocol_sha256'])
    admission=json.loads((DATA/'motion_admission.json').read_text()); admission['identity']=tuple(admission['identity'])
    owner=MotionState(ArticulatedCollisionGeometry(URDF),velocity_prior=velocity,region_prior=region,admission=admission)
    controller=MotionIdentificationController(owner)
    decisions=json.loads((DATA/'motion_decisions.json').read_text()); result=json.loads((DATA/'result.json').read_text())
    records={};indices=[]
    for frame in range(result['rgbd_frames']):
        policy,depth=load_rgbd_observation(DATA,frame); now=policy['sensor_state']['decision_ns']; fast=load_fast_packet(DATA,frame)
        if frame<len(decisions):
            row=controller.observe(policy,depth,fast,now_ns=now);json_same(row,decisions[frame]['decision'])
            state=row['state']
            if row['prediction'] is not None: indices.append(frame)
        else: state=owner.observe(policy,depth,fast,now_ns=now)
        if state['terminal']: raise ValueError('A replay failed; no fit from changed state')
        if state['handoff_ready']:
            sensed,_=policy_state(owner,policy,now_ns=now)
            records[frame]=dict(state=sensed,rotation=owner._memory._rays.rotation.copy(),
                relative=owner.relative_observation(now_ns=now),time_ns=now)
    rows=[]; excluded=[]
    for frame in indices:
        before,after=records[frame],records[frame+1]; motion=after['relative']['motion']
        if motion is None or motion['rank']!=3 or motion['translation_previous_body_m'] is None:
            excluded.append(dict(frame=frame,reason='future_translation_not_fully_observed'));continue
        s,n=before['state'],after['state']; applied=n['prior']
        bx,jx=features(s['q'],s['dq'],s['velocity'],s['gyro'],s['prior'],applied)
        rotation=Rotation.from_matrix(before['rotation'].T@after['rotation']).as_rotvec()
        by=np.r_[np.asarray(motion['translation_previous_body_m'])/.03,rotation/.05,n['velocity']/.3,n['gyro']/.5]
        jy=np.column_stack(((n['q']-s['q'])/.5,n['dq']/5.))
        rows.append(dict(observation_index=frame,measured_ns=before['time_ns'],target_ns=after['time_ns'],
            body_features=bx.tolist(),body_target=by.tolist(),joint_features=jx.tolist(),joint_target=jy.tolist(),
            target_role='future_deployment_valid_depth_gyro_joints',native_label_used=False))
    return rows,excluded


def main():
    if OUTPUT.exists(): raise ValueError('fresh A-only fit output; no overwrite or refit')
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        rows,excluded=sensor_training_rows(launch)
        if len(rows)<14: raise ValueError('insufficient fully observed A transitions for fixed fit')
        model=fit_response(rows)
        write_json(OUTPUT/'training_rows.json',dict(rows=rows,excluded=excluded))
        write_json(OUTPUT/'model.json',model)
        residual_body=np.asarray([r['body_features'] for r in rows])@np.asarray(model['body_weights'])-np.asarray([r['body_target'] for r in rows])
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        verify_native_bindings(launch['native_sha256']);verify_extensions(launch['native_geometry_sha256'])
        result=dict(status='A_ONLY_SENSOR_RESPONSE_MODEL_FROZEN',training_rows=len(rows),excluded_rows=excluded,
            model_canonical_sha256=model_identity(model),model_file_sha256=digest(OUTPUT/'model.json'),
            training_rows_sha256=digest(OUTPUT/'training_rows.json'),
            in_sample_normalized_body_rmse=float(np.sqrt(np.mean(residual_body**2))),
            coefficient_count=11*12+12*7*2,native_training_labels_used=False,
            independent_validation_complete=False,navigation_qualified=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='FIT_FAILED',error=repr(error)));raise


if __name__=='__main__':main()

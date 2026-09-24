"""Raw acquisition, live force/prediction reconstruction and native-only scoring."""
import json
import numpy as np

from lewm.foot_load_sensor_development import FEET,IdealFootForceSample
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from lewm.physical_execution_development import rotation_xyzw
from lewm.simulated_body_observation_development import CALIBRATION
from lewm.support_friction_challenge_development import CONDITIONS,specification,schedule
from lewm.support_kinematics_development import FootJacobians,CausalQuietUp,predict_support_motion
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_causal_support_kinematics_development_v1 import MODES,stats
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.rgbd_shadow_motion_raw_audit_development import audit_sensors,contact_packet
from scripts.run_go2_support_friction_collection_v1 import OUTPUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources

IDENTITIES={'launch.json':'1602091d4f49713495798cb1ecd354294a164ca0a79cb003217348b6753a2980',
 'result.json':'39afad5afcc546f8017d1920b79cd524566c3c51f12a0ea0b8cc819ced38f175'}


def audit_condition(condition,result):
    directory=OUTPUT/condition; spec=specification(condition)
    if result['status']!='PHYSICAL_CHALLENGE_COMPLETE_AUDIT_REQUIRED' or result['physical_stop'] is not None:
        raise ValueError('partial challenge requires separately named partial audit')
    raw,contacts,topology,roles,cameras,relatives,geometry,sensors=audit_sensors(directory,spec,result)
    assert len(raw['timestamp_s'])==12000 and len(cameras)==226 and all(r['within1mm'] for r in sensors['depth_checks'])
    np.testing.assert_array_equal(raw['requested_command'][:750],np.zeros((750,3)))
    np.testing.assert_array_equal(raw['applied_command'],raw['post_slew_applied_command'])
    tape=read_json(directory,'command_tape.json'); assert len(tape)==225
    for i,(item,wanted) in enumerate(zip(tape,schedule(),strict=True)):
        a,b=749+i*50,799+i*50
        assert item['tick']==i and item['completed'] and item['pre_sample_index']==a and item['post_sample_index']==b
        assert all(item[k]==v for k,v in wanted.items()) and item['end_perf_counter_ns']>=item['start_perf_counter_ns']
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(wanted['requested_command'],(50,1)))
        applied=raw['applied_command'][a]+np.clip(np.array(wanted['requested_command'],np.float32)-raw['applied_command'][a],[-.25,0,-.35],[.25,0,.35])
        np.testing.assert_allclose(raw['applied_command'][a+1:b+1],np.tile(applied,(50,1)),atol=1e-7,rtol=0)
        np.testing.assert_array_equal(raw['phase'][a+1:b+1],np.full(50,wanted['phase']))
    friction=read_json(directory,'friction_checks.json');assert len(friction)==228
    assert [r['stage'] for r in friction]==['before_settle','after_settle']+['before_command']*225+['terminal']
    for r in friction:
        np.testing.assert_allclose(r['solver_friction'],spec['friction_mu'],atol=1e-7,rtol=0)
        np.testing.assert_allclose(r['cached_friction'],spec['friction_mu'],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(r['solver_ratio'],np.ones((1,28)))
    assert friction[0]['physics_steps']==0 and friction[-1]['physics_steps']==12000
    assert read_json(directory,'actuator_identity.json')['effective']==read_json(directory,'terminal_actuator_gains.json')
    setup=read_json(directory,'setup_checks.json');assert setup['setup']['velocity_and_nonfloor_setup_checks_pass']
    assert setup['support']['initial_native_support_witness_present']
    guard=dict(robot_geom_ids=friction[0]['robot_geom_ids'],foot_geom_ids=[int(k) for k in setup['feet']['native_foot_geom_to_shape']],
        ground_geom_ids=roles['physical_ground_geom_ids']); guards=[]
    for i in range(750,12000):
        pose=raw['base_pose_world'][i];R=rotation_xyzw(pose[3:]);speed=float(np.linalg.norm(raw['base_twist_world'][i,:3]))
        nonfeet=nonfoot_ground_contact_indices(contact_packet(contacts,i),**guard);inside=bool((np.abs(pose[:2])<8).all())
        assert not nonfeet and speed<=.3 and inside and pose[2]>=.15
        assert max(abs(np.arctan2(R[2,1],R[2,2])),abs(np.arcsin(np.clip(-R[2,0],-1,1))))<=.70
        guards.append(dict(sample_index=i,nonfoot_ground_contact_indices=nonfeet,base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
    assert guards==read_json(directory,'native_guard_rows.json')
    live=read_npz(directory,'live_foot_sensor.npz');slow=read_npz(directory,'ideal_sensor_samples.npz');fast=read_npz(directory,'fast_gyro_samples.npz')
    predictions=read_json(directory,'support_predictions.json');assert predictions['failure'] is None
    rows=predictions['rows'];assert len(rows)==1126
    mapping={int(k):FEET.index(v.split(':')[0]) for k,v in setup['feet']['native_foot_geom_to_shape'].items()}
    kin=FootJacobians(URDF);up_model=CausalQuietUp();window=[];cursor=0;force_error=0.;score=[];contact_speed={};nonzero_contact_sides=0
    np.testing.assert_array_equal(live['measured_ns'],np.arange(1,12001)*2_000_000)
    np.testing.assert_array_equal(live['available_ns'],live['measured_ns']);assert live['valid'].all() and not live['saturated'].any()
    for i in range(12000):
        stamp=int(live['measured_ns'][i]);q=raw['joint_position'][i];pose=raw['base_pose_world'][i];R=rotation_xyzw(pose[3:])
        k=kin.calculate(q);world=np.zeros((4,3),np.longdouble);packet=contact_packet(contacts,i);active=[]
        for c in np.flatnonzero(packet['valid_mask']):
            for side,other in [('a','b'),('b','a')]:
                g=int(packet['geom_'+side][c]);h=int(packet['geom_'+other][c])
                if g not in mapping:continue
                foot=mapping[g];f=packet['force_'+side][c];world[foot]+=f.astype(np.longdouble)
                if h in roles['physical_ground_geom_ids'] and np.linalg.norm(f)>0:
                    active.append((foot,packet['position'][c]));nonzero_contact_sides+=1
        for f in range(4):
            recovered=(R@k['rotation_body_from_foot'][f]).astype(np.longdouble)@live['force_foot_n'][i,f].astype(np.longdouble)
            force_error=max(force_error,float(np.max(np.abs(recovered-world[f]))))
        if force_error>1e-8:raise ValueError('live force conservation mismatch')
        # Contact-point speed is derived evaluation, never a prediction input.
        phase=str(int(raw['phase'][i]));contact_speed.setdefault(phase,[])
        dq=raw['joint_velocity'][i];omega=R.T@raw['base_twist_world'][i,3:];v=R.T@raw['base_twist_world'][i,:3]
        for f,point in active:
            r=k['position_body_m'][f];local=R.T@(point-pose[:3])
            centre=v+np.cross(omega,r)+k['linear_jacobian'][f]@dq
            angular=omega+k['angular_jacobian'][f]@dq
            point_world_velocity=R@(centre+np.cross(angular,local-r))
            contact_speed[phase].append(float(np.linalg.norm(point_world_velocity[:2])))
        sample=IdealFootForceSample(spec['ideal_foot_sensor_identity'],stamp,stamp,live['force_foot_n'][i],live['valid'][i],live['saturated'][i])
        window.append(sample);window=window[-11:]
        if stamp<1_300_000_000:continue
        b=(i+1)//10-1;has_body=stamp%20_000_000==0
        up=up_model.update(stamp,fast['values'][i],slow['specific_force_values'][b] if has_body else None)
        if not has_body or up is None:continue
        body=dict(identity=spec['ideal_foot_sensor_identity'],calibration_id=CALIBRATION,measured_ns=stamp,available_ns=stamp,
            q=slow['joints_values'][b,:12],dq=slow['joints_values'][b,12:],gyro=slow['gyro_values'][b],
            specific_force=slow['specific_force_values'][b],valid=True)
        predicted=predict_support_motion(kin,body,window,identity=spec['ideal_foot_sensor_identity'],up_body=up)
        predicted['rotation_gyro_anchor_from_body']=up_model.Q.tolist();assert predicted==rows[cursor];cursor+=1
        item=dict(measured_ns=stamp,phase=phase,modes={})
        for mode in MODES:
            m=predicted['modes'][mode];estimate=m['consensus_velocity_body_m_s']
            item['modes'][mode]=dict(error_m_s=float(np.linalg.norm(np.array(estimate)-v)) if estimate is not None else None,
                disagreement_m_s=m['maximum_disagreement_m_s'])
        score.append(item)
    write_json(directory/'support_native_evaluation.json',score)
    motion={}
    for name in dict.fromkeys(t['segment'] for t in tape):
        commands=[t for t in tape if t['segment']==name];a,b=commands[0]['pre_sample_index'],commands[-1]['post_sample_index']
        p=raw['base_pose_world'][a:b+1];yaw=np.unwrap([np.arctan2(rotation_xyzw(s[3:])[1,0],rotation_xyzw(s[3:])[0,0]) for s in p])
        tail=raw['base_twist_world'][b-99:b+1]
        motion[name]=dict(net_translation_m=float(np.linalg.norm(p[-1,:3]-p[0,:3])),yaw_change_rad=float(yaw[-1]-yaw[0]),
            final200ms_max_speed_m_s=float(np.linalg.norm(tail[:,:3],axis=1).max()))
    phases={}
    for phase in sorted({r['phase'] for r in score},key=int):
        selected=[r for r in score if r['phase']==phase]
        phases[phase]={mode:dict(error_m_s=stats([r['modes'][mode]['error_m_s'] for r in selected if r['modes'][mode]['error_m_s'] is not None]),
            disagreement_m_s=stats([r['modes'][mode]['disagreement_m_s'] for r in selected if r['modes'][mode]['disagreement_m_s'] is not None]),
            unavailable=sum(r['modes'][mode]['error_m_s'] is None for r in selected)) for mode in MODES}
    return dict(raw_sensor_audit=sensors,live_force_vectors=48000,maximum_force_coordinate_error_n=force_error,
        exact_support_prediction_rows=cursor,friction_checks=len(friction),motion=motion,phases=phases,
        derived_contact_tangential_speed_m_s_by_phase={k:stats(v) for k,v in contact_speed.items()},
        nonzero_foot_ground_contact_sides=nonzero_contact_sides,contact_speed_uses_native_kinematics_and_solver_contact_positions=True,
        contact_position_timing_and_physical_slip_not_independently_calibrated=True,navigation_qualified=False)


def main():
    if (OUTPUT/'raw_support_audit_launch.json').exists():raise ValueError('exclusive raw challenge audit')
    ids={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(ids)
    launch=read_json(OUTPUT,'launch.json');verify(launch);result=read_json(OUTPUT,'result.json')
    if result['absent_expected_artifacts']:raise ValueError('complete expected acquisition required')
    inputs=launch['input_sha256']|ids|{str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    sources=discover_sources(('scripts/audit_go2_support_friction_collection_v1.py',),launch['source_sha256']);verify_bindings(sources|inputs)
    write_json(OUTPUT/'raw_support_audit_launch.json',dict(source_sha256=sources,input_sha256=inputs,scope='raw acquisition and frozen support model audit; native-only scoring'))
    try:
        summaries={}
        for c in CONDITIONS:
            print('BEGIN_RAW_FRICTION_AUDIT',c,flush=True);summaries[c]=audit_condition(c,result['conditions'][c]);print('RAW_FRICTION_AUDIT_PASS',c,flush=True)
        verify_bindings(sources|inputs)
        write_json(OUTPUT/'raw_support_audit.json',dict(status='RAW_FRICTION_SUPPORT_AUDIT_PASS',conditions=summaries,
            native_evaluation_sha256={c:digest(OUTPUT/c/'support_native_evaluation.json') for c in CONDITIONS},
            model_fitting=False,physical_error_calibrated=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'raw_support_audit_failure.json',dict(status='TERMINAL_RAW_SUPPORT_AUDIT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

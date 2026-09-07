"""Independent raw acquisition, full runtime replay and native per-leg/home score."""
import json
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from lewm.room_return_pulse_development import STAGES
from lewm.inner_goal_room_return_development import InnerGoalRoomReturn
from scripts.fixed_nominal_pulse_table_development import load_fixed_table
from lewm.intent_room_return_scene_development import specification,TRIALS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from scripts.intent_return_sensor_audit_development import audit_sensors,contact_packet
from scripts.run_go2_inner_arrival_room_return_v1 import OUTPUT,PREVIOUS
from scripts.audit_go2_intent_room_return_v1 import score_hold,score_winding
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import validate_root,verify_artifacts


def verify_paired_setup(raw,cameras,baseline,baseline_cameras):
    for key in ('base_pose_world','base_twist_world','joint_position','joint_velocity','requested_command','applied_command'):
        assert len(raw[key])>=750 and len(baseline[key])>=750
        np.testing.assert_array_equal(raw[key][:750],baseline[key][:750])
    assert cameras[0]['rgb_sha256']==baseline_cameras[0]['rgb_sha256']


def audit_condition(c,result):
    directory=OUTPUT/c;spec=specification(c)
    raw,contacts,topology,roles,cameras,_,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']);assert n<=181250
    old=read_npz(PREVIOUS/c,'physics_trace.npz')
    baseline_camera=read_json(PREVIOUS/c,'camera_audit.json')
    verify_paired_setup(raw,cameras,old,baseline_camera)
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],spec['geometry']['spawn_se2_world'][:2],rtol=0,atol=.002)
    initial_yaw=Rotation.from_quat(raw['base_pose_world'][0,3:]).as_euler('xyz')[2]
    assert abs(initial_yaw-spec['geometry']['spawn_se2_world'][2])<.002
    friction=read_json(directory,'friction_checks.json')
    for f in friction:
        np.testing.assert_allclose(f['solver_friction'],spec['friction_mu'],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(f['solver_ratio'],np.ones((1,28)))
    assert friction[0]['physics_steps']==0 and friction[-1]['physics_steps']==n
    assert read_json(directory,'actuator_identity.json')['effective']==read_json(directory,'terminal_actuator_gains.json')
    rows=read_json(directory,'servo_decisions.json');tape=read_json(directory,'command_tape.json')
    assert len(rows)==result['decisions'] and len(tape)==result['command_ticks']
    model=InnerGoalRoomReturn(spec['turn_sign'],load_fixed_table());desired=[]
    reader=IntentReturnRGBDReplay(directory)
    for tick,row in enumerate(rows):
        assert row['tick']==row['observation_index']==tick
        p,d,f,now=reader.packet(tick)
        assert type(row['resource_free_bytes']) is int and row['resource_free_bytes']>=0
        if row['resource_free_bytes']<40*1024**3:model.runtime.executor.fail('STORAGE_RESERVE_STOP',now_ns=now)
        actual=model.observe(p,d,f,now_ns=now)
        serialized=json.loads(json.dumps(actual))
        assert serialized==row['decision'] and serialized['evidence']==row['evidence']
        assert row['pre_sample_index']==cameras[tick]['physical_sample_index']==749+tick*50
        if actual['terminal'] is None:
            local=actual['execution']['local_decision']
            desired.append((actual['requested_command'],local['phase'] if local else 2,'visual_feedback'))
        elif actual['terminal']=='ROOM_RETURN_FAILED':desired.extend([([0.,0.,0.],9,'terminal_zero_command_drain')]*10)
    if result['physical_stop'] is None:assert len(desired)==len(tape)
    else:
        assert len(tape)<=len(desired)
        model.finish_physical_stop(result['physical_stop'],now_ns=int(round(raw['timestamp_s'][-1]*1e9)))
    assert json.loads(json.dumps(model.snapshot()))==read_json(directory,'return_memory.json')
    for i,(item,(requested,phase,role)) in enumerate(zip(tape,desired)):
        assert item['tick']==i and item['requested_command']==requested and item['phase']==phase and item['role']==role
        a,b=item['pre_sample_index'],item['post_sample_index'];assert a==749+i*50 and a<=b<=a+50
        if item['completed']:assert b==a+50
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(requested,(b-a,1)))
        applied=raw['applied_command'][a]+np.clip(np.asarray(requested,np.float32)-raw['applied_command'][a],[-.25,0,-.35],[.25,0,.35])
        np.testing.assert_allclose(raw['applied_command'][a+1:b+1],np.tile(applied,(b-a,1)),rtol=0,atol=1e-7)
        np.testing.assert_array_equal(raw['phase'][a+1:b+1],np.full(b-a,phase))
    assert n==750+sum(t['post_sample_index']-t['pre_sample_index'] for t in tape)
    np.testing.assert_array_equal(raw['requested_command'][:750],np.zeros((750,3)))
    setup=read_json(directory,'setup_checks.json');assert setup['setup']['velocity_and_nonfloor_setup_checks_pass']
    guard=dict(robot_geom_ids=friction[0]['robot_geom_ids'],foot_geom_ids=[int(k) for k in setup['feet']['native_foot_geom_to_shape']],ground_geom_ids=roles['physical_ground_geom_ids'])
    guards=[]
    for i in range(750,n):
        bad=nonfoot_ground_contact_indices(contact_packet(contacts,i),**guard)
        speed=float(np.linalg.norm(raw['base_twist_world'][i,:3]));inside=bool((abs(raw['base_pose_world'][i,:2])<8).all())
        if result['physical_stop'] is None:assert not bad and speed<=.3 and inside
        guards.append(dict(sample_index=i,nonfoot_ground_contact_indices=bad,base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
    logged=read_json(directory,'native_guard_rows.json');assert logged==guards[:len(logged)]
    if result['physical_stop'] is None:assert logged==guards
    poses=raw['base_pose_world'];rotations=Rotation.from_quat(poses[:,3:]).as_matrix();R0=rotations[749]
    relative=(poses[:,:3]-poses[749,:3])@R0
    relative_R=np.einsum('ij,njk->nik',R0.T,rotations);yaw=np.arctan2(relative_R[:,1,0],relative_R[:,0,0])
    holds=[]
    for leg in model.snapshot()['executor']['legs']:
        if leg['status']!='LOCAL_GOAL_CANDIDATE':continue
        end=leg['finished_ns']//2_000_000-1;g=leg['goal']
        h=score_hold(relative,yaw,raw['base_twist_world'],end=end,target_xy=g['target_xy'],target_yaw=g['target_yaw_rad'])
        np.testing.assert_array_equal(raw['requested_command'][end-499:end+1],np.zeros((500,3)))
        np.testing.assert_array_equal(raw['phase'][end-499:end+1],np.full(500,4))
        holds.append(dict(leg_index=leg['leg_index'],goal=g,native_hold=h,native_winding_hold=score_winding(yaw,g,end=end)))
    home=None
    if model.terminal=='ROOM_RETURN_CANDIDATE':
        assert len(model.completed)==len(STAGES) and result['physical_stop'] is None
        home=score_hold(relative,yaw,raw['base_twist_world'],end=n-1,target_xy=[0.,0.],target_yaw=0.)
    errors=[float(np.linalg.norm(np.asarray(row['evidence']['current_pose']['position_initial_body_m'])-relative[row['pre_sample_index']]))
            for row in rows if row['evidence'] is not None and row['evidence']['current_pose'] is not None]
    return dict(raw_sensor_audit_pass=True,replayed_decisions=len(rows),raw_depth_checks=len(sensors['depth_checks']),
                all_depth_checks_within1mm=all(v['within1mm'] for v in sensors['depth_checks']),physics_samples=n,
                duration_s=float(raw['timestamp_s'][-1]),native_stop=result['physical_stop'],paired_setup_prefix_and_first_rgb_verified=True,
                terminal=model.terminal,reason=model.reason,completed_stages=len(model.completed),completed_leg_holds=holds,
                native_home_hold=home,full_room_return_success=home is not None and home['passed'] and all(h['native_hold']['passed'] and h['native_winding_hold']['passed'] for h in holds),
                maximum_visual_position_error_m=max(errors,default=None),native_final_position=relative[-1].tolist(),native_final_yaw_rad=float(yaw[-1]),
                native_path_length_m=float(np.linalg.norm(np.diff(poses[749:,:3],axis=0),axis=1).sum()),
                mechanical_energy_j=None,scripted_motion_assay=True,maze_navigation=False,goal_achieved=False)


def main():
    validate_root(OUTPUT)
    target=OUTPUT/'raw_return_audit_launch.json'
    if target.exists():raise ValueError('exclusive raw return audit')
    cv2.setNumThreads(1);launch=read_json(OUTPUT,'launch.json');verify(launch);r=read_json(OUTPUT,'result.json')
    if r['absent_expected_artifacts']:raise ValueError('partial artifacts require explicit partial audit')
    if launch['output_root']!=str(OUTPUT):raise ValueError('bound external output mismatch')
    assert launch['paired_baseline_root']==str(PREVIOUS)
    verify_artifacts(PREVIOUS,launch['paired_baseline_artifact_sha256'])
    inputs=launch['input_sha256']
    external=r['artifact_sha256']|{n:digest(OUTPUT/n) for n in ('launch.json','result.json')}
    verify_artifacts(OUTPUT,external)
    sources=discover_sources(('scripts/audit_go2_inner_arrival_room_return_v1.py',),launch['source_sha256'])
    verify_bindings(inputs|sources);write_json(target,dict(source_sha256=sources,input_sha256=inputs,output_root=str(OUTPUT),external_input_sha256=external));reports={}
    try:
        for c in TRIALS:
            reports[c]=audit_condition(c,r['conditions'][c]);write_json(OUTPUT/(c+'_return_evaluation.json'),reports[c])
            print('ROOM_RETURN_AUDIT',c,{k:v for k,v in reports[c].items() if k!='completed_leg_holds'},flush=True)
        verify(launch);verify_bindings(inputs|sources);verify_artifacts(OUTPUT,external)
        verify_artifacts(PREVIOUS,launch['paired_baseline_artifact_sha256'])
        write_json(OUTPUT/'raw_return_audit.json',dict(status='RAW_RETURN_AUDIT_PASS',conditions=reports,
                   evaluation_sha256={c:digest(OUTPUT/(c+'_return_evaluation.json')) for c in TRIALS},goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'raw_return_audit_failure.json',dict(status='TERMINAL_RAW_RETURN_AUDIT_FAILURE',reason=repr(error),completed_conditions=reports));raise


if __name__=='__main__':main()

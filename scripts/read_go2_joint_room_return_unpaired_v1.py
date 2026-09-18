"""Distinct unpaired raw/replay/native analysis; the original paired audit stays failed."""
import json
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from lewm.room_return_pulse_development import STAGES
from lewm.joint_inner_goal_room_return_development import JointInnerGoalRoomReturn
from scripts.fixed_nominal_pulse_table_development import load_fixed_table
from lewm.intent_room_return_scene_development import specification,TRIALS
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices
from scripts.intent_return_sensor_audit_development import audit_sensors,contact_packet
from scripts.run_go2_joint_room_return_v1 import OUTPUT as INPUT, PREVIOUS
from scripts.navigation_artifact_root_development import BASE,create_output
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from PIL import Image
from scripts.audit_go2_intent_room_return_v1 import score_hold,score_winding
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import validate_root,verify_artifacts


OUTPUT=BASE/'go2_joint_room_return_unpaired_readout_v1_attempt_001'
PROTOCOL='docs/go2_joint_room_return_unpaired_readout_v1_2026-09-07.md'
INPUT_IDENTITIES={
    'result.json':'a62a1de6837a758709d386eaa3aacfa8d49b043a20e477c867058cc49174a2ca',
    'launch.json':'83b1e0ad2913cc4693bfdaf3cd3f939d7452b97566dff4c4e18c332e72c16260',
    'raw_return_audit_launch.json':'aee6daaf65f75f78352c33a7398e05dd1f080823ab9b98e8e869ae01eb8818cc',
    'raw_return_audit_failure.json':'09ce73a3d590004db18ff53c95e6e52ce6f7b8d436fb1f04401e2a8276bb8042'}


def describe_setup(raw,cameras,baseline,baseline_cameras,*,directory,predecessor):
    """Record all equality outcomes; never substitute tolerances for pairing."""
    names=('base_pose_world','base_twist_world','joint_position','joint_velocity','requested_command','applied_command')
    exact={}
    for key in names:
        assert len(raw[key])>=750 and len(baseline[key])>=750
        exact[key]=bool(np.array_equal(raw[key][:750],baseline[key][:750]))
    with Image.open(directory/'rgb_0000.png') as im:a=np.array(im)
    with Image.open(predecessor/'rgb_0000.png') as im:b=np.array(im)
    assert a.shape==b.shape==(480,640,3) and a.dtype==b.dtype==np.uint8
    delta=a.astype(np.int16)-b.astype(np.int16)
    depth=read_npz(directory,'native_depth_0000.npz')['optical_depth_m']
    old_depth=read_npz(predecessor,'native_depth_0000.npz')['optical_depth_m']
    rgb_equal=cameras[0]['rgb_sha256']==baseline_cameras[0]['rgb_sha256']
    assert rgb_equal==bool(np.array_equal(a,b))
    return dict(native_prefix_channel_exact=exact,native_prefix_exact=all(exact.values()),
        first_rgb_exact=rgb_equal,paired_setup_prefix_and_first_rgb_verified=all(exact.values()) and rgb_equal,
        first_rgb_sha256=[cameras[0]['rgb_sha256'],baseline_cameras[0]['rgb_sha256']],
        different_rgb_pixels=int(np.any(delta!=0,axis=2).sum()),different_rgb_channels=int((delta!=0).sum()),
        maximum_rgb_channel_difference=int(abs(delta).max()),mean_absolute_rgb_channel_difference=float(abs(delta).mean()),
        first_native_depth_exact=bool(np.array_equal(depth,old_depth)),
        different_native_depth_pixels=int(np.count_nonzero(depth!=old_depth)),
        maximum_native_depth_difference_m=float(np.max(abs(depth.astype(float)-old_depth.astype(float)))),
        renderer_difference_cause_established=False,original_pairing_criterion_relaxed=False)


def read_condition(c,result):
    directory=INPUT/c;spec=specification(c)
    raw,contacts,topology,roles,cameras,_,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']);assert n<=181250
    old=read_npz(PREVIOUS/c,'physics_trace.npz')
    baseline_camera=read_json(PREVIOUS/c,'camera_audit.json')
    pairing=describe_setup(raw,cameras,old,baseline_camera,directory=directory,predecessor=PREVIOUS/c)
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
    model=JointInnerGoalRoomReturn(spec['turn_sign'],load_fixed_table());desired=[]
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
    timing=verify_timings(rows,tape)
    orientation_errors=[float(Rotation.from_matrix(np.asarray(row['evidence']['current_pose']['rotation_initial_body_from_current_body']).T@relative_R[row['pre_sample_index']]).magnitude())
        for row in rows if row['evidence'] is not None and row['evidence']['current_pose'] is not None]
    errors=[float(np.linalg.norm(np.asarray(row['evidence']['current_pose']['position_initial_body_m'])-relative[row['pre_sample_index']]))
            for row in rows if row['evidence'] is not None and row['evidence']['current_pose'] is not None]
    return dict(raw_sensor_audit_pass=True,replayed_decisions=len(rows),raw_depth_checks=len(sensors['depth_checks']),
                all_depth_checks_within1mm=all(v['within1mm'] for v in sensors['depth_checks']),physics_samples=n,
                duration_s=float(raw['timestamp_s'][-1]),native_stop=result['physical_stop'],pairing=pairing,
                paired_setup_prefix_and_first_rgb_verified=pairing['paired_setup_prefix_and_first_rgb_verified'],
                terminal=model.terminal,reason=model.reason,completed_stages=len(model.completed),completed_leg_holds=holds,
                native_home_hold=home,unpaired_native_full_return_success=home is not None and home['passed'] and all(h['native_hold']['passed'] and h['native_winding_hold']['passed'] for h in holds) and all(v['within1mm'] for v in sensors['depth_checks']),
                maximum_visual_position_error_m=max(errors,default=None),
                maximum_visual_orientation_error_rad=max(orientation_errors,default=None),
                available_pose_frames=len(errors),unavailable_pose_frames=len(rows)-len(errors),
                accepted_position_allocation_violations=sum(e>.02 for e in errors),
                accepted_orientation_allocation_violations=sum(e>math.radians(2) for e in orientation_errors),
                full_loop_timing=timing,native_final_position=relative[-1].tolist(),native_final_yaw_rad=float(yaw[-1]),
                native_path_length_m=float(np.linalg.norm(np.diff(poses[749:,:3],axis=0),axis=1).sum()),
                mechanical_energy_j=None,scripted_motion_assay=True,maze_navigation=False,goal_achieved=False)



def verify_timings(rows,tape):
    """Complete timing accounting, including partial and terminal drain commands."""
    consumed=0
    for row in rows:
        before,after=row['command_ticks_before'],row['command_ticks_after']
        assert type(before) is int and type(after) is int and before==consumed and before<=after<=len(tape)
        for name in ('acquisition_wall_ms','controller_wall_ms','observation_and_control_wall_ms',
                     'decision_elapsed_wall_ms','iteration_wall_ms'):
            assert type(row[name]) in (int,float) and math.isfinite(row[name]) and row[name]>=0
        command_ms=0.
        for item in tape[before:after]:
            assert type(item['wall_ms']) in (int,float) and math.isfinite(item['wall_ms']) and item['wall_ms']>=0
            command_ms+=item['wall_ms']
        assert row['observation_and_control_wall_ms']>=row['acquisition_wall_ms']+row['controller_wall_ms']
        assert row['decision_elapsed_wall_ms']>=row['observation_and_control_wall_ms']
        assert row['iteration_wall_ms']+1e-6>=row['decision_elapsed_wall_ms']+command_ms
        consumed=after
    assert consumed==len(tape)
    def summary(values):
        return dict(count=len(values),median_ms=float(np.median(values)) if values else None,
            maximum_ms=max(values,default=None),above100ms=sum(v>100 for v in values))
    return dict(acquisition=summary([r['acquisition_wall_ms'] for r in rows]),
        controller=summary([r['controller_wall_ms'] for r in rows]),
        decision=summary([r['decision_elapsed_wall_ms'] for r in rows]),
        normal_iteration=summary([r['iteration_wall_ms'] for r in rows if r['decision']['terminal'] is None]),
        terminal_iteration=summary([r['iteration_wall_ms'] for r in rows if r['decision']['terminal'] is not None]),
        physics_command_interval=summary([t['wall_ms'] for t in tape]),
        total_recorded_iteration_wall_ms=sum(r['iteration_wall_ms'] for r in rows),
        includes_friction_check_sensor_acquisition_controller_and_command_simulation=True,
        includes_startup_or_final_artifact_persistence=False,
        physics_paused_during_compute=True,real_time_qualified=False)



def metadata(name,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(data)>16*1024**2:raise ValueError('bounded metadata file required')
    if name not in ('launch.json','result.json','failure.json',*(c+'_evaluation.json' for c in TRIALS)):
        raise ValueError('explicit output metadata path required')
    # Only these six metadata files can be written; no raw copying.
    total=sum((OUTPUT/n).stat().st_size for n in
        ('launch.json','result.json','failure.json',*(c+'_evaluation.json' for c in TRIALS)) if (OUTPUT/n).is_file())
    if total+len(data)>64*1024**2:raise ValueError('whole metadata allowance exhausted')
    with (OUTPUT/name).open('xb') as stream:stream.write(data)


def main():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists():raise ValueError('exclusive distinct unpaired readout')
    cv2.setNumThreads(1)
    verify_artifacts(INPUT,INPUT_IDENTITIES)
    original=read_json(INPUT,'launch.json');verify(original)
    result=read_json(INPUT,'result.json');failed=read_json(INPUT,'raw_return_audit_failure.json')
    if (result['status']!='ROOM_RETURN_PULSE_COLLECTION_TERMINAL' or result['absent_expected_artifacts']
            or set(result['conditions'])!=set(TRIALS)
            or failed['status']!='TERMINAL_RAW_RETURN_AUDIT_FAILURE'
            or failed['completed_conditions']):
        raise ValueError('exact complete collection and preserved first-condition audit failure required')
    bindings=INPUT_IDENTITIES|result['artifact_sha256']
    verify_artifacts(INPUT,bindings)
    verify_artifacts(PREVIOUS,original['paired_baseline_artifact_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/read_go2_joint_room_return_unpaired_v1.py',
        'lewm/tests/test_joint_room_return_unpaired_development.py'),original['source_sha256'])
    definition=original|dict(source_sha256=sources)
    verify(definition)
    resources=hardware()
    if resources['memory_available_bytes']<16*1024**3 or resources['artifact_free_bytes']<40*1024**3+64*1024**2:
        raise ValueError('bounded CPU readout needs available RAM and artifact reserve')
    create_output(OUTPUT)
    metadata('launch.json',dict(input_root=str(INPUT),input_artifact_sha256=bindings,
        source_sha256=sources,protocol=PROTOCOL,hardware=resources,
        maximum_metadata_bytes=64*1024**2,original_paired_audit_remains_failed=True))
    reports={}
    try:
        for c in TRIALS:
            reports[c]=read_condition(c,result['conditions'][c])
            metadata(c+'_evaluation.json',reports[c])
            print('UNPAIRED_NATIVE_READOUT',c,{k:v for k,v in reports[c].items() if k!='completed_leg_holds'},flush=True)
        verify(definition);verify_artifacts(INPUT,bindings)
        verify_artifacts(PREVIOUS,original['paired_baseline_artifact_sha256'])
        metadata('result.json',dict(status='UNPAIRED_JOINT_ROOM_RETURN_READOUT_COMPLETE',conditions=reports,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json',*(c+'_evaluation.json' for c in TRIALS))},
            original_paired_audit_remains_failed=True,initial_rgb_pairing_established=False,
            independent_maze_navigation=False,learned_planning_benefit=False,goal_achieved=False))
        print('UNPAIRED_READOUT_TERMINAL',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        metadata('failure.json',dict(status='UNPAIRED_JOINT_READOUT_TERMINAL_FAILURE',
            reason=repr(error),completed_conditions=list(reports),
            original_paired_audit_remains_failed=True))
        raise


if __name__=='__main__':main()

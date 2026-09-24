"""Independent sensor/command/stop replay and honest matched-context accounting.

No fitting or action selection. Native state is used only for offline auditing
and target labels. Missing/unequal sibling prefixes are results, not retries.
"""
from dataclasses import asdict
import hashlib
import json
import cv2
import numpy as np
import torch
from lewm.independent_pulse_context_development import TRIALS,SUPPORTS,WARMUP_TICKS,specification,schedule,decision
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.visual_led_motion_development import VisualLedMotion
from lewm.physical_execution_development import rotation_xyzw
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices,match_native_foot_geometries
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot,initial_ground_support_witness
from lewm.pulse_timed_observation_pairing_development import pulse_window
from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from scripts.pulse_context_sensor_audit_development import audit_sensors,contact_packet,classify,read_json,read_npz
from scripts.pulse_context_setup_development import context_priors
from scripts.run_go2_independent_pulse_context_pilot_v1 import OUTPUT,PROTOCOL,RESERVE,artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import validate_root,verify_artifacts

PREFIX_SAMPLES=750+50*WARMUP_TICKS
PREFIX_FRAMES=WARMUP_TICKS+1


def fingerprint(value):
    """Typed, length-delimited exact content identity, including array dtype/shape."""
    h=hashlib.sha256()
    def add(tag,data):
        h.update(tag.encode()+str(len(data)).encode()+b':'+data)
    def visit(v):
        if isinstance(v,np.ndarray):
            if v.dtype.hasobject:raise ValueError('no object-array fingerprint')
            add('array',json.dumps([v.dtype.str,v.shape]).encode());add('bytes',v.tobytes())
        elif isinstance(v,dict):
            add('dict',str(len(v)).encode())
            for k in sorted(v):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            add(type(v).__name__,str(len(v)).encode())
            for x in v:visit(x)
        else:add(type(v).__name__,json.dumps(v,allow_nan=False).encode())
    visit(value);return h.hexdigest()


def prefix_witness(raw,contacts,reader):
    samples=len(raw['timestamp_s']);frames=len(reader.frames) if reader is not None else 0
    if samples<PREFIX_SAMPLES or frames<PREFIX_FRAMES:
        return dict(status='MISSING_DEPARTURE_PREFIX',native_samples=samples,frames=frames,sha256={})
    stamps=[reader.frames[i]['decision_ns'] for i in range(PREFIX_FRAMES)]
    if stamps!=[1_500_000_000+i*100_000_000 for i in range(PREFIX_FRAMES)]:
        raise ValueError('exact nine prefix frame clocks required')
    hashes={'native/'+k:fingerprint(v[:PREFIX_SAMPLES]) for k,v in raw.items()}
    stop=int(contacts['frame_offsets'][PREFIX_SAMPLES])
    for k,v in contacts.items():
        length=PREFIX_SAMPLES+1 if k=='frame_offsets' else PREFIX_SAMPLES if k=='frame_timestamp_s' else stop
        hashes['contact/'+k]=fingerprint(v[:length])
    for i in range(PREFIX_FRAMES):hashes['packet/'+str(i)]=fingerprint(reader.packet(i))
    return dict(status='COMPLETE_PREFIX',native_samples=PREFIX_SAMPLES,frames=PREFIX_FRAMES,sha256=hashes)


def compare_prefixes(reference,candidate):
    if reference['status']!='COMPLETE_PREFIX' or candidate['status']!='COMPLETE_PREFIX':
        return dict(status='UNAVAILABLE_PREFIX',matched=False,unequal_fields=[])
    a,b=reference['sha256'],candidate['sha256']
    unequal=sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))
    return dict(status='UNEQUAL_PREFIX' if unequal else 'EXACT_PREFIX_MATCH',matched=not unequal,unequal_fields=unequal)


def audit_commands(raw,tape,rows,result,action_index):
    n=len(raw['timestamp_s']);planned=schedule(action_index)
    assert len(tape)==result['command_ticks'] and len(rows)==result['decisions']
    assert result['completed_ticks']==sum(t['completed'] for t in tape)
    assert len(tape)<=len(planned) and len(rows)<=len(planned)+1
    assert len(tape)==sum(not r['decision']['terminal'] for r in rows)
    assert result['departure_present']==any(r['tick']==WARMUP_TICKS for r in rows)
    assert all(t['completed'] for t in tape[:-1])
    if tape and not tape[-1]['completed']:assert result['physical_stop'] is not None
    for i,item in enumerate(tape):
        assert item['tick']==i and type(item['completed']) is bool
        for k in ('phase','role','requested_command'):assert item[k]==planned[i][k]==rows[i]['decision'][k]
        a,b=item['pre_sample_index'],item['post_sample_index']
        assert a==749+50*i and type(b) is int and a<=b<=a+50 and b<n
        if item['completed']:assert b==a+50
        requested=np.asarray(item['requested_command'],np.float32)
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(requested,(b-a,1)))
        applied=raw['applied_command'][a]+np.clip(requested-raw['applied_command'][a],[-.25,0,-.35],[.25,0,.35])
        np.testing.assert_allclose(raw['applied_command'][a+1:b+1],np.tile(applied,(b-a,1)),rtol=0,atol=1e-7)
        np.testing.assert_array_equal(raw['phase'][a+1:b+1],np.full(b-a,item['phase']))
    assert n==min(n,750)+sum(t['post_sample_index']-t['pre_sample_index'] for t in tape)
    np.testing.assert_array_equal(raw['requested_command'][:min(n,750)],np.zeros((min(n,750),3)))
    if not result['setup_admitted']:assert not rows and not tape and result['physical_stop'] is not None
    if result['schedule_terminal'] is not None:
        assert result['schedule_terminal']=='FIXED_CONTEXT_PULSE_COMPLETE'
        assert result['physical_stop'] is None and result['acquisition_stop'] is None
        assert len(tape)==len(planned) and rows[-1]['decision']['terminal'] and all(t['completed'] for t in tape)
    else:assert result['physical_stop'] is not None or result['acquisition_stop'] is not None
    if result['acquisition_stop'] is not None:
        assert result['physical_stop'] is None and result['setup_admitted']
        assert result['acquisition_stop']=='STORAGE_RESERVE_STOP' or result['acquisition_stop'].startswith('PACKET_CONTRACT_STOP: ')


def audit_setup(directory,raw,contacts,topology,geometry,result,definition):
    if not result['setup_checked']:
        assert len(raw['timestamp_s'])<=750 and not result['setup_admitted']
        return None
    saved=read_json(directory,'setup_checks.json');pose=raw['base_pose_world'][749];q=raw['joint_position'][749]
    R=rotation_xyzw(pose[3:]);velocity,region=context_priors(definition,geometry,q)
    static=read_json(directory,'static_objects.json')
    feet=match_native_foot_geometries(read_json(directory,'startup_native_robot_geometry.json'),geometry,q,pose)
    check=check_setup_snapshot(velocity,region,identity=(0,0,0),measured_ns=1_500_000_000,
        position_world_m=pose[:3],rotation_world_from_initial_body=R,velocity_world_m_s=raw['base_twist_world'][749,:3],
        native_static_boxes=static,expected_nonfloor_names=tuple(b['native_name'] for b in static),geometry=geometry,joint_position=q)
    support=initial_ground_support_witness(classify(contact_packet(contacts,749),topology),
        expected_support_groups=['FL_calf','FR_calf','RL_calf','RR_calf'],ground_link_ids=sorted(topology['ground_link_ids']),
        geometry=geometry,joint_position=q,position_world_m=pose[:3],rotation_world_from_body=R)
    expected=dict(velocity_prior=asdict(velocity),region_prior=asdict(region),setup=check,support=support,feet=feet,
        sample_index=749,definition_sha256=definition,setup_region_used_for_navigation=False,
        evidence_role='EVALUATOR_ONLY_INITIAL_SETUP_NOT_FUTURE_GAIT_QUALIFICATION')
    assert json.loads(json.dumps(expected))==saved
    assert result['setup_admitted']==bool(check['velocity_and_nonfloor_setup_checks_pass'] and support['initial_native_support_witness_present'])
    return saved


def audit_stops(raw,contacts,roles,friction,setup,logged,result):
    """Recompute the first physical stop, including the superclass stop row."""
    n=len(raw['timestamp_s']);guards=[];first=None
    guard=None if not result['setup_admitted'] else dict(robot_geom_ids=friction[0]['robot_geom_ids'],
        foot_geom_ids=[int(k) for k in setup['feet']['native_foot_geom_to_shape']],ground_geom_ids=roles['physical_ground_geom_ids'])
    for i in range(n):
        pose=raw['base_pose_world'][i];R=rotation_xyzw(pose[3:])
        reason='DISALLOWED_CONTACT' if raw['physics_contact'][i] else None
        if reason is None and (pose[2]<.15 or max(abs(np.arctan2(R[2,1],R[2,2])),abs(np.arcsin(np.clip(-R[2,0],-1,1))))>.70):
            reason='BODY_STABILITY_LIMIT'
        if reason is None and guard is not None and i>=750:
            bad=nonfoot_ground_contact_indices(contact_packet(contacts,i),**guard)
            speed=float(np.linalg.norm(raw['base_twist_world'][i,:3]));inside=bool((abs(pose[:2])<8).all())
            guards.append(dict(sample_index=i,nonfoot_ground_contact_indices=bad,base_speed_m_s=speed,in_domain=inside,evaluator_only=True))
            if bad or speed>.3 or not inside:reason='CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP'
        if reason is not None:
            first=dict(sample_index=i,reason=reason);break
    assert guards==logged
    if first is not None:
        assert first['sample_index']==n-1 and result['physical_stop']==first['reason']
    elif not result['setup_admitted']:
        assert setup is not None and n==750 and result['physical_stop']=='FRESH_MISSION_INITIAL_SETUP_REJECTED'
    else:assert result['physical_stop'] is None
    return first


def target_row(window,labels):
    return {k:window[k] for k in ('condition','departure_tick','decision_ns','action_index')}|dict(target_only=True,
        targets=[dict(offset_ns=int(labels['target_offsets_ns'][i]),motion_valid=bool(labels['motion_valid'][i]),
            contact_valid=bool(labels['contact_valid'][i]),motion=labels['motion'][i].tolist() if labels['motion_valid'][i] else None,
            contact=float(labels['contact'][i]) if labels['contact_valid'][i] else None,
            image_target_valid=t['future_valid'],status=labels['accounting']['target_status'][i]) for i,t in enumerate(window['targets'])],
        accounting=labels['accounting'],label_definition=labels['label_definition'])


def audit_condition(trial,result,definition):
    directory=OUTPUT/trial;spec=specification(trial)
    assert read_json(directory,'specification.json')==spec and read_json(directory,'result.json')==result
    raw,contacts,topology,roles,cameras,_,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']);assert n<=2400
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],spec['geometry']['spawn_se2_world'][:2],rtol=0,atol=.002)
    R=rotation_xyzw(raw['base_pose_world'][0,3:]);assert abs(np.arctan2(R[1,0],R[0,0]))<.002
    friction=read_json(directory,'friction_checks.json')
    for f in friction:
        np.testing.assert_allclose(f['solver_friction'],spec['friction_mu'],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(f['solver_ratio'],np.ones((1,28)))
    assert friction[0]['stage']=='before_settle' and friction[0]['physics_steps']==0
    assert friction[-1]['stage']=='terminal' and friction[-1]['physics_steps']==n
    assert read_json(directory,'actuator_identity.json')['effective']==read_json(directory,'terminal_actuator_gains.json')
    rows=read_json(directory,'context_decisions.json');tape=read_json(directory,'command_tape.json')
    for tick,f in enumerate(friction[1:-1]):
        assert f['stage']=='before_decision' and f['tick']==tick and f['physics_steps']==750+50*tick
    assert len(rows)<=len(friction)-2<=len(rows)+1
    assert result['tracker_required_for_commands'] is False and result['native_state_used_for_commands'] is False
    assert result['navigation_qualified'] is False
    reader=IntentReturnRGBDReplay(directory) if cameras else None;tracker=VisualLedMotion('gyro',identity=(0,0,0));available=0
    for tick,row in enumerate(rows):
        assert row['tick']==row['observation_index']==tick
        assert row['pre_sample_index']==cameras[tick]['physical_sample_index']==749+50*tick
        assert type(row['resource_free_bytes']) is int and row['resource_free_bytes']>=RESERVE
        assert np.isfinite(row['observation_and_control_wall_ms']) and row['observation_and_control_wall_ms']>=0
        p,d,f,now=reader.packet(tick)
        assert json.loads(json.dumps(decision(spec['action_index'],tick,p)))==row['decision']
        shadow=tracker.observe(p,d,f,now_ns=now);assert json.loads(json.dumps(shadow))==row['shadow']
        available+=shadow.get('current_pose') is not None
    assert len(cameras)<=len(rows)+1 and len(cameras)>=len(rows)
    audit_commands(raw,tape,rows,result,spec['action_index'])
    setup=audit_setup(directory,raw,contacts,topology,geometry,result,definition)
    stop=audit_stops(raw,contacts,roles,friction,setup,read_json(directory,'native_guard_rows.json'),result)
    window=labels=None
    if result['departure_present']:
        window=dict(condition=trial,departure_tick=WARMUP_TICKS,decision_ns=2_300_000_000,action_index=spec['action_index'],
            command=spec['command'],pulse_ticks=spec['pulse_ticks'])|pulse_window(reader.frames,tape,
            departure_tick=WARMUP_TICKS,departure_ns=2_300_000_000,command=tuple(spec['command']),pulse_ticks=spec['pulse_ticks'])
        labels=target_row(window,RecordedPulseNativeTargets(raw).labels(window))
    depth_ok=all(r['within1mm'] for r in sensors['depth_checks']) if cameras else None
    return dict(trial=trial,recorded_sensor_reconstruction_pass=True,physics_samples=n,paired_frames=len(cameras),
        setup_checked=result['setup_checked'],setup_admitted=result['setup_admitted'],departure_present=result['departure_present'],
        schedule_complete=result['schedule_terminal'] is not None,physical_stop=stop,physical_stop_message=result['physical_stop'],
        acquisition_stop=result['acquisition_stop'],replayed_decisions=len(rows),shadow_pose_available_decisions=available,
        depth_checks=sensors['depth_checks'],all_depth_checks_within1mm=depth_ok,
        native_disallowed_contact_samples=int(np.count_nonzero(raw['physics_contact'])),
        native_path_length_m=float(np.linalg.norm(np.diff(raw['base_pose_world'][:,:3],axis=0),axis=1).sum()),
        target_motion_valid=sum(t['motion_valid'] for t in labels['targets']) if labels else 0,
        target_contact_positive=sum(t['contact']==1. for t in labels['targets']) if labels else 0,
        future_rgb_valid=sum(t['future_valid'] for t in window['targets']) if window else 0,
        missing_known_targets=sum(t['offset_ns']>0 and not t['contact_valid'] for t in labels['targets']) if labels else None,
        partial_setup=sensors['partial_setup'],native_wall_inventory_checked=sensors['native_wall_inventory_checked'],
        native_state_used_for_commands=False,tracker_required_for_commands=False,navigation_qualified=False),prefix_witness(raw,contacts,reader),window,labels


def main():
    validate_root(OUTPUT,must_exist=True)
    if (OUTPUT/'context_audit_launch.json').exists():raise ValueError('exclusive context audit; no retry')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    launch=read_json(OUTPUT,'launch.json');verify(launch);result=read_json(OUTPUT,'result.json')
    assert result['status']=='CONTEXT_COLLECTION_TERMINAL' and result['planned_trials']==list(TRIALS)
    assert set(result['conditions'])==set(TRIALS) and not result['absent_expected_artifacts']
    expected={c+'/'+p for c in TRIALS for p in artifacts(c,result['conditions'][c])}
    assert set(result['artifact_sha256'])==expected
    bindings=result['artifact_sha256']|{p:digest(OUTPUT/p) for p in ('launch.json','result.json')}
    verify_artifacts(OUTPUT,bindings)
    write_json(OUTPUT/'context_audit_launch.json',dict(source_sha256=launch['source_sha256'],external_artifact_sha256=bindings,
        model_training=False,navigation_qualified=False))
    reports={};prefixes={};windows=[];targets=[]
    try:
        for c in TRIALS:
            report,prefix,window,labels=audit_condition(c,result['conditions'][c],launch['source_sha256'][PROTOCOL])
            reports[c]=report;prefixes[c]=prefix
            if window is not None:windows.append(window);targets.append(labels)
            write_json(OUTPUT/(c+'_context_evaluation.json'),report)
            print('CONTEXT_AUDIT',c,{k:v for k,v in report.items() if k!='depth_checks'},flush=True)
        pairs={c:compare_prefixes(prefixes[s+'_action_0'],prefixes[c]) for s in SUPPORTS for c in TRIALS if c.startswith(s+'_action_')}
        roles={w['condition']:dict(layout_id=specification(w['condition'])['layout_id'],role='train') for w in windows}
        dataset=PulseTimedDataset(windows,targets,roles) if windows else None
        integrated=[]
        if dataset is not None:
            for i,w in enumerate(windows):
                if not w['history_ready']:continue
                sample=dataset.sample(i,{w['condition']:IntentReturnRGBDReplay(OUTPUT/w['condition'])})
                integrated.append(dict(condition=w['condition'],input_fields=sorted(sample['inputs']),
                    motion_valid=int(sample['targets']['motion_valid'].sum())))
        write_json(OUTPUT/'context_windows.json',windows);write_json(OUTPUT/'context_targets.json',targets)
        write_json(OUTPUT/'context_prefix_witnesses.json',prefixes)
        verify(launch);verify_artifacts(OUTPUT,bindings)
        write_json(OUTPUT/'context_audit.json',dict(status='CONTEXT_RECORDED_EVIDENCE_AUDITED',conditions=reports,prefix_comparisons=pairs,
            expected_trials=12,setup_admitted=sum(r['setup_admitted'] for r in reports.values()),departures=len(windows),
            schedule_completions=sum(r['schedule_complete'] for r in reports.values()),materialized_samples=integrated,
            training_action_coverage=dataset.coverage('train') if dataset else {},
            exact_prefix_siblings=sum(p['matched'] for c,p in pairs.items() if not c.endswith('_action_0')),
            contact_positive_targets=sum(r['target_contact_positive'] for r in reports.values()),
            output_sha256={p:digest(OUTPUT/p) for p in ['context_windows.json','context_targets.json','context_prefix_witnesses.json']+
                [c+'_context_evaluation.json' for c in TRIALS]},
            independent_layouts=1,evaluation_layouts=0,model_trained=False,maze_navigation=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'context_audit_failure.json',dict(status='TERMINAL_CONTEXT_AUDIT_FAILURE',reason=repr(error),completed_conditions=list(reports)));raise


if __name__=='__main__':main()

"""All-episode available evidence; insufficient depth rays remain failed measurement checks."""
import argparse
import json
import cv2
import numpy as np
from lewm.geometry_progress_pilot_development import (TRIALS, WARMUP_TICKS, HORIZON_TICKS,
    specification, assignments, schedule, decision, progress_outcome, panel_informativeness)
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.visual_led_motion_development import VisualLedMotion
from lewm.physical_execution_development import rotation_xyzw
from scripts.geometry_progress_available_sensor_evidence_development import audit_sensors, read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import audit_setup, audit_stops, fingerprint
from scripts.run_go2_geometry_progress_pilot_v1 import OUTPUT as INPUT, PROTOCOL, RESERVE, artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources


def audit_commands(raw,tape,rows,result,action):
    n=len(raw['timestamp_s']);planned=schedule(action)
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
        requested=np.asarray(item['requested_command'],np.float64)
        assert raw['requested_command'].dtype==raw['applied_command'].dtype==np.float64
        assert raw['post_slew_applied_command'].dtype==np.float64
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(requested,(b-a,1)))
        previous=raw['applied_command'][a].astype(np.float32)
        delta=np.array([.25,0.,.35],dtype=np.float32)
        applied=np.clip(requested.astype(np.float32),previous-delta,previous+delta).astype(np.float64)
        np.testing.assert_array_equal(raw['applied_command'][a+1:b+1],np.tile(applied,(b-a,1)))
        np.testing.assert_array_equal(raw['post_slew_applied_command'][a+1:b+1],np.tile(applied,(b-a,1)))
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


def native_horizons(raw, frames):
    """Every known horizon is kept, independently masking image/motion/contact."""
    start=750+50*WARMUP_TICKS-1
    if len(raw['timestamp_s'])<=start:return None
    pose=raw['base_pose_world'];R0=rotation_xyzw(pose[start,3:]);origin=pose[start,:3]
    if raw['physics_contact'][:start+1].any():raise ValueError('no post-contact departure')
    camera_by_sample={f['physical_sample_index']:i for i,f in enumerate(frames)}
    targets=[]
    for block in range(1,9):
        at=start+block*250;complete=at<len(pose)
        event=bool(raw['physics_contact'][start+1:min(at+1,len(pose))].any())
        motion=None
        if complete and not event:
            delta=R0.T@(pose[at,:3]-origin);relative=R0.T@rotation_xyzw(pose[at,3:])
            motion=[float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))]
        targets.append(dict(offset_ns=block*500_000_000,motion_valid=motion is not None,motion=motion,
            contact_valid=complete or event,contact=float(event) if complete or event else None,
            future_image_valid=at in camera_by_sample,future_observation_index=camera_by_sample.get(at),
            status='CONTACT_EVENT' if event else 'OBSERVED_ENDPOINT' if complete else 'CENSORED_EXECUTION'))
    return dict(departure_tick=WARMUP_TICKS,departure_ns=1_800_000_000,
        history_observation_indices=list(range(4)),targets=targets,target_only=True,
        label_definition='departure_body_xy_and_projected_relative_yaw;disallowed_native_contact')


def prefix_witness(raw,reader):
    n=750+50*WARMUP_TICKS
    if len(raw['timestamp_s'])<n or reader is None or len(reader.frames)<4:
        return dict(complete=False,sha256={})
    hashes={'native/'+k:fingerprint(v[:n]) for k,v in raw.items()}
    for i in range(4):hashes['packet/'+str(i)]=fingerprint(reader.packet(i))
    return dict(complete=True,sha256=hashes)


def audit_condition(trial,result,definition):
    directory=INPUT/trial;spec=specification(trial);cell=assignments()[trial]
    assert read_json(directory,'specification.json')==spec and read_json(directory,'result.json')==result
    raw,contacts,topology,roles,cameras,_,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']);assert n<=750+50*(WARMUP_TICKS+HORIZON_TICKS)
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],[0,0],rtol=0,atol=.002)
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
    reader=IntentReturnRGBDReplay(directory) if cameras else None
    tracker=VisualLedMotion('gyro',identity=(0,0,0));available=0
    for tick,row in enumerate(rows):
        assert row['tick']==row['observation_index']==tick
        assert row['pre_sample_index']==cameras[tick]['physical_sample_index']==749+50*tick
        assert type(row['resource_free_bytes']) is int and row['resource_free_bytes']>=RESERVE
        assert np.isfinite(row['observation_and_control_wall_ms']) and row['observation_and_control_wall_ms']>=0
        p,d,f,now=reader.packet(tick)
        assert json.loads(json.dumps(decision(cell['action'],tick,p)))==row['decision']
        shadow=tracker.observe(p,d,f,now_ns=now);assert json.loads(json.dumps(shadow))==row['shadow']
        available+=shadow.get('current_pose') is not None
    assert len(rows)<=len(cameras)<=len(rows)+1
    audit_commands(raw,tape,rows,result,cell['action'])
    setup=audit_setup(directory,raw,contacts,topology,geometry,result,definition)
    stop=audit_stops(raw,contacts,roles,friction,setup,read_json(directory,'native_guard_rows.json'),result)
    start=750+50*WARMUP_TICKS-1;delta=None
    if result['departure_present']:
        pose=raw['base_pose_world'];delta=(rotation_xyzw(pose[start,3:]).T@(pose[-1,:3]-pose[start,:3]))[:2].tolist()
    outcome=progress_outcome(delta,complete=result['schedule_terminal'] is not None,
        disallowed_contact=bool(raw['physics_contact'].any()),physical_stop=result['physical_stop'],
        acquisition_stop=result['acquisition_stop'])
    return dict(trial=trial,**cell,outcome=outcome,terminal_displacement_departure_body_xy_m=delta,
        raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True,
        physics_samples=n,frames=len(cameras),decisions=len(rows),setup_admitted=result['setup_admitted'],
        shadow_pose_available_decisions=available,physical_stop=stop,
        depth_checks=sensors['depth_checks'],all_depth_checks_within1mm=all(r['within1mm'] for r in sensors['depth_checks']),
        prefix=prefix_witness(raw,reader),targets=native_horizons(raw,cameras) if result['departure_present'] else None,
        observation_and_control_wall_ms=[r['observation_and_control_wall_ms'] for r in rows],
        navigation_qualified=False)


def compare_prefixes(rows):
    comparisons=[]
    for r in rows:
        siblings=[x for x in rows if (x['geometry'],x['appearance_seed'])==(r['geometry'],r['appearance_seed'])]
        reference=next(x for x in siblings if x['action']=='hold')
        a,b=reference['prefix'],r['prefix'];complete=a['complete'] and b['complete']
        unequal=sorted(k for k in set(a['sha256'])|set(b['sha256']) if a['sha256'].get(k)!=b['sha256'].get(k))
        comparisons.append(dict(trial=r['trial'],reference=reference['trial'],is_reference=r is reference,
            complete_prefixes=complete,exact_equal=bool(complete and not unequal),unequal_fields=unequal,
            used_as_exclusion=False,same_observation_counterfactual_claim=False))
    return comparisons


FAILED_READOUT=BASE/'go2_geometry_progress_command_readout_v1_attempt_001'
FAILED_IDENTITIES={'launch.json':'2ae65e7d5fe774c16a2a2a7e0becc4953681a8d96ee2f54eee4c546cf2001a91',
    'failure.json':'6cbec6405ef94b34c9ab56eba276223968514fae2f32904a91be339666d67c7b'}
OUTPUT=BASE/'go2_geometry_progress_available_evidence_v1_attempt_001'
READOUT_PROTOCOL='docs/go2_geometry_progress_available_evidence_v1_2026-09-07.md'
IDENTITIES={
    'launch.json':'265357dc47e8ceaecf7932f63441cd7020312c14bcce182e0cd07563888a2323',
    'result.json':'067a03f208cfc389e08ebba25b2c7a739dfb1b62f2001e1dc4238140cadfe6cc',
    'geometry_progress_audit_launch.json':'5e6adc64fdec375b867149292f7f61836680b909c8ce16ba919238bbcb94f42c',
    'geometry_progress_audit_failure.json':'28318e3a5510bcbb04009af3d78f8a2703f2fcd0bc9430b053b00588e2bca3d7',
}


def main():
    if not __debug__:raise ValueError('audit assertions must be enabled')
    validate_root(INPUT);validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists():raise ValueError('exclusive available-evidence readout; no retry/resume')
    cv2.setNumThreads(1);verify_artifacts(INPUT,IDENTITIES);verify_artifacts(FAILED_READOUT,FAILED_IDENTITIES)
    prior=read_json(FAILED_READOUT,'launch.json');prior_failure=read_json(FAILED_READOUT,'failure.json')
    assert prior_failure['status']=='TERMINAL_COMMAND_REPRESENTATION_READOUT_FAILURE'
    assert prior_failure['completed_conditions']==list(TRIALS[:8])
    failed_bindings=FAILED_IDENTITIES|{c+'_geometry_progress_audit.json':digest(FAILED_READOUT/(c+'_geometry_progress_audit.json')) for c in TRIALS[:8]}
    verify_artifacts(FAILED_READOUT,failed_bindings)
    launch=read_json(INPUT,'launch.json');result=read_json(INPUT,'result.json');verify(launch)
    failed=read_json(INPUT,'geometry_progress_audit_failure.json')
    assert failed['status']=='TERMINAL_RAW_AUDIT_FAILURE' and failed['completed_conditions']==[]
    assert result['status']=='GEOMETRY_PROGRESS_COLLECTION_TERMINAL'
    assert result['planned_trials']==list(TRIALS) and set(result['conditions'])==set(TRIALS)
    assert result['absent_expected_artifacts']==[] and launch['randomized_assignment']==assignments()
    expected={c+'/'+name for c in TRIALS for name in artifacts(c,result['conditions'][c])}
    assert set(result['artifact_sha256'])==expected
    bindings=IDENTITIES|result['artifact_sha256'];verify_artifacts(INPUT,bindings)
    sources=discover_sources((READOUT_PROTOCOL,'scripts/read_go2_geometry_progress_available_evidence_v1.py',
        'lewm/tests/test_geometry_progress_available_evidence_development.py'),launch['source_sha256']|prior['source_sha256'])
    definition=launch|dict(source_sha256=sources);verify(definition);create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(status='AVAILABLE_EVIDENCE_READOUT_LAUNCHED',
        input_sha256=bindings,source_sha256=sources,protocol=READOUT_PROTOCOL,failed_readout_sha256=failed_bindings,
        prior_command_readout_remains_failed=True,
        original_audit_remains_failed=True,changes='same exact command representation; insufficient native depth rays explicitly fail measurement gate',
        same_observation_counterfactual_claim=False,model_training=False))
    reports=[]
    try:
        for c in TRIALS:
            report=audit_condition(c,result['conditions'][c],launch['source_sha256'][PROTOCOL])
            write_json(OUTPUT/(c+'_geometry_progress_audit.json'),report);reports.append(report)
            print('AVAILABLE_GEOMETRY_PROGRESS_EVIDENCE',c,report['outcome'],flush=True)
        verify(definition);verify_artifacts(INPUT,bindings);verify_artifacts(FAILED_READOUT,failed_bindings)
        products=['launch.json']+[c+'_geometry_progress_audit.json' for c in TRIALS]
        write_json(OUTPUT/'result.json',dict(status='GEOMETRY_PROGRESS_AVAILABLE_EVIDENCE_READOUT_COMPLETE',
            collection_sha256={k:IDENTITIES[k] for k in ('launch.json','result.json')},
            original_failure_sha256=IDENTITIES['geometry_progress_audit_failure.json'],
            original_audit_remains_failed=True,prior_command_readout_remains_failed=True,
            prior_command_readout_failure_sha256=FAILED_IDENTITIES['failure.json'],
            measurement_gate_pass=all(r['all_depth_checks_within1mm'] for r in reports),
            output_sha256={n:digest(OUTPUT/n) for n in products},source_sha256=sources,
            planned_episodes=24,audited_episodes=len(reports),
            successful_progress=sum(r['outcome']['successful_progress'] for r in reports),
            depth_comparison_failures=[r['trial'] for r in reports if not r['all_depth_checks_within1mm']],
            panel=panel_informativeness(reports),prefix_comparisons=compare_prefixes(reports),
            conditions={r['trial']:{k:r[k] for k in ('geometry','appearance_seed','action','outcome','physics_samples','frames','decisions')} for r in reports},
            same_observation_counterfactual_claim=False,model_trained=False,goal_achieved=False,navigation_qualified=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AVAILABLE_EVIDENCE_READOUT_FAILURE',
            reason=repr(error),completed_conditions=[r['trial'] for r in reports]));raise


if __name__=='__main__':main()

"""Raw independent inventory episode audit; history-aware exact float64 requests."""
from dataclasses import asdict
import hashlib
import json
import cv2
import numpy as np
import torch
from lewm.independent_layout_collection_development import schedule,decision
WARMUP_TICKS=8
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.visual_led_motion_development import VisualLedMotion
from lewm.physical_execution_development import rotation_xyzw
from lewm.native_foot_geometry_evaluation_development import nonfoot_ground_contact_indices,match_native_foot_geometries
from lewm.setup_snapshot_evaluation_development import check_setup_snapshot,initial_ground_support_witness
from lewm.pulse_timed_observation_pairing_development import pulse_window
from lewm.recorded_pulse_native_targets_development import RecordedPulseNativeTargets
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from scripts.near_field_sensor_audit_development import audit_sensors,contact_packet,classify,read_json,read_npz
from scripts.pulse_context_setup_development import context_priors
from scripts.independent_layout_batch_development import RESERVE
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import validate_root,verify_artifacts

from scripts.audit_go2_independent_pulse_context_pilot_v1 import audit_setup,audit_stops,target_row,prefix_witness,compare_prefixes
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
def audit_commands(raw,tape,rows,result,action_index,history_kind):
    assert raw['requested_command'].dtype==np.float64
    n=len(raw['timestamp_s']);planned=schedule(action_index,history_kind)
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
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],np.tile(np.asarray(item['requested_command'],np.float64),(b-a,1)))
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

def audit_condition(directory,spec,result,definition):
    trial=spec['trial']
    assert read_json(directory,'specification.json')==spec and read_json(directory,'result.json')==result
    raw,contacts,topology,roles,cameras,_,geometry,sensors=audit_sensors(directory,spec,result)
    n=len(raw['timestamp_s']);assert n<=2400
    np.testing.assert_allclose(raw['base_pose_world'][0,:2],spec['geometry']['spawn_se2_world'][:2],rtol=0,atol=.002)
    R=rotation_xyzw(raw['base_pose_world'][0,3:]);difference=np.arctan2(R[1,0],R[0,0])-spec['geometry']['spawn_se2_world'][2]
    assert abs(np.arctan2(np.sin(difference),np.cos(difference)))<.002
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
        assert json.loads(json.dumps(decision(spec['action_index'],spec['history_kind'],tick,p)))==row['decision']
        shadow=tracker.observe(p,d,f,now_ns=now);assert json.loads(json.dumps(shadow))==row['shadow']
        available+=shadow.get('current_pose') is not None
    assert len(cameras)<=len(rows)+1 and len(cameras)>=len(rows)
    audit_commands(raw,tape,rows,result,spec['action_index'],spec['history_kind'])
    setup=audit_setup(directory,raw,contacts,topology,geometry,result,definition)
    stop=audit_stops(raw,contacts,roles,friction,setup,read_json(directory,'native_guard_rows.json'),result)
    window=labels=None
    if result['departure_present']:
        window=dict(condition=trial,departure_tick=WARMUP_TICKS,decision_ns=2_300_000_000,action_index=spec['action_index'],
            command=spec['command'],pulse_ticks=spec['pulse_ticks'])|pulse_window(reader.frames,tape,
            departure_tick=WARMUP_TICKS,departure_ns=2_300_000_000,command=tuple(spec['command']),pulse_ticks=spec['pulse_ticks'])
        labels=target_row(window,RecordedPulseNativeTargets(raw).labels(window))
    depth_ok=all(r['within1mm'] for r in sensors['depth_checks']) if cameras else None
    visibility_ok=all(r['physical_visibility']['passes_sampled_physical_visibility'] for r in sensors['depth_checks']) if cameras else None
    return dict(trial=trial,layout_id=spec['layout_id'],data_role=spec['data_role'],context_kind=spec['context_kind'],
        history_kind=spec['history_kind'],support=spec['support'],recorded_sensor_reconstruction_pass=True,physics_samples=n,paired_frames=len(cameras),
        setup_checked=result['setup_checked'],setup_admitted=result['setup_admitted'],departure_present=result['departure_present'],
        schedule_complete=result['schedule_terminal'] is not None,physical_stop=stop,physical_stop_message=result['physical_stop'],
        acquisition_stop=result['acquisition_stop'],replayed_decisions=len(rows),shadow_pose_available_decisions=available,
        depth_checks=sensors['depth_checks'],all_depth_checks_within1mm=depth_ok,physical_visibility_pass=visibility_ok,
        native_disallowed_contact_samples=int(np.count_nonzero(raw['physics_contact'])),
        native_path_length_m=float(np.linalg.norm(np.diff(raw['base_pose_world'][:,:3],axis=0),axis=1).sum()),
        target_motion_valid=sum(t['motion_valid'] for t in labels['targets']) if labels else 0,
        target_contact_positive=sum(t['contact']==1. for t in labels['targets']) if labels else 0,
        future_rgb_valid=sum(t['future_valid'] for t in window['targets']) if window else 0,
        missing_known_targets=sum(t['offset_ns']>0 and not t['contact_valid'] for t in labels['targets']) if labels else None,
        partial_setup=sensors['partial_setup'],native_wall_inventory_checked=sensors['native_wall_inventory_checked'],
        native_state_used_for_commands=False,tracker_required_for_commands=False,navigation_qualified=False),prefix_witness(raw,contacts,reader),window,labels

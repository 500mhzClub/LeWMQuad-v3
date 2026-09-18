"""Prospective raw near-field/raster audit for the same geometry-progress bank."""
import argparse
import json
import cv2
import numpy as np
from lewm.geometry_progress_near_field_development import (TRIALS,WARMUP_TICKS,HORIZON_TICKS,
    specification,assignments,decision,progress_outcome,measurement_gate)
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.visual_led_motion_development import VisualLedMotion
from lewm.physical_execution_development import rotation_xyzw
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.near_field_sensor_audit_development import audit_sensors,read_json,read_npz
from scripts.read_go2_geometry_progress_commands_v1 import (audit_commands,audit_setup,audit_stops,
    native_horizons,prefix_witness,compare_prefixes)
from scripts.run_go2_geometry_progress_height_union_v1 import OUTPUT as INPUT,PROTOCOL,RESERVE,artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.navigation_artifact_root_development import validate_root,verify_artifacts


def validate_rasters(records,cameras):
    if len(records)!=len(cameras):raise ValueError('complete per-frame raster witnesses required')
    first=None
    for record,camera in zip(records,cameras,strict=True):
        order,precision=record['order'],record['precision']
        if (record['physical_sample_index']!=camera['physical_sample_index']
                or order['order']!='floor_first' or order['roles']!=['floor','walls']
                or set(order['surfaces'])!={'floor','walls'}):
            raise ValueError('same-capture ordered floor/union-wall witness required')
        positions=np.asarray(precision['rgb_target_sample_positions'],float)
        samples=precision['rgb_target_samples']
        if (type(samples) is not int or not 1<=samples<=32 or positions.shape!=(samples,2)
                or not np.isfinite(positions).all() or (positions<0).any() or (positions>1).any()
                or not 1<=precision['subpixel_bits']<=32 or not 1<=precision['depth_target_depth_bits']<=64):
            raise ValueError('bounded native raster precision readbacks required')
        if first is not None and (order,precision)!=first:raise ValueError('raster order/precision changed during episode')
        first=(order,precision)


def audit_rasters_and_footprints(directory,spec,cameras,sensors):
    records=[read_json(directory,f'raster_{i:04d}.json') for i in range(len(cameras))]
    validate_rasters(records,cameras);scores=[]
    for i,camera in enumerate(cameras):
        native=read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m']
        score=evaluate_footprint(native,spec['geometry']['wall_boxes'],camera['world_from_optical'],render_near_m=.005)
        assert score['original_strict_score']==sensors['depth_checks'][i]['physical_visibility']
        scores.append(score)
    return scores


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
    footprints=audit_rasters_and_footprints(directory,spec,cameras,sensors)
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
        strict_physical_visibility_pass=bool(cameras and all(f['original_strict_score']['passes_sampled_physical_visibility'] for f in footprints)),
        footprint_checks=footprints,hard_measurement_failed_frames=[i for i,f in enumerate(footprints) if not f['stable_interior_metric_pass'] or f['near_occlusion_failure']],
        prefix=prefix_witness(raw,reader),targets=native_horizons(raw,cameras) if result['departure_present'] else None,
        observation_and_control_wall_ms=[r['observation_and_control_wall_ms'] for r in rows],
        navigation_qualified=False)


def main():
    if not __debug__:raise ValueError('assertions required for raw audit')
    parser=argparse.ArgumentParser();parser.add_argument('--launch-sha256',required=True);parser.add_argument('--result-sha256',required=True)
    args=parser.parse_args();validate_root(INPUT)
    for name in ('near_field_audit_launch.json','near_field_audit.json','near_field_audit_failure.json'):
        if (INPUT/name).exists():raise ValueError('exclusive prospective near-field audit; no retry/resume')
    cv2.setNumThreads(1);ids={'launch.json':args.launch_sha256,'result.json':args.result_sha256}
    verify_artifacts(INPUT,ids);launch=read_json(INPUT,'launch.json');collection=read_json(INPUT,'result.json');verify(launch)
    assert collection['status']=='GEOMETRY_PROGRESS_HEIGHT_UNION_COLLECTION_TERMINAL'
    assert collection['planned_trials']==list(TRIALS) and set(collection['conditions'])==set(TRIALS)
    assert collection['absent_expected_artifacts']==[] and launch['randomized_assignment']==assignments()
    expected={c+'/'+n for c in TRIALS for n in artifacts(c,collection['conditions'][c])}
    assert set(collection['artifact_sha256'])==expected
    bindings=ids|collection['artifact_sha256'];verify_artifacts(INPUT,bindings)
    write_json(INPUT/'near_field_audit_launch.json',dict(status='PROSPECTIVE_NEAR_FIELD_AUDIT_LAUNCHED',
        input_sha256=bindings,source_sha256=launch['source_sha256'],model_training=False))
    reports=[]
    try:
        for c in TRIALS:
            report=audit_condition(c,collection['conditions'][c],launch['source_sha256'][PROTOCOL])
            write_json(INPUT/(c+'_near_field_audit.json'),report);reports.append(report)
            print('NEAR_FIELD_PROGRESS_AUDIT',c,report['outcome'],report['hard_measurement_failed_frames'],flush=True)
        verify(launch);verify_artifacts(INPUT,bindings)
        names=['near_field_audit_launch.json']+[c+'_near_field_audit.json' for c in TRIALS]
        gate=measurement_gate(reports)
        write_json(INPUT/'near_field_audit.json',dict(status='PROSPECTIVE_GEOMETRY_PROGRESS_NEAR_FIELD_AUDIT_COMPLETE',
            collection_sha256=ids,planned_episodes=24,audited_episodes=len(reports),
            output_sha256={n:digest(INPUT/n) for n in names},gate=gate,
            prefix_comparisons=compare_prefixes(reports),
            conditions={r['trial']:{k:r[k] for k in ('geometry','appearance_seed','action','outcome','physics_samples','frames','decisions',
                'hard_measurement_failed_frames','strict_physical_visibility_pass')} for r in reports},
            model_trained=False,depth_navigation_qualified=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(INPUT/'near_field_audit_failure.json',dict(status='TERMINAL_NEAR_FIELD_AUDIT_FAILURE',
            reason=repr(error),completed_conditions=[r['trial'] for r in reports]));raise


if __name__=='__main__':main()

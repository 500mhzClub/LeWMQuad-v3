"""Complete-population readout of the fresh, independently audited joint assay.

Reads bound metadata and sensor decisions; no native collection, raw audit,
observer inference, training, controller restart or navigation qualification.
"""
import argparse
import hashlib
import json
from collections import Counter

from scripts import run_go2_joint_room_return_v1 as experiment
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.startup_source_inventory_development import discover_sources

DOCUMENT='docs/go2_joint_room_return_scientific_readout_2026-09-07.json'


def sensor_summary(rows,expected):
    if len(rows)!=expected or not rows:raise ValueError('complete actual decision population required')
    modes=Counter();bridges=[];active=None;failures=[];responses=[]
    for i,row in enumerate(rows):
        if row['tick']!=i or row['observation_index']!=i:raise ValueError('ordered actual decisions required')
        evidence=row['evidence'];decision=row['decision']
        if evidence!=decision['evidence']:raise ValueError('same saved controller input required')
        if evidence is None or evidence['current_pose'] is None:
            failures.append(i);mode='UNAVAILABLE'
        else:
            if evidence['current_pose']['frame']!=i or evidence['current_pose']['mode']!='joint':
                raise ValueError('same-frame explicit joint pose required')
            mode=evidence['continuity_evidence']['status']
        modes[mode]+=1
        if mode=='MEASURED_INCREMENT_BRIDGE':
            if active is None:active=dict(start_frame=i,end_frame=i,frames=0)
            active['end_frame']=i;active['frames']+=1
        elif active is not None:
            bridges.append(active|dict(following_frame=i,following_mode=mode));active=None
        local=decision['execution']['local_decision']
        if local and 'completed_action_response' in local['diagnostic']:
            responses.append(dict(frame=i,**local['diagnostic']['completed_action_response']))
    if active is not None:bridges.append(active|dict(following_frame=None,following_mode='END_OF_RECORDING'))
    return dict(frames=len(rows),pose_modes=dict(modes),unavailable_frames=failures,
        bridge_spans=bridges,completed_action_responses=responses,
        final_stage=rows[-1]['decision']['stage'],terminal=rows[-1]['decision']['terminal'],
        reason=rows[-1]['decision']['reason'],completed_stages=rows[-1]['decision']['completed_stages'],
        mission_pulses=rows[-1]['decision']['execution']['mission_pulses'],
        attempted_legs=rows[-1]['decision']['execution']['attempted_legs'])


def readout(collection_sha256,audit_sha256):
    output=experiment.OUTPUT
    terminal={'result.json':collection_sha256,'raw_return_audit.json':audit_sha256}
    verify_artifacts(output,terminal)
    result=read_json(output,'result.json');audit=read_json(output,'raw_return_audit.json')
    launch=read_json(output,'launch.json');verify(launch)
    if (result['status']!='ROOM_RETURN_PULSE_COLLECTION_TERMINAL' or result['absent_expected_artifacts']
            or audit['status']!='RAW_RETURN_AUDIT_PASS'):
        raise ValueError('complete collection and independent audit required')
    if set(result['conditions'])!=set(experiment.TRIALS) or set(audit['conditions'])!=set(experiment.TRIALS):
        raise ValueError('fixed three-condition population required')
    if set(audit['evaluation_sha256'])!=set(experiment.TRIALS):
        raise ValueError('all three evaluation bindings required')
    bindings=result['artifact_sha256']|terminal|{c+'_return_evaluation.json':h for c,h in audit['evaluation_sha256'].items()}
    bindings|={n:experiment.digest(output/n) for n in ('launch.json','raw_return_audit_launch.json')}
    verify_artifacts(output,bindings)
    audit_launch=read_json(output,'raw_return_audit_launch.json')
    verify_artifacts(output,audit_launch['external_input_sha256'])
    if audit_launch['external_input_sha256']['result.json']!=collection_sha256:
        raise ValueError('audit must bind this exact collection')
    verify_artifacts(experiment.PREVIOUS,launch['paired_baseline_artifact_sha256'])
    old=read_json(experiment.PREVIOUS,'raw_return_audit.json')
    reports={}
    for c in experiment.TRIALS:
        score=audit['conditions'][c];r=result['conditions'][c]
        if score!=read_json(output,c+'_return_evaluation.json') or score['replayed_decisions']!=r['decisions']:
            raise ValueError('exact independently replayed score population required')
        sensor=sensor_summary(read_json(output/c,'servo_decisions.json'),r['decisions'])
        if len(sensor['unavailable_frames'])!=score['unavailable_pose_frames']:
            raise ValueError('same accepted/unavailable score population required')
        reports[c]=dict(sensor=sensor,native_and_raw_score=score,
            historical_predecessor_score=old['conditions'][c],
            initial_setup_exactly_paired=score['paired_setup_prefix_and_first_rgb_verified'],
            controller_terminal=r['controller_terminal']['terminal'],physics_samples=r['physics_samples'],
            camera_frames=r['rgbd_frames'],command_ticks=r['command_ticks'])
    sources=discover_sources(('scripts/read_go2_joint_room_return_science_v1.py',
        'lewm/tests/test_joint_room_return_science_development.py'),launch['source_sha256'])
    verify(launch|dict(source_sha256=sources));verify_artifacts(output,bindings)
    return dict(status='JOINT_ROOM_RETURN_SCIENTIFIC_READOUT_COMPLETE',output_root=str(output),
        collection_sha256=collection_sha256,audit_sha256=audit_sha256,
        artifact_sha256=bindings,source_sha256=sources,conditions=reports,
        native_full_return_successes=sum(r['native_and_raw_score']['full_room_return_success'] for r in reports.values()),
        total_trials=len(reports),hardware=launch['hardware'],
        limitations=dict(exposed_room=True,independent_maze_layouts=0,scripted_stage_order=True,
            historical_baseline_observer_family_changed=True,isolated_rotation_fitting_effect_established=False,
            learned_world_model_selected_commands=False,online_pulse_model_adaptation=False,
            ideal_sensors=True,robot_hidden_from_camera=True,physics_paused_during_compute=True,
            real_time_qualified=False,hardware_validated=False,navigation_qualified=False,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-sha256',required=True);p.add_argument('--audit-sha256',required=True)
    a=p.parse_args();result=readout(a.collection_sha256,a.audit_sha256)
    target=experiment.ROOT/DOCUMENT
    if target.resolve()!=target:raise ValueError('exact nonsymlink summary path required')
    data=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(data)>32*1024**2:raise ValueError('bounded scientific metadata required')
    with target.open('xb') as stream:stream.write(data)
    print(result['status'],hashlib.sha256(data).hexdigest(),DOCUMENT)

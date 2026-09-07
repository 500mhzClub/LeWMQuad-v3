#!/usr/bin/env python3
"""Eighteen fresh Go2 turns: measured-gyro feedback versus nominal timed turns."""
import argparse
import contextlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.gyro_turn_assay_development import TARGETS,timed_decision,reduce_turn
from lewm.relative_gyro_turn_development import RelativeGyroTurn
from scripts.run_go2_causal_rgb_body_capture_development_v1 import probe_spec
from scripts.run_go2_multijunction_route_development_v1 import RouteSession,PhysicalStop
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.run_go2_online_choice_maze_pilot_development_v1 import digest,write_json


def trials():
    rows=[]
    for angle_index,(name,target) in enumerate(TARGETS):
        for heading_index,heading in enumerate((-.15,0.,.15)):
            case=angle_index*3+heading_index
            for method in ('timed','gyro'):
                spec=probe_spec(0)
                spec.update(scene_id=f'gyro-turn-assay-development-v1-{case:02d}-{method}',family='GYRO_TURN_ASSAY',
                    procedural_seed=2026091600+case,case_index=case,target_name=name,target_yaw_rad=target,method=method)
                spec['geometry']['spawn_se2_world']=[0.,0.,heading]
                rows.append(spec)
    return rows


def live_packet(session,output):
    index=session.capture_current(); frame=session.model_manifest[index]
    with Image.open(output/frame['rgb_file']) as image: pixels=np.asarray(image).copy()
    return index,session.observations.packet(pixels,frame['image_ns'])


def collect(spec,output):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    session=None; stop_reason=None; terminal=None; decisions=[]; tape=[]; start=None; binding=None
    try:
        session=RouteSession(spec,output); session.install_contact_identity()
        gains=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(output/'actuator_identity.json',gains)
        try: session.settle_recorded()
        except PhysicalStop as error: stop_reason=str(error)
        start=len(session.samples)-1; packet_index=session.capture_current(); binding=prefix_binding(session,packet_index)
        if stop_reason is None:
            controller=RelativeGyroTurn() if spec['method']=='gyro' else None
            try:
                for tick in range(121):
                    index,packet=live_packet(session,output)
                    if controller is None: decision=timed_decision(tick,spec['target_yaw_rad'])
                    elif tick==0: decision=controller.begin(packet,spec['target_yaw_rad'])
                    else: decision=controller.step(packet)
                    done=decision['status'].startswith(('COMPLETE','FAILED_'))
                    decisions.append({'tick':tick,'observation_index':index,'pre_sample_index':len(session.samples)-1,
                        'decision_ns':packet['sensor_state']['decision_ns'],'executed':not done,'controller':decision})
                    if done: terminal=decision['status']; break
                    if tick==120: raise ValueError('controller exceeded frozen twelve-second budget')
                    session.phase=1
                    tape.append({'phase':1,'pre_sample_index':len(session.samples)-1,'requested_command':decision['requested_command']})
                    session.command_tick(decision['requested_command'])
                for _ in range(5):
                    session.phase=2; tape.append({'phase':2,'pre_sample_index':len(session.samples)-1,'requested_command':[0.,0.,0.]})
                    session.command_tick([0.,0.,0.])
            except PhysicalStop as error: stop_reason=str(error)
        session.capture_current(); raw=session.persist(output); session.persist_observations(output)
        terminal_gains=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains!=gains['effective']: raise ValueError('actuator gain drift')
        write_json(output/'terminal_actuator_gains.json',terminal_gains)
        write_json(output/'turn_decisions.json',decisions); write_json(output/'command_tape.json',tape)
        response=reduce_turn(raw,start,decisions,spec['target_yaw_rad'],terminal,stop_reason)
        return {'scene_id':spec['scene_id'],'case_index':spec['case_index'],'method':spec['method'],
            'target_yaw_rad':spec['target_yaw_rad'],'prefix_terminal_sample_index':start,
            'prefix_binding':binding,'branch_start_observation_index':packet_index,'response':response,
            'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),'rgb_packets':len(session.packet_rows)}
    except Exception:
        if session is not None and not (output/'physics_trace.npz').exists():
            session.persist(output); session.persist_observations(output)
        raise
    finally:
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output!=ROOT/'.generated/go2_gyro_turn_assay_development_v1_attempt_001' or output.exists(): raise ValueError('fresh exact assay root required')
    if shutil.disk_usage(output.parent).free<10*1024**3: raise ValueError('less than ten GiB free')
    prior=json.loads((ROOT/'.generated/go2_online_choice_maze_pilot_development_v1_attempt_001/launch.json').read_text())
    for name,expected in (prior['source_sha256'] | prior['gait_sha256']).items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or any(s in ('sealed','sealed_test.json') or s.startswith('sealed_') for s in p.parts): raise ValueError('unsafe binding')
        if digest(ROOT/p)!=expected: raise ValueError('predecessor source drift')
    new_sources=('lewm/relative_gyro_turn_development.py','lewm/gyro_turn_assay_development.py',
        'scripts/run_go2_gyro_turn_assay_development_v1.py','docs/go2_gyro_turn_assay_development_v1_2026-09-05.md')
    sources=prior['source_sha256'] | {name:digest(ROOT/name) for name in new_sources}
    specs=trials(); output.mkdir()
    launch={'schema':'gyro_turn_assay_development.v1','trial_specs':specs,'source_sha256':sources,'gait_sha256':prior['gait_sha256'],
        'scope':'paired ideal-gyro/timed reorientation assay, not maze or hardware qualification'}
    write_json(output/'launch.json',launch); rows=[]; prefixes={}
    try:
        for spec in specs:
            directory=output/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event':'turn_started','scene_id':spec['scene_id'],'completed':len(rows),'total':18}),flush=True)
            with (directory/'process.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log): row=collect(spec,directory)
            leaves=['physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
                'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
                'turn_decisions.json','command_tape.json','process.log']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            row['artifact_sha256']={name:digest(directory/name) for name in leaves}; write_json(directory/'result.json',row)
            reference=prefixes.setdefault(spec['case_index'],row['prefix_binding'])
            for key in ('physics_arrays','history_arrays','physics_samples','timestamp_ns'):
                if reference[key]!=row['prefix_binding'][key]: raise ValueError('unmatched physical/sensor settling prefix')
            rows.append(row)
            print(json.dumps({'event':'turn_finished','completed':len(rows),'total':18,'scene_id':spec['scene_id'],'response':row['response']}),flush=True)
        for name,expected in sources.items():
            if digest(ROOT/name)!=expected: raise ValueError('assay source changed')
        result={'status':'COMPLETE','completed_trials':18,'planned_trials':18,'trials':rows,'launch_sha256':digest(output/'launch.json')}
        write_json(output/'result.json',result); print(json.dumps({'status':'COMPLETE','trials':18}),flush=True)
    except Exception as error:
        write_json(output/'result.json',{'status':'INFRASTRUCTURE_FAILURE','error':str(error),'completed_trials':len(rows),
            'planned_trials':18,'trials':rows,'launch_sha256':digest(output/'launch.json')}); raise


if __name__=='__main__': main()

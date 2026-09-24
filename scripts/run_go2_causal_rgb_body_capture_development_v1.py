#!/usr/bin/env python3
"""Live corrected-gait command probes with separated causal RGB/body observations."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.physical_execution_development import build_case,rotation_xyzw
from lewm.simulated_body_observation_development import (
    JOINT_NAMES,SCHEMAS,IdealBodySensor,BodyObservationBuffer,CAMERA_CALIBRATION,
)
from scripts.run_go2_local_control_factorial_development_v1 import FactorialSession,PhysicalStop

COMMANDS=(('zero',(0.,0.,0.)),('forward',(.2,0.,0.)),('reverse',(-.2,0.,0.)),
    ('left',(0.,0.,.3)),('right',(0.,0.,-.3)),('forward_left',(.2,0.,.3)),
    ('forward_right',(.2,0.,-.3)),('reverse_left',(-.2,0.,.3)),('reverse_right',(-.2,0.,-.3)))


def probe_spec(index):
    name,command=COMMANDS[index]
    spec=build_case('straight',1.)
    spec.update(scene_id=f'go2-causal-rgb-body-dev-v1-{name}',family='SENSOR_COMMAND_PROBE',
                procedural_seed=2026090900+index,probe_index=index,stimulus_name=name,stimulus_command=list(command))
    spec['geometry']['wall_boxes']=[{'wall_id':name,'centre_xyz':centre,'size_xyz':size,
        'yaw_rad':0.,'material_id':'NEUTRAL_WALL'} for name,centre,size in (
            ('arena_front',[2.54,0,.3],[.08,5.16,.6]),('arena_back',[-2.54,0,.3],[.08,5.16,.6]),
            ('arena_left',[0,2.54,.3],[5.,.08,.6]),('arena_right',[0,-2.54,.3],[5.,.08,.6]))]
    return spec


class ObservationSession(FactorialSession):
    def __init__(self,spec):
        self.sensor=IdealBodySensor()
        self.observations=BodyObservationBuffer((0,0,0))
        self.sensor_rows=[]
        self.packet_rows=[]
        self.image_audit=[]
        self.model_manifest=[]
        super().__init__(spec)
        lookup={int(joint.dof_start):str(joint.name) for joint in self.ctx.build.robot.joints if joint.n_dofs==1}
        self.joint_names=tuple(lookup[index] for index in self.ctx.runner._leg_dof_idx.tolist())
        if self.joint_names!=JOINT_NAMES:
            raise ValueError('native joint order differs from sensor schema')

    def _sample(self,requested,applied,timestamp_s):
        before=len(self.samples)
        try:
            return super()._sample(requested,applied,timestamp_s)
        finally:
            if len(self.samples)>before:
                row=self.samples[-1]
                ns=int(round(float(row['timestamp_s'])*1e9))
                if ns%20_000_000==0:
                    values=self.sensor.sample(measured_ns=ns,quaternion_xyzw=row['base_pose_world'][3:],
                        velocity_world=row['base_twist_world'][:3],angular_velocity_world=row['base_twist_world'][3:],
                        joint_position=row['joint_position'],joint_velocity=row['joint_velocity'],joint_names=self.joint_names)
                    self.observations.append_sensors(values,ns)
                    self.sensor_rows.append({'measured_ns':np.int64(ns),**{f'{name}_{field}':pair[i]
                        for name,pair in values.items() for i,field in enumerate(('values','valid'))}})
                if ns%100_000_000==0:
                    self.observations.append_applied_command(row['applied_command'],ns)

    def capture_observation(self,output):
        from PIL import Image
        name=f'rgb_{len(self.packet_rows):04d}'
        start=time.monotonic()
        camera=self.capture_fixed_rgb(output,name)
        pixels=np.asarray(Image.open(output/f'{name}.png'))
        ns=int(round(camera['timestamp_s']*1e9))
        packet=self.observations.packet(pixels,ns)
        state=packet['sensor_state']
        row={'image_ns':np.int64(ns),'decision_ns':np.int64(ns)}
        for schema in SCHEMAS:
            for field in ('values','valid','measured_ns','available_ns'):
                row[f'{schema.name}_{field}']=state[schema.role][schema.name][field]
        self.packet_rows.append(row)
        # Only fixed relative RGB paths and observation times enter the model manifest.
        self.model_manifest.append({'rgb_file':f'{name}.png','image_ns':ns,'decision_ns':ns})
        self.image_audit.append({'rgb_file':f'{name}.png',**camera,
            'capture_wall_time_s':time.monotonic()-start,'physical_sample_index':len(self.samples)-1})

    def persist_observations(self,output):
        for leaf,rows in (('ideal_sensor_samples.npz',self.sensor_rows),('policy_histories.npz',self.packet_rows)):
            np.savez_compressed(output/leaf,**({key:np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}))
        model={'schema':'causal_rgb_body_development.v1','camera_calibration_id':CAMERA_CALIBRATION,
            'sensor_assumption':'ideal_simulated_body_origin_50hz_zero_latency',
            'history_file':'policy_histories.npz','frames':self.model_manifest,
            'sensor_schemas':[{'name':s.name,'channels':s.channels,'units':s.units,'role':s.role,
                'calibration_id':s.calibration_id,'history_length':s.history_length,'max_age_ns':s.max_age_ns} for s in SCHEMAS]}
        (output/'policy_observations.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
        (output/'camera_audit.json').write_text(json.dumps(self.image_audit,indent=2,allow_nan=False)+'\n')


def reduce_response(arrays,command,stop_reason):
    phase=arrays['phase']
    indices=np.flatnonzero(phase==1)[-500:]
    body_velocity=np.asarray([rotation_xyzw(arrays['base_pose_world'][i,3:]).T@arrays['base_twist_world'][i,:3] for i in indices])
    body_angular=np.asarray([rotation_xyzw(arrays['base_pose_world'][i,3:]).T@arrays['base_twist_world'][i,3:] for i in indices])
    release=np.flatnonzero(phase==2)[-100:]
    stable=bool(len(release)==100 and all(np.linalg.norm(arrays['base_twist_world'][i,:2])<=.10
        and abs(arrays['base_twist_world'][i,5])<=.25 for i in release))
    mean_velocity=body_velocity.mean(axis=0) if len(indices) else None
    mean_angular=body_angular.mean(axis=0) if len(indices) else None
    return {'stop_reason':stop_reason,'completed_fixed_tape':bool(stop_reason is None and len(arrays['phase'])==3000),
        'last_excitation_second_samples':len(indices),'mean_body_velocity_mps':None if mean_velocity is None else mean_velocity.tolist(),
        'mean_body_angular_velocity_radps':None if mean_angular is None else mean_angular.tolist(),
        'mean_forward_error_mps':None if mean_velocity is None else float(mean_velocity[0]-command[0]),
        'mean_yaw_error_radps':None if mean_angular is None else float(mean_angular[2]-command[2]),
        'release_motion_window_pass':stable,'contact':bool(arrays['physics_contact'].any()),
        'final_xy_world_m':arrays['base_pose_world'][-1,:2].tolist()}


def collect(spec,output):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    session,stop_reason=None,None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=ObservationSession(spec)
        session.install_contact_identity()
        identity=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        (output/'actuator_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        try:
            session.settle_recorded()
            for tick in range(45):
                session.capture_observation(output)
                session.phase=1 if tick<30 else 2
                session.command_tick(spec['stimulus_command'] if tick<30 else [0.,0.,0.])
        except PhysicalStop as exc:
            stop_reason=str(exc)
        session.capture_observation(output)
        arrays=session.persist(output)
        session.persist_observations(output)
        terminal=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=identity['effective']:
            raise ValueError('actuator gains changed during capture')
        (output/'terminal_actuator_gains.json').write_text(json.dumps(terminal,indent=2)+'\n')
        return {'scene_id':spec['scene_id'],'stimulus':spec['stimulus_name'],
            'response':reduce_response(arrays,spec['stimulus_command'],stop_reason),
            'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),'rgb_packets':len(session.packet_rows)}
    except Exception:
        if session is not None and not (output/'physics_trace.npz').exists():
            session.persist(output)
            session.persist_observations(output)
        raise
    finally:
        try:
            if session is not None:
                session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts):
        parser.error('protected output forbidden')
    output.mkdir(exist_ok=False)
    paths=('scripts/run_go2_causal_rgb_body_capture_development_v1.py','lewm/simulated_body_observation_development.py',
        'lewm/causal_sensor_state.py','lewm/actuator_gain_development.py','lewm/physical_execution_development.py',
        'lewm/physical_semantics.py','lewm/safety/contact_attribution.py','lewm/safety/contact_hazard_ontology_v1.py',
        'scripts/run_go2_local_control_factorial_development_v1.py','lewm/local_execution_controller_development.py',
        'scripts/run_go2_contact_attributed_execution_development_v1.py','scripts/run_physical_graph_edge_handoff_qualification_v1.py',
        'lewm_genesis/lewm_genesis/rollout.py','lewm_genesis/lewm_genesis/scene_builder.py','lewm_genesis/lewm_genesis/scene_loader.py',
        'config/go2_platform_manifest.yaml','config/go2_primitive_registry.yaml',
        'docs/go2_causal_rgb_body_capture_development_v1_2026-09-05.md')
    policy=yaml.safe_load((ROOT/'config/go2_platform_manifest.yaml').read_text())['locomotion']['policy_artifact']
    gait={}
    for key,sha_key in (('path','sha256'),('cfg_path','cfg_sha256')):
        digest=hashlib.sha256((ROOT/policy[key]).read_bytes()).hexdigest()
        if digest!=policy[sha_key]: raise ValueError('gait binding mismatch')
        gait[policy[key]]=digest
    specs=[probe_spec(i) for i in range(9)]
    launch={'schema':'go2_causal_rgb_body_capture_development.v1','trial_specs':specs,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},'gait_sha256':gait,
        'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'ideal simulated sensor/command development; no learned policy, held-out or hardware evaluation'}
    (output/'launch.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    rows,status=[],'COMPLETE'
    try:
        for spec in specs:
            directory=output/spec['scene_id']
            directory.mkdir(exist_ok=False)
            print(json.dumps({'event':'trial_started','trial':spec['scene_id'],'completed':len(rows),'total':9}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect(spec,directory)
            leaves=['physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
                'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json','process.log']
            leaves.extend(frame['rgb_file'] for frame in json.loads((directory/'policy_observations.json').read_text())['frames'])
            row['artifact_sha256']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in leaves}
            (directory/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            rows.append(row)
            print(json.dumps({'event':'trial_finished','trial':spec['scene_id'],'response':row['response'],
                'rgb_packets':row['rgb_packets'],'completed':len(rows),'total':9}),flush=True)
    except Exception as exc:
        traceback.print_exc()
        status='INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),
            'completed_trials':len(rows)},indent=2)+'\n')
    report={'status':status,'planned_trials':9,'completed_trials':len(rows),'trials':rows,
        'launch_sha256':hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='trials'},indent=2))
    return 0 if status=='COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())

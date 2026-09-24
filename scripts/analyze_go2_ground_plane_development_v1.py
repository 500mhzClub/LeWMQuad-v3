#!/usr/bin/env python3
"""Body-sensor ground hypothesis against separate actual-pose and URDF references."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.causal_ground_plane_development import CausalGroundPlane,verify_robot_geometry,URDF_SHA256
from lewm.causal_sensor_state import SensorContractError
from lewm.physical_execution_development import rotation_xyzw
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.simulated_body_observation_development import JOINT_NAMES
from scripts.analyze_go2_route_turn_command_odometry_development_v1 import ROOTS
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_ground_plane_development_v1_attempt_001'
URDF=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf')
NEW_SOURCES=('lewm/causal_ground_plane_development.py','lewm/tests/test_causal_ground_plane_development.py',
    'scripts/analyze_go2_ground_plane_development_v1.py','docs/go2_ground_plane_development_v1_2026-09-05.md')


def verify_bindings(bindings):
    # The common source guard intentionally accepts only repository-relative
    # paths. Keep that guard unchanged and separately validate the one exact
    # installed robot asset; no general external-path exception is introduced.
    ordinary={k:v for k,v in bindings.items() if k!=str(URDF)}
    verify(ordinary)
    if str(URDF) in bindings:
        if bindings[str(URDF)]!=URDF_SHA256: raise ValueError('unexpected external robot identity')
        verify_robot_geometry(URDF)


def reference_feet(joints,q):
    """Generic homogeneous URDF chain, independent of the closed-form runtime FK."""
    angles=dict(zip(JOINT_NAMES,q,strict=True)); positions=[]
    for leg in ('FL','FR','RL','RR'):
        transform=np.eye(4)
        for suffix in ('hip','thigh','calf','foot'):
            node=joints[f'{leg}_{suffix}_joint']; origin=node.find('origin')
            xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' '); rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            if np.any(rpy): raise ValueError('reviewed foot ancestor has unexpected origin rotation')
            translation=np.eye(4); translation[:3,3]=xyz; transform=transform@translation
            if node.get('type')=='revolute':
                axis=np.fromstring(node.find('axis').get('xyz'),sep=' '); angle=angles[node.get('name')]
                x,y,z=axis; skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
                rotation=np.eye(4); rotation[:3,:3]=np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)
                transform=transform@rotation
        positions.append((transform@np.array([-.002,0,0,1.]))[:3])
    return np.stack(positions)


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh ground hypothesis replay required')
    previous=ROOT/'.generated/go2_route_turn_command_odometry_development_v1_attempt_001'
    inputs={str((previous/'result.json').relative_to(ROOT)):'46a25db62091b935df95d4f94ac11232a6354b9d672cdf83f0b9c004b0d79b3c',
        str((previous/'launch.json').relative_to(ROOT)):'5e0347daecd6607ac66e425f2d473c3861c6e02f6eada944ba227a6b26eca93e',str(URDF):URDF_SHA256}
    verify_bindings(inputs)
    launch=json.loads((previous/'launch.json').read_text()); inputs.update(launch['input_sha256'])
    sources=launch['source_sha256']|{p:digest(ROOT/p) for p in NEW_SOURCES}; verify_bindings(sources|inputs)
    joints={j.get('name'):j for j in ET.parse(URDF).getroot().findall('joint')}
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'scope':'candidate ground hypothesis on26 old routes/turns; separate evaluation poses; no controller use or metric qualification'})
    rows=[]; maximum_fk=0.
    try:
        for group,path in ROOTS.items():
            report=json.loads((path/'result.json').read_text())
            for member in report['trials']:
                directory=path/member['scene_id']; names=['physics_trace.npz','camera_audit.json','policy_histories.npz','policy_observations.json']
                names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
                verify({str((directory/name).relative_to(ROOT)):member['artifact_sha256'][name] for name in names})
                with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive: poses=archive['base_pose_world']
                cameras=json.loads((directory/'camera_audit.json').read_text()); hypothesis=CausalGroundPlane(); samples=[]; fault=None
                for index,camera in enumerate(cameras):
                    packet=load_route_observation(directory,index); ns=packet['image']['measured_ns']
                    if ns%100_000_000: raise ValueError('unexpected source offclock frame')
                    try:
                        state=hypothesis.begin(packet,now_ns=ns) if index==0 else hypothesis.step(packet,now_ns=ns)
                    except SensorContractError as error:
                        fault={'observation_index':index,'timestamp_ns':ns,'reason':str(error)}; break
                    q=packet['sensor_state']['sensed']['joints']['values'][-1,:12]
                    reference=reference_feet(joints,q); error=float(np.max(np.abs(reference-state['foot_sphere_centres_body_m'])))
                    maximum_fk=max(maximum_fk,error)
                    if error>1e-12: raise ValueError('runtime FK disagrees with independent URDF chain')
                    pose=poses[camera['physical_sample_index']]; up=rotation_xyzw(pose[3:]).T@np.array([0,0,1.])
                    predicted_up=np.asarray(state['up_current_body'])
                    angle=math.atan2(float(np.linalg.norm(np.cross(predicted_up,up))),float(predicted_up@up))
                    height_error=state['body_origin_height_m']-float(pose[2])
                    camera_height=state['body_origin_height_m']+predicted_up@np.array([.326,0,.043])
                    samples.append({'timestamp_ns':ns,'estimated_body_height_m':state['body_origin_height_m'],
                        'evaluation_true_body_height_m':float(pose[2]),'signed_height_error_m':height_error,
                        'up_direction_error_rad':angle,'camera_height_error_m':float(camera_height-np.asarray(camera['world_from_optical'])[2,3])})
                errors=np.array([s['signed_height_error_m'] for s in samples]); angles=np.array([s['up_direction_error_rad'] for s in samples])
                rows.append({'group':group,'scene_id':member['scene_id'],'status':'SENSOR_UNAVAILABLE' if fault else 'REPLAYED',
                    'fault':fault,'samples':samples,'mean_abs_height_error_m':float(np.abs(errors).mean()) if len(errors) else None,
                    'maximum_abs_height_error_m':float(np.abs(errors).max()) if len(errors) else None,
                    'maximum_up_error_rad':float(angles.max()) if len(angles) else None})
                print(json.dumps({'event':'ground_stream_checked','completed':len(rows),'planned':26,'status':rows[-1]['status'],
                    'height_mae_m':rows[-1]['mean_abs_height_error_m']}),flush=True)
        verify_bindings(sources|inputs)
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','attempted_streams':len(rows),'sensor_unavailable_streams':sum(r['fault'] is not None for r in rows),
            'maximum_independent_fk_error_m':maximum_fk,'trials':rows,'launch_sha256':digest(OUTPUT/'launch.json'),
            'ground_plane_qualified':False,'scope':'flat-support hypothesis errors and sensor rejection counts; not clearance, place localization or hardware calibration'})
        print(json.dumps({'status':'COMPLETE','attempted_streams':len(rows),'maximum_fk_error':maximum_fk}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'FAIL','error':str(error),'attempted_streams':len(rows),'trials':rows,
            'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()

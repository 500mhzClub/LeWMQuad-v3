#!/usr/bin/env python3
"""No-refit translation-baseline transfer to eight routes and eighteen turns."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.causal_command_odometry_development import CausalCommandOdometry
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.audit_go2_successive_choice_maze_development_v1 import body_delta
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_route_turn_command_odometry_development_v1_attempt_001'
ROOTS={'routes':ROOT/'.generated/go2_multijunction_route_development_v1_attempt_001',
    'turns':ROOT/'.generated/go2_gyro_turn_assay_development_v1_attempt_001'}
BINDINGS={'routes':('46e88ab805cd3ea5e285f03e7136070fbc95b80e3d6c55b2bd4c368f5df62801',
    'f49b871983e97e88f6f3b923f257d6ad8907585cd326e5a49e33c55aafe95b35','6f7883b69285de7b8b6cfd4e79b34c8307bf51d1c24d9d52e988ba6844c6488d'),
    'turns':('a35dfbbcd50758af0d3466e333a40e5d2f71d4cf491365803fa776e55e7ee52c',
    '7d6199d4b8f507252b0471d6fa73071ff8ba78c4cddc6ff6b262d141e5ed3c7d','30e8c2a3256efe08baa59c179954d8c874389b14a205e794481ab10991f55442')}


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh no-refit replay required')
    previous=ROOT/'.generated/go2_causal_command_odometry_development_v1_attempt_001'
    inputs={str((previous/'result.json').relative_to(ROOT)):'947efe6dcfb2143386d1aba8dec496b93b8de915fe96d36a7c8d0e4db632a64b',
        str((previous/'launch.json').relative_to(ROOT)):'fcffc4323df1811ba7099d7a3a7613ddec38a0e8acadefe57e26a9988d01c38b'}
    for group,path in ROOTS.items():
        inputs.update({str((path/name).relative_to(ROOT)):sha for name,sha in zip(('result.json','launch.json','raw_artifact_audit.json'),BINDINGS[group],strict=True)})
    verify(inputs); old_launch=json.loads((previous/'launch.json').read_text())
    sources=old_launch['source_sha256']|{p:digest(ROOT/p) for p in (
        'scripts/analyze_go2_route_turn_command_odometry_development_v1.py','docs/go2_route_turn_command_odometry_development_v1_2026-09-05.md')}
    reports={}
    for group,path in ROOTS.items():
        launch=json.loads((path/'launch.json').read_text()); sources.update(launch['source_sha256'])
        report=json.loads((path/'result.json').read_text()); audit=json.loads((path/'raw_artifact_audit.json').read_text())
        count=8 if group=='routes' else 18
        if (report['status']!='COMPLETE' or report['completed_trials']!=count or audit['status']!='PASS'
                or audit['audited_trials']!=count or audit['study_result_sha256']!=digest(path/'result.json')):
            raise ValueError('complete audited source panel required')
        reports[group]=report
    verify(sources); OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'scope':'unchanged command/gyro estimator on all eight continuous routes and18 turns; no fitting/physics/localization qualification'})
    rows=[]
    try:
        for group,path in ROOTS.items():
            for member in reports[group]['trials']:
                directory=path/member['scene_id']
                if json.loads((directory/'result.json').read_text())!=member: raise ValueError('member/result disagreement')
                names=['physics_trace.npz','camera_audit.json','policy_histories.npz','policy_observations.json']
                names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
                verify({str((directory/name).relative_to(ROOT)):member['artifact_sha256'][name] for name in names})
                with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                    raw={k:archive[k] for k in ('timestamp_s','base_pose_world','physics_contact')}
                if raw['physics_contact'].any(): raise ValueError('source contact-free claim contradicted')
                camera=json.loads((directory/'camera_audit.json').read_text())
                start=camera[0]['physical_sample_index']; origin=raw['base_pose_world'][start]
                p=load_route_observation(directory,0); initial=p['image']['measured_ns']
                estimator=CausalCommandOdometry(); estimator.begin(p,now_ns=initial)
                samples=[]; excluded=[]
                for index in range(1,len(camera)):
                    metadata=camera[index]; ns=round(float(metadata['timestamp_s'])*1e9)
                    if ns%100_000_000: excluded.append(index); continue
                    estimate=estimator.step(load_route_observation(directory,index),now_ns=ns)
                    truth=body_delta(origin,raw['base_pose_world'][metadata['physical_sample_index']])
                    predicted=np.asarray(estimate['command_integrated_position_initial_body_m'])
                    samples.append({'offset_ns':ns-initial,'xy_error_m':float(np.linalg.norm(predicted[:2]-truth[:2])),
                        'zero_translation_xy_error_m':float(np.linalg.norm(truth[:2])),
                        'command_integrated_initial_body_xyz_m':predicted.tolist(),'evaluation_true_initial_body_xyz_m':truth.tolist()})
                if not samples: raise ValueError('no observed replay interval')
                rows.append({'group':group,'scene_id':member['scene_id'],'method_or_motif':member.get('method',member.get('motif')),
                    'samples':samples,'excluded_offclock_images':excluded,'endpoint':samples[-1],
                    'maximum_observed_xy_error_m':max(s['xy_error_m'] for s in samples)})
                print(json.dumps({'event':'route_turn_odometry_replayed','completed':len(rows),'planned':26,'scene_id':member['scene_id'],
                    'duration_s':samples[-1]['offset_ns']/1e9,'endpoint_error_m':samples[-1]['xy_error_m']}),flush=True)
        verify(sources|inputs)
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','replayed_trials':len(rows),'trial_results':rows,
            'launch_sha256':digest(OUTPUT/'launch.json'),'metric_translation_qualified':False,
            'scope':'descriptive no-refit replay on26 previously audited development routes/turns; not new independent mazes or a visual localization method'})
        print(json.dumps({'status':'COMPLETE','replayed_trials':len(rows)}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'FAIL','error':str(error),'replayed_trials':len(rows),
            'trial_results':rows,'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()

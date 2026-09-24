#!/usr/bin/env python3
"""Fixed no-fitting odometry-baseline replay on the completed 144-trial panel."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.causal_command_odometry_development import CausalCommandOdometry
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.audit_go2_successive_choice_maze_development_v1 import body_delta
from scripts.run_go2_successive_choice_maze_development_v1 import OUTPUT as PHYSICAL,digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_causal_command_odometry_development_v1_attempt_001'
NEW_SOURCES=('lewm/causal_command_odometry_development.py','lewm/tests/test_causal_command_odometry_development.py',
    'scripts/analyze_go2_causal_command_odometry_development_v1.py','docs/go2_causal_command_odometry_development_v1_2026-09-05.md')


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh no-fitting replay required')
    inputs={str((PHYSICAL/name).relative_to(ROOT)):sha for name,sha in {
        'launch.json':'e9a4bd281e631f06e01a134c3d68b969613d4599bf0554299a55d34bc20f7bf5',
        'result.json':'bd97ff363da74beebcb4a4770e9182b6554c791167a5a360d8535155d2bf8c7f',
        'raw_artifact_audit_clock_boundary_v2.json':'659c7d552ef49e0e9328a341da60fb97f2160639c33ba8fc7495783d4dba6d1f',
        'full_audit_source_dependency_witness_clock_boundary_v2.json':'f91c7537ad8897fdf835545a05fe557d73acc56c4fe568ad8da8c73767e28398'}.items()}
    verify(inputs)
    witness=json.loads((PHYSICAL/'full_audit_source_dependency_witness_clock_boundary_v2.json').read_text())
    physical_launch=json.loads((PHYSICAL/'launch.json').read_text())
    sources=physical_launch['source_sha256']|witness['source_sha256']|{p:digest(ROOT/p) for p in NEW_SOURCES}
    sources['scripts/audit_go2_successive_choice_maze_development_v1.py']=digest(ROOT/'scripts/audit_go2_successive_choice_maze_development_v1.py')
    verify(sources)
    report=json.loads((PHYSICAL/'result.json').read_text()); audit=json.loads((PHYSICAL/'raw_artifact_audit_clock_boundary_v2.json').read_text())
    if report['status']!='COMPLETE' or audit['status']!='PASS' or audit['audited_trials']!=144: raise ValueError('completed raw-audited physical panel required')
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'source_sha256':sources,'input_sha256':inputs,
        'scope':'descriptive command-plus-gyro translation baseline; no physics, fitting, selection or localization qualification'})
    rows=[]
    try:
        for member in report['trials']:
            directory=PHYSICAL/member['scene_id']
            if digest(directory/'result.json')!=member['result_sha256']: raise ValueError('physical member changed')
            names=['physics_trace.npz','camera_audit.json','command_tape.json','policy_histories.npz','policy_observations.json']
            names.extend(f'rgb_{i:04d}.png' for i in range(member['rgb_packets']))
            verify({str((directory/name).relative_to(ROOT)):member['artifact_sha256'][name] for name in names})
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                raw={k:archive[k] for k in ('timestamp_s','base_pose_world','physics_contact')}
            camera=json.loads((directory/'camera_audit.json').read_text()); tape=json.loads((directory/'command_tape.json').read_text())
            start=member['prefix_terminal_sample_index']; initial=member['branch_start_observation_index']
            control=[e for e in tape if e['stage']=='control']; end=control[-1]['post_sample_index'] if control else start
            origin=raw['base_pose_world'][start]; start_ns=round(float(raw['timestamp_s'][start])*1e9)
            tracker=CausalCommandOdometry(); samples=[]; excluded=[]
            tracker.begin(load_route_observation(directory,initial),now_ns=start_ns)
            for index in range(initial+1,len(camera)):
                metadata=camera[index]; physical=metadata['physical_sample_index']; ns=round(float(metadata['timestamp_s'])*1e9)
                if raw['physics_contact'][physical] or ns%100_000_000:
                    excluded.append({'image_index':index,'offset_ns':ns-start_ns,'contact':bool(raw['physics_contact'][physical]),'offclock':bool(ns%100_000_000)}); continue
                state=tracker.step(load_route_observation(directory,index),now_ns=ns)
                truth=body_delta(origin,raw['base_pose_world'][physical]); predicted=np.asarray(state['command_integrated_position_initial_body_m'])
                samples.append({'offset_ns':ns-start_ns,'stage':'control' if physical<=end else 'release',
                    'command_integrated_initial_body_xyz_m':predicted.tolist(),'evaluation_true_initial_body_xyz_m':truth.tolist(),
                    'xy_error_m':float(np.linalg.norm(predicted[:2]-truth[:2])),
                    'zero_translation_xy_error_m':float(np.linalg.norm(truth[:2])),
                    'xyz_error_m':float(np.linalg.norm(predicted-truth))})
            control_samples=[s for s in samples if s['stage']=='control']
            rows.append({'scene_id':member['scene_id'],'layout_id':member['layout_id'],'method':member['method'],
                'native_contact':member['metrics']['any_contact'],'prefix_available':member['branchable'],
                'samples':samples,'excluded_terminal_images':excluded,
                'last_observed_precontact_control_sample':control_samples[-1] if control_samples else None,
                'four_second_control_sample':next((s for s in control_samples if s['offset_ns']==4_000_000_000),None)})
            print(json.dumps({'event':'odometry_stream_replayed','completed':len(rows),'planned':144,'scene_id':member['scene_id'],'samples':len(samples)}),flush=True)
        verify(sources|inputs)
        summaries=[]
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['method']==method]
            for endpoint in ('last_observed_precontact_control_sample','four_second_control_sample'):
                points=[r[endpoint] for r in selected if r[endpoint] is not None]
                errors=np.asarray([p['xy_error_m'] for p in points]); zero=np.asarray([p['zero_translation_xy_error_m'] for p in points])
                summaries.append({'method':method,'endpoint':endpoint,'observed_trials':len(points),'missing_trials':len(selected)-len(points),
                    'xy_error_mean_m':float(errors.mean()) if len(errors) else None,
                    'xy_error_p95_m':float(np.quantile(errors,.95)) if len(errors) else None,
                    'xy_error_max_m':float(errors.max()) if len(errors) else None,
                    'zero_translation_xy_error_mean_m':float(zero.mean()) if len(zero) else None,
                    'observed_duration_mean_s':float(np.mean([p['offset_ns']/1e9 for p in points])) if points else None})
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','replayed_trials':len(rows),'trial_results':rows,'summaries':summaries,
            'launch_sha256':digest(OUTPUT/'launch.json'),'metric_translation_qualified':False,
            'limitations':['Commands are not measured velocity; ideal gyro has no calibrated hardware bias model.',
                'Reported endpoints exclude contact/offclock terminal images; fixed-four-second censoring is explicit.',
                'Methods visit different states; baseline-error differences are not causal method effects.',
                'No thresholds, corrections or noise covariances are fit to these development traces.']})
        print(json.dumps({'status':'COMPLETE','replayed_trials':len(rows),'summaries':summaries}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'FAIL','error':str(error),'replayed_trials':len(rows),
            'trial_results':rows,'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()

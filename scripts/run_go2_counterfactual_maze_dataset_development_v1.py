#!/usr/bin/env python3
"""Fixed fresh-prefix counterfactual corpus; no snapshot loading or learned selection."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.counterfactual_maze_development import ACTIONS,branch_spec,corpus,horizon_labels
from lewm.local_execution_controller_development import evaluate_edge
from scripts.run_go2_multijunction_route_development_v1 import RouteSession,PhysicalStop


def array_binding(values):
    result={}
    for name,value in values.items():
        value=np.ascontiguousarray(value)
        metadata=json.dumps({'shape':value.shape,'dtype':value.dtype.str},sort_keys=True).encode()
        result[name]=hashlib.sha256(metadata+b'\n'+value.tobytes()).hexdigest()
    return result


def prefix_binding(session,packet_index):
    raw={key:np.stack([row[key] for row in session.samples]) for key in session.samples[0]}
    return {'physics_arrays':array_binding(raw),'history_arrays':array_binding(session.packet_rows[packet_index]),
        'rgb_pixels_sha256':session.image_audit[packet_index]['rgb_sha256'],
        'physics_samples':len(session.samples),'timestamp_ns':int(round(session.samples[-1]['timestamp_s']*1e9))}


def collect(spec,output):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    session,stop_reason,prefix=None,None,None
    start_index,start_packet,binding=None,None,None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=RouteSession(spec,output)
        session.install_contact_identity()
        identity=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        (output/'actuator_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        try:
            session.settle_recorded()
        except PhysicalStop as exc:
            prefix=evaluate_edge(spec,session.edge_arrays(),stop_reason=str(exc),crossing=None)
            prefix.update(edge_index=0,geometry=spec['geometry'],controller_terminal_reason=None,
                          terminal_global_sample_index=len(session.samples)-1)
        else:
            prefix=session.run_edge(spec)
        start_packet=session.capture_current()
        start_index=len(session.samples)-1
        binding=prefix_binding(session,start_packet)
        branchable=bool(prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing']
                        and prefix['checks']['no_disallowed_contact'])
        tape=[]
        if branchable:
            session.edge_index=1
            try:
                for tick in range(45):
                    command=spec['branch_command'] if tick<40 else [0.,0.,0.]
                    tape.append({'tick':tick,'pre_sample_index':len(session.samples)-1,
                        'timestamp_s':float(session.samples[-1]['timestamp_s']),'requested_command':command})
                    session.phase=1 if tick<40 else 2
                    session.command_tick(command)
            except PhysicalStop as exc:
                stop_reason=str(exc)
        else:
            stop_reason=prefix['stop_reason'] or 'PREFIX_NOT_CROSSED'
        session.capture_current()
        raw=session.persist(output)
        session.persist_observations(output)
        (output/'prefix_decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
        (output/'branch_tape.json').write_text(json.dumps(tape,indent=2,allow_nan=False)+'\n')
        terminal=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=identity['effective']: raise ValueError('actuator gain drift')
        (output/'terminal_actuator_gains.json').write_text(json.dumps(terminal,indent=2)+'\n')
        labels=horizon_labels(raw,start_index) if branchable else []
        label_artifact={'branchable':branchable,'prefix_terminal_sample_index':start_index,
            'branch_start_observation_index':start_packet,'prefix_result':prefix,'stop_reason':stop_reason,
            'planned_command':spec['branch_command'],'planned_branch_ticks':40,'release_ticks':5,'horizon_labels':labels}
        (output/'outcome_labels.json').write_text(json.dumps(label_artifact,indent=2,allow_nan=False)+'\n')
        return {'scene_id':spec['scene_id'],'layout_id':spec['layout_id'],'data_role':spec['data_role'],
            'action_name':spec['action_name'],'action_index':spec['action_index'],'branchable':branchable,
            'stop_reason':stop_reason,'prefix_binding':binding,'horizon_labels':labels,
            'branch_start_observation_index':start_packet,'prefix_terminal_sample_index':start_index,
            'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),'rgb_packets':len(session.packet_rows)}
    except Exception:
        if session is not None and not (output/'physics_trace.npz').exists():
            session.persist(output)
            session.persist_observations(output)
            (output/'prefix_decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
        raise
    finally:
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts): parser.error('protected output forbidden')
    if shutil.disk_usage(output.parent).free<10*1024**3: parser.error('less than 10 GiB free; collection not started')
    layouts=corpus()
    specs=[branch_spec(layout,i) for layout in layouts for i in range(len(ACTIONS))]
    output.mkdir(exist_ok=False)
    source=('scripts/run_go2_counterfactual_maze_dataset_development_v1.py','lewm/counterfactual_maze_development.py',
        'scripts/run_go2_multijunction_route_development_v1.py','lewm/multijunction_routes_development.py',
        'scripts/run_go2_causal_rgb_body_capture_development_v1.py','lewm/simulated_body_observation_development.py',
        'lewm/causal_sensor_state.py','lewm/actuator_gain_development.py','lewm/physical_execution_development.py',
        'lewm/physical_semantics.py','lewm/safety/contact_attribution.py','lewm/safety/contact_hazard_ontology_v1.py',
        'scripts/run_go2_local_control_factorial_development_v1.py','lewm/local_execution_controller_development.py',
        'scripts/run_go2_contact_attributed_execution_development_v1.py','scripts/run_physical_graph_edge_handoff_qualification_v1.py',
        'lewm_genesis/lewm_genesis/rollout.py','lewm_genesis/lewm_genesis/scene_builder.py','lewm_genesis/lewm_genesis/scene_loader.py',
        'config/go2_platform_manifest.yaml','config/go2_primitive_registry.yaml',
        'docs/go2_counterfactual_maze_dataset_development_v1_2026-09-05.md')
    policy=yaml.safe_load((ROOT/'config/go2_platform_manifest.yaml').read_text())['locomotion']['policy_artifact']
    gait={}
    for key,sha_key in (('path','sha256'),('cfg_path','cfg_sha256')):
        value=hashlib.sha256((ROOT/policy[key]).read_bytes()).hexdigest()
        if value!=policy[sha_key]: raise ValueError('gait binding mismatch')
        gait[policy[key]]=value
    launch={'schema':'go2_counterfactual_maze_dataset_development.v1','layout_specs':layouts,'trial_specs':specs,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source},'gait_sha256':gait,
        'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'scene-separated development counterfactual corpus; no model fitting or final held-out evaluation'}
    (output/'launch.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    rows,bindings,status=[],{},'COMPLETE'
    try:
        for spec in specs:
            directory=output/spec['scene_id']
            directory.mkdir(exist_ok=False)
            print(json.dumps({'event':'branch_started','trial':spec['scene_id'],'role':spec['data_role'],'completed':len(rows),'total':120}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect(spec,directory)
            leaves=['physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
                'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
                'prefix_decisions.json','branch_tape.json','outcome_labels.json','process.log']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            row['artifact_sha256']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in leaves}
            (directory/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            rows.append(row)
            if spec['layout_id'] in bindings:
                if bindings[spec['layout_id']]!=row['prefix_binding']: raise ValueError('fresh prefix replay mismatch')
            else: bindings[spec['layout_id']]=row['prefix_binding']
            print(json.dumps({'event':'branch_finished','trial':row['scene_id'],'branchable':row['branchable'],
                'stop_reason':row['stop_reason'],'rgb_packets':row['rgb_packets'],'completed':len(rows),'total':120}),flush=True)
    except Exception as exc:
        traceback.print_exc()
        status='INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),
            'completed_trials':len(rows)},indent=2)+'\n')
    report={'status':status,'planned_trials':120,'completed_trials':len(rows),'trials':rows,
        'branchable_trials':sum(r['branchable'] for r in rows),'contact_stops':sum(r['stop_reason']=='DISALLOWED_CONTACT' for r in rows),
        'launch_sha256':hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
    return 0 if status=='COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())

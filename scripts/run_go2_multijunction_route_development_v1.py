#!/usr/bin/env python3
"""Fresh multi-junction oracle routes with causal RGB/body capture and true continuation."""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.local_execution_controller_development import evaluate_edge
from lewm.multijunction_routes_development import MOTIFS,WIDTHS,route_spec
from scripts.run_go2_causal_rgb_body_capture_development_v1 import ObservationSession,PhysicalStop


class RouteSession(ObservationSession):
    def __init__(self,spec,output):
        self.output=output
        super().__init__(spec)

    def capture_current(self):
        ns=int(round(float(self.samples[-1]['timestamp_s'])*1e9))
        if not self.model_manifest or self.model_manifest[-1]['image_ns']!=ns:
            self.capture_observation(self.output)
        return len(self.model_manifest)-1

    def command_tick(self,requested):
        self.capture_current()
        return super().command_tick(requested)

    def persist_observations(self,output):
        super().persist_observations(output)
        path=output/'policy_observations.json'
        manifest=json.loads(path.read_text())
        manifest['schema']='causal_rgb_body_routes_development.v1'
        path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


def collect(spec,output):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    session,edges,route_stop=None,[],None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=RouteSession(spec,output)
        session.install_contact_identity()
        identity=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        (output/'actuator_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        try:
            session.settle_recorded()
        except PhysicalStop as exc:
            row=evaluate_edge(spec,session.edge_arrays(),stop_reason=str(exc),crossing=None)
            row.update(edge_index=0,geometry=copy.deepcopy(session.geometry),controller_terminal_reason=None,
                       terminal_global_sample_index=len(session.samples)-1)
            edges.append(row)
            route_stop=str(exc)
        else:
            for index,geometry in enumerate(spec['route_geometries']):
                session.edge_index=index
                session.geometry=copy.deepcopy(geometry)
                row=session.run_edge(spec)
                row['terminal_observation_index']=session.capture_current()
                edges.append(row)
                if row['stop_reason'] is not None:
                    route_stop=row['stop_reason']
                    break
                if not row['checks']['sustained_correct_crossing']:
                    route_stop='EDGE_NOT_CROSSED'
                    break
        session.capture_current()
        session.persist(output)
        session.persist_observations(output)
        (output/'decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
        terminal=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=identity['effective']: raise ValueError('actuator gains changed')
        (output/'terminal_actuator_gains.json').write_text(json.dumps(terminal,indent=2)+'\n')
        all_crossed=len(edges)==len(spec['route_geometries']) and all(e['checks']['sustained_correct_crossing'] and e['checks']['no_disallowed_contact'] for e in edges)
        return {'scene_id':spec['scene_id'],'arm':'baseline','case_index':spec['case_index'],'motif':spec['motif'],
            'status':'SUCCESS' if all_crossed and edges[-1]['status']=='SUCCESS' else 'TASK_FAILURE',
            'route_stop':route_stop,'planned_edges':len(spec['route_geometries']),'edges':edges,
            'all_contact_free_crossings':all_crossed,'all_usable_arrivals':all_crossed and all(e['status']=='SUCCESS' for e in edges),
            'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),'rgb_packets':len(session.packet_rows),
            'first_disallowed_contact':next((e for e in session.contact_events if e['disallowed_contacts']),None)}
    except Exception:
        if session is not None and not (output/'physics_trace.npz').exists():
            session.persist(output)
            session.persist_observations(output)
            (output/'decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
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
    if any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts):
        parser.error('protected output forbidden')
    output.mkdir(exist_ok=False)
    source=('scripts/run_go2_multijunction_route_development_v1.py','lewm/multijunction_routes_development.py',
        'scripts/run_go2_causal_rgb_body_capture_development_v1.py','lewm/simulated_body_observation_development.py',
        'lewm/causal_sensor_state.py','lewm/actuator_gain_development.py','lewm/physical_execution_development.py',
        'lewm/physical_semantics.py','lewm/safety/contact_attribution.py','lewm/safety/contact_hazard_ontology_v1.py',
        'scripts/run_go2_local_control_factorial_development_v1.py','lewm/local_execution_controller_development.py',
        'scripts/run_go2_contact_attributed_execution_development_v1.py','scripts/run_physical_graph_edge_handoff_qualification_v1.py',
        'lewm_genesis/lewm_genesis/rollout.py','lewm_genesis/lewm_genesis/scene_builder.py','lewm_genesis/lewm_genesis/scene_loader.py',
        'config/go2_platform_manifest.yaml','config/go2_primitive_registry.yaml',
        'docs/go2_multijunction_route_development_v1_2026-09-05.md')
    policy=yaml.safe_load((ROOT/'config/go2_platform_manifest.yaml').read_text())['locomotion']['policy_artifact']
    gait={}
    for key,sha_key in (('path','sha256'),('cfg_path','cfg_sha256')):
        value=hashlib.sha256((ROOT/policy[key]).read_bytes()).hexdigest()
        if value!=policy[sha_key]: raise ValueError('gait identity mismatch')
        gait[policy[key]]=value
    specs=[route_spec(m,w) for m in MOTIFS for w in WIDTHS]
    launch={'schema':'go2_multijunction_route_development.v1','trial_specs':specs,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source},'gait_sha256':gait,
        'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'oracle-route development with causal observations; no visual/JEPA or held-out evaluation'}
    (output/'launch.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    rows,status=[],'COMPLETE'
    try:
        for spec in specs:
            directory=output/spec['scene_id']
            directory.mkdir(exist_ok=False)
            print(json.dumps({'event':'route_started','trial':spec['scene_id'],'completed':len(rows),'total':8}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect(spec,directory)
            leaves=['physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
                'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json',
                'terminal_actuator_gains.json','decisions.json','process.log']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            row['artifact_sha256']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in leaves}
            (directory/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            rows.append(row)
            print(json.dumps({'event':'route_finished','trial':row['scene_id'],'status':row['status'],
                'route_stop':row['route_stop'],'edges':[e['status'] for e in row['edges']],
                'rgb_packets':row['rgb_packets'],'completed':len(rows),'total':8}),flush=True)
    except Exception as exc:
        traceback.print_exc()
        status='INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),
            'completed_trials':len(rows)},indent=2)+'\n')
    report={'status':status,'planned_trials':8,'completed_trials':len(rows),'trials':rows,
        'task_successes':sum(r['status']=='SUCCESS' for r in rows),
        'launch_sha256':hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
    return 0 if status=='COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())

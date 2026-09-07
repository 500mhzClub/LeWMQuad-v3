#!/usr/bin/env python3
"""Sixteen fresh paired actuator-gain trials with an unchanged local controller."""
import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0, str(path))

from lewm.actuator_gain_development import GAIN_ARMS, as_array, configure_gains, read_gains
from lewm.local_execution_controller_development import continuation_geometry, evaluate_edge, trial_spec
from lewm.physical_execution_development import KINDS, WIDTHS
from scripts.run_go2_local_control_factorial_development_v1 import FactorialSession, PhysicalStop


def gain_spec(kind, width, arm):
    if arm not in GAIN_ARMS:
        raise ValueError('unknown gain arm')
    spec = trial_spec(kind,width,'baseline')
    spec.update(scene_id=f'go2-actuator-gain-pair-dev-v1-{kind}-width-{int(width*100):03d}-{arm}',
                procedural_seed=2026090800+spec['case_index'], gain_arm=arm)
    return spec


def initial_state(session):
    robot = session.ctx.build.robot
    indices = session.ctx.runner._leg_dof_idx.tolist()
    return {key:as_array(value).tolist() for key,value in {
        'position':robot.get_pos(), 'quaternion_wxyz':robot.get_quat(),
        'velocity':robot.get_vel(), 'angular_velocity':robot.get_ang(),
        'joint_position':robot.get_dofs_position(indices),
        'joint_velocity':robot.get_dofs_velocity(indices)}.items()}


def collect_trial(spec, output):
    from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
    session, edges, images = None, [], {}
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session = FactorialSession(spec)
        session.install_contact_identity()
        pre_state = initial_state(session)
        identity = configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                                   session.ctx.policy.env_cfg, spec['gain_arm'])
        identity['initial_state_before_intervention'] = pre_state
        (output/'actuator_identity.json').write_text(json.dumps(identity,indent=2,allow_nan=False)+'\n')
        try:
            session.settle_recorded()
        except PhysicalStop as exc:
            result = evaluate_edge(spec,session.edge_arrays(),stop_reason=str(exc),crossing=None)
            result.update(edge_index=0,geometry=copy.deepcopy(session.geometry),controller_terminal_reason=None,
                          terminal_global_sample_index=len(session.samples)-1)
            edges.append(result)
        else:
            images['initial'] = session.capture_fixed_rgb(output,'initial_rgb')
            for index in range(2):
                session.edge_index = index
                if index == 1:
                    session.geometry = continuation_geometry(spec['geometry'])
                result = session.run_edge(spec)
                edges.append(result)
                images[f'edge{index}'] = session.capture_fixed_rgb(output,f'edge{index}_rgb')
                if result['stop_reason'] is not None:
                    break
        images['final'] = session.capture_fixed_rgb(output,'final_rgb')
        session.persist(output)
        (output/'decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
        final_gains = read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        (output/'terminal_actuator_gains.json').write_text(json.dumps(final_gains,indent=2,allow_nan=False)+'\n')
        if final_gains != identity['effective']:
            raise ValueError('actuator gains changed during execution')
        crossings = len(edges)==2 and all(e['checks']['sustained_correct_crossing'] and e['checks']['no_disallowed_contact'] for e in edges)
        return {'scene_id':spec['scene_id'],'arm':'baseline','gain_arm':spec['gain_arm'],'case_index':spec['case_index'],
            'status':'SUCCESS' if crossings and edges[-1]['status']=='SUCCESS' else 'TASK_FAILURE',
            'edges':edges,'images':images,'two_contact_free_crossings':crossings,
            'two_usable_arrivals':len(edges)==2 and all(e['status']=='SUCCESS' for e in edges),
            'physics_samples':len(session.samples),
            'first_disallowed_contact':next((e for e in session.contact_events if e['disallowed_contacts']),None)}
    except Exception:
        if session is not None and not (output/'physics_trace.npz').exists():
            session.persist(output)
            (output/'decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
        raise
    finally:
        try:
            if session is not None:
                session.ctx.build.scene.destroy()
        finally:
            shutdown_genesis()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output = parser.parse_args().output_dir.absolute()
    if any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts):
        parser.error('protected output forbidden')
    output.mkdir(exist_ok=False)
    specs = [gain_spec(kind,width,arm) for kind in KINDS for width in WIDTHS for arm in GAIN_ARMS]
    source = ('scripts/run_go2_actuator_gain_pair_development_v1.py','lewm/actuator_gain_development.py',
        'scripts/run_go2_local_control_factorial_development_v1.py','lewm/local_execution_controller_development.py',
        'scripts/run_go2_contact_attributed_execution_development_v1.py','lewm/physical_execution_development.py',
        'lewm/physical_semantics.py','lewm/safety/contact_attribution.py','lewm/safety/contact_hazard_ontology_v1.py',
        'scripts/run_physical_graph_edge_handoff_qualification_v1.py','lewm_genesis/lewm_genesis/rollout.py',
        'lewm_genesis/lewm_genesis/scene_builder.py','lewm_genesis/lewm_genesis/scene_loader.py',
        'config/go2_platform_manifest.yaml','config/go2_primitive_registry.yaml',
        'docs/go2_actuator_gain_pair_development_v1_2026-09-05.md')
    policy = yaml.safe_load((ROOT/'config/go2_platform_manifest.yaml').read_text())['locomotion']['policy_artifact']
    gait = {}
    for key,sha_key in (('path','sha256'),('cfg_path','cfg_sha256')):
        value = hashlib.sha256((ROOT/policy[key]).read_bytes()).hexdigest()
        if value != policy[sha_key]:
            raise ValueError('gait artifact binding mismatch')
        gait[policy[key]] = value
    launch = {'schema':'go2_actuator_gain_pair_development.v1','trial_specs':specs,
        'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source},
        'gait_sha256':gait,'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'development single actuator-gain intervention; no model training, held-out or hardware use'}
    (output/'launch.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    rows, status = [], 'COMPLETE'
    try:
        for spec in specs:
            directory = output/spec['scene_id']
            directory.mkdir(exist_ok=False)
            print(json.dumps({'event':'trial_started','trial':spec['scene_id'],'completed':len(rows),'total':16}),flush=True)
            start = time.monotonic()
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row = collect_trial(spec,directory)
            leaves = ('physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','decisions.json',
                'actuator_identity.json','terminal_actuator_gains.json','initial_rgb.png','edge0_rgb.png','edge1_rgb.png','final_rgb.png','process.log')
            row['artifact_sha256'] = {leaf:hashlib.sha256((directory/leaf).read_bytes()).hexdigest() for leaf in leaves if (directory/leaf).exists()}
            (directory/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            rows.append(row)
            print(json.dumps({'event':'trial_finished','trial':spec['scene_id'],'status':row['status'],
                'edges':[e['status'] for e in row['edges']],'two_crossings':row['two_contact_free_crossings'],
                'elapsed_s':round(time.monotonic()-start,3),'completed':len(rows),'total':16}),flush=True)
    except Exception as exc:
        traceback.print_exc()
        status = 'INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}',
            'traceback':traceback.format_exc(),'completed_trials':len(rows)},indent=2)+'\n')
    report = {'status':status,'planned_trials':16,'completed_trials':len(rows),'trials':rows,
        'by_gain_arm':{arm:{'trials':sum(r['gain_arm']==arm for r in rows),
            'task_successes':sum(r['gain_arm']==arm and r['status']=='SUCCESS' for r in rows),
            'two_crossings':sum(r['gain_arm']==arm and r['two_contact_free_crossings'] for r in rows)} for arm in GAIN_ARMS},
        'launch_sha256':hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='trials'},indent=2))
    return 0 if status=='COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())

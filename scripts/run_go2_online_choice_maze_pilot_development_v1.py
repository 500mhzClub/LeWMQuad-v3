#!/usr/bin/env python3
"""Fresh online local selection: three fixed methods, eight layouts, three intents."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(path))
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.counterfactual_maze_development import ACTIONS,horizon_labels
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.local_execution_controller_development import evaluate_edge
from lewm.online_choice_maze_pilot_development import trials,realized_cost
from lewm.online_local_choice_development import OnlineLocalChoice,CHECKPOINTS,SEEDS
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.run_go2_multijunction_route_development_v1 import RouteSession,PhysicalStop


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def collect(spec,output,template):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    session=None; stop_reason=None; selection=None; prefix=None; tape=[]
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=RouteSession(spec,output); session.install_contact_identity()
        actuator=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(output/'actuator_identity.json',actuator)
        try: session.settle_recorded()
        except PhysicalStop as error:
            prefix=evaluate_edge(spec,session.edge_arrays(),stop_reason=str(error),crossing=None)
            prefix.update(edge_index=0,geometry=spec['geometry'],controller_terminal_reason=None,
                terminal_global_sample_index=len(session.samples)-1)
        else: prefix=session.run_edge(spec)
        start_packet=session.capture_current(); start_index=len(session.samples)-1
        binding=prefix_binding(session,start_packet)
        branchable=bool(prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing'] and prefix['checks']['no_disallowed_contact'])
        if branchable:
            # Actual newly captured RGB + live causal buffer, not a reference
            # sibling image, dataset packet, model label or simulator pose.
            image_name=session.model_manifest[start_packet]['rgb_file']
            with Image.open(output/image_name) as image: pixels=np.asarray(image).copy()
            now_ns=session.model_manifest[start_packet]['image_ns']
            packet=session.observations.packet(pixels,now_ns)
            policy=OnlineLocalChoice(spec['method'],template.models,template.bindings)
            policy.begin_episode(packet['sensor_state']['identity'])
            selection=policy.select(packet,spec['intent_xy_body_start_m'],now_ns=now_ns)
            write_json(output/'online_selection.json',selection)
            session.edge_index=1
            try:
                for tick,command in enumerate(selection['requested_command_tape']):
                    tape.append({'tick':tick,'pre_sample_index':len(session.samples)-1,
                        'timestamp_s':float(session.samples[-1]['timestamp_s']),'requested_command':command})
                    session.phase=1 if tick<40 else 2
                    session.command_tick(command)
            except PhysicalStop as error: stop_reason=str(error)
        else:
            stop_reason=prefix['stop_reason'] or 'PREFIX_NOT_CROSSED'
            write_json(output/'online_selection.json',None)
        session.capture_current(); raw=session.persist(output); session.persist_observations(output)
        write_json(output/'prefix_decisions.json',session.decisions); write_json(output/'branch_tape.json',tape)
        terminal=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=actuator['effective']: raise ValueError('actuator gain drift')
        write_json(output/'terminal_actuator_gains.json',terminal)
        labels=horizon_labels(raw,start_index) if branchable else []
        action_index=selection['selected_action_index'] if selection else None
        action_name=ACTIONS[action_index][0] if selection else 'unselected'
        command=list(ACTIONS[action_index][1]) if selection else None
        write_json(output/'outcome_labels.json',{'branchable':branchable,'prefix_terminal_sample_index':start_index,
            'branch_start_observation_index':start_packet,'prefix_result':prefix,'stop_reason':stop_reason,
            'planned_command':command,'planned_branch_ticks':40,'release_ticks':5,'horizon_labels':labels})
        utility=realized_cost(labels,prefix_available=branchable,stop_reason=stop_reason,intent_xy=spec['intent_xy_body_start_m'])
        return {'scene_id':spec['scene_id'],'layout_id':spec['layout_id'],'data_role':spec['data_role'],
            'method':spec['method'],'intent_name':spec['intent_name'],'intent_xy_body_start_m':spec['intent_xy_body_start_m'],
            'action_index':action_index,'action_name':action_name,'branchable':branchable,'stop_reason':stop_reason,
            'prefix_binding':binding,'horizon_labels':labels,'branch_start_observation_index':start_packet,
            'prefix_terminal_sample_index':start_index,'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),
            'rgb_packets':len(session.packet_rows),'utility':utility,
            'selected_stop':action_index==0 if branchable else None,
            'capture_wall_time_s':session.image_audit[start_packet]['capture_wall_time_s'],
            'adapter_ms':selection['adapter_ms'] if selection else None}
    except Exception:
        if session is not None and not (output/'physics_trace.npz').exists():
            session.persist(output); session.persist_observations(output)
            if not (output/'prefix_decisions.json').exists(): write_json(output/'prefix_decisions.json',session.decisions)
        raise
    finally:
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def paired_reduction(rows):
    layout_rows=[]
    for layout in sorted({r['layout_id'] for r in rows}):
        entry={'layout_id':layout,'methods':{}}
        for method in ('always_stop','supervised_rollout','jepa'):
            selected=[r for r in rows if r['layout_id']==layout and r['method']==method]
            if len(selected)!=3 or {r['intent_name'] for r in selected}!={'forward','left','right'}: raise ValueError('incomplete paired pilot population')
            entry['methods'][method]={'realized_cost':float(np.mean([r['utility']['cost'] for r in selected])),
                'contact_stop_fraction':float(np.mean([r['stop_reason']=='DISALLOWED_CONTACT' for r in selected])),
                'prefix_failure_fraction':float(np.mean([not r['branchable'] for r in selected])),
                'selected_stop_fraction':float(np.mean([r['selected_stop'] is True for r in selected]))}
        layout_rows.append(entry)
    comparisons={}
    for a,b in (('supervised_rollout','always_stop'),('jepa','always_stop'),('jepa','supervised_rollout')):
        delta=np.array([r['methods'][a]['realized_cost']-r['methods'][b]['realized_cost'] for r in layout_rows])
        rng=np.random.default_rng(2026091499)
        bootstrap=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
        comparisons[f'{a}_minus_{b}']={'mean_cost_delta':float(delta.mean()),'per_layout':delta.tolist(),
            'descriptive_layout_bootstrap_95_percentile':np.quantile(bootstrap,[.025,.975]).tolist()}
    return {'layouts':layout_rows,'paired_cost_comparisons':comparisons}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output!=ROOT/'.generated/go2_online_choice_maze_pilot_development_v1_attempt_001' or output.exists(): raise ValueError('fresh exact pilot output required')
    if shutil.disk_usage(output.parent).free<10*1024**3: raise ValueError('less than ten GiB free')
    replay=ROOT/'.generated/go2_online_local_choice_replay_development_v1_attempt_001'
    replay_result=json.loads((replay/'result.json').read_text()); replay_launch=json.loads((replay/'launch.json').read_text())
    if replay_result['status']!='PASS' or replay_result['replayed_choices']!=72 or digest(replay/'launch.json')!=replay_result['launch_sha256']:
        raise ValueError('adapter replay prerequisite')
    prior=json.loads((ROOT/'.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001/launch.json').read_text())
    learning=json.loads((ROOT/'.generated/go2_rgb_body_learning_comparison_development_v1_attempt_001/launch.json').read_text())
    for bindings in (prior['source_sha256'] | prior['gait_sha256'],learning['source_sha256'],replay_launch['source_sha256']):
        for name,expected in bindings.items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or any(part in ('sealed','sealed_test.json') or part.startswith('sealed_') for part in p.parts): raise ValueError('invalid prerequisite source path')
            if digest(ROOT/p)!=expected: raise ValueError('prerequisite source changed')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    # Load before resetting physics seeds; selection itself consumes no random draws.
    templates={method:OnlineLocalChoice.from_completed_study(method) for method in ('always_stop','supervised_rollout','jepa')}
    specs=trials()
    new_sources=('scripts/run_go2_online_choice_maze_pilot_development_v1.py','lewm/online_choice_maze_pilot_development.py',
        'docs/go2_online_choice_maze_pilot_development_v1_2026-09-05.md')
    sources=prior['source_sha256'] | learning['source_sha256'] | replay_launch['source_sha256'] | {p:digest(ROOT/p) for p in new_sources}
    launch={'schema':'online_choice_maze_pilot_development.v1','trial_specs':specs,'source_sha256':sources,
        'gait_sha256':prior['gait_sha256'],'model_checkpoint_sha256':{f'{seed}-{method}':sha for method,hashes in CHECKPOINTS.items() for seed,sha in zip(SEEDS,hashes,strict=True)},
        'adapter_replay_result_sha256':digest(replay/'result.json'),'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'fresh online conditional local choices; teacher initialization, ideal sensors and privileged simulator emergency stops; no autonomous maze or hardware claim'}
    output.mkdir(); write_json(output/'launch.json',launch); rows=[]; references={}
    try:
        for spec in specs:
            directory=output/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event':'trial_started','scene_id':spec['scene_id'],'completed':len(rows),'total':72}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect(spec,directory,templates[spec['method']])
            leaves=['actuator_identity.json','terminal_actuator_gains.json','physics_trace.npz','native_contacts.npz','contact_topology.json','contact_events.json','process.log',
                'ideal_sensor_samples.npz','policy_histories.npz','policy_observations.json','camera_audit.json',
                'prefix_decisions.json','branch_tape.json','outcome_labels.json']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            row['artifact_sha256']={name:digest(directory/name) for name in leaves}
            row['online_selection_sha256']=digest(directory/'online_selection.json')
            write_json(directory/'result.json',row)
            reference=references.setdefault(spec['layout_id'],(directory,row))
            match=compare_prefix(reference[0],reference[1],directory,row)
            match={k:v for k,v in match.items() if not k.startswith('canonical_model_context_')}
            match.update(reference_scene_id=reference[1]['scene_id'],selection_used_own_actual_packet=True)
            rows.append(row | {'result_sha256':digest(directory/'result.json'),'prefix_match':match})
            print(json.dumps({'event':'trial_finished','scene_id':spec['scene_id'],'action':row['action_name'],
                'stop_reason':row['stop_reason'],'cost':row['utility']['cost'],'completed':len(rows),'total':72}),flush=True)
        for name,expected in sources.items():
            if digest(ROOT/name)!=expected: raise ValueError('pilot source changed during execution')
        result={'status':'COMPLETE','planned_trials':72,'completed_trials':len(rows),'trials':rows,
            'reduction':paired_reduction(rows),'launch_sha256':digest(output/'launch.json')}
        write_json(output/'result.json',result); print(json.dumps({'status':'COMPLETE','trials':len(rows),'reduction':result['reduction']},indent=2))
    except Exception as error:
        write_json(output/'result.json',{'status':'INFRASTRUCTURE_FAILURE','error':str(error),'completed_trials':len(rows),'trials':rows,
            'planned_trials':72,'launch_sha256':digest(output/'launch.json')})
        raise


if __name__=='__main__': main()

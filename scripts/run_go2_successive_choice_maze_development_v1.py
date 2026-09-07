#!/usr/bin/env python3
"""One fixed fresh 144-trial successive RGB/body choice panel; no training."""
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
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.causal_sensor_state import SensorContractError
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.local_execution_controller_development import evaluate_edge
from lewm.online_temporal_choice_development import METHODS,OnlineTemporalChoice
from lewm.successive_choice_maze_development import trials,execute_control
from lewm.successive_choice_metrics_development import reduce_trial,paired_reduction
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.run_go2_multijunction_route_development_v1 import RouteSession,PhysicalStop

OUTPUT=ROOT/'.generated/go2_successive_choice_maze_development_v1_attempt_001'
PROTOCOL='docs/go2_successive_choice_maze_development_v1_2026-09-05.md'
NEW_SOURCES=(str(Path(__file__).relative_to(ROOT)),PROTOCOL,'lewm/successive_choice_maze_development.py',
    'lewm/successive_choice_metrics_development.py','lewm/tests/test_successive_choice_maze_development.py',
    'lewm/tests/test_successive_choice_metrics_development.py','lewm/tests/test_successive_choice_live_session_development.py')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def verify(bindings):
    for name,sha in bindings.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts or any(s in ('sealed','sealed_test.json') or s.startswith('sealed_') for s in p.parts):
            raise ValueError('nonprotected explicit relative binding required')
        path=ROOT/p
        if path.resolve()!=path or digest(path)!=sha: raise ValueError('source/evidence identity changed: '+name)


class LiveSession(RouteSession):
    def __init__(self,spec,output,policy):
        self.policy=policy; self.last_observed_ns=None; self.observed_indices=[]; self.observe_enabled=True
        super().__init__(spec,output)

    def observe_current(self):
        index=self.capture_current(); row=self.model_manifest[index]; ns=row['image_ns']
        if ns==self.last_observed_ns: return index
        with Image.open(self.output/row['rgb_file']) as image: pixels=np.asarray(image).copy()
        packet=self.observations.packet(pixels,ns)
        self.policy.observe(packet,now_ns=ns)
        self.last_observed_ns=ns; self.observed_indices.append(index)
        return index

    def command_tick(self,requested):
        if self.observe_enabled: self.observe_current()
        return super().command_tick(requested)


def collect(spec,output,template):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    policy=OnlineTemporalChoice(spec['method'],template.models,template.bindings)
    policy.begin_episode((0,0,0))
    session=None; prefix=None; tape=[]; events=[]; sensor_fault=None; stop_reason=None
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=LiveSession(spec,output,policy); session.install_contact_identity()
        actuator=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(output/'actuator_identity.json',actuator)
        try:
            session.settle_recorded(); prefix=session.run_edge(spec)
        except (PhysicalStop,SensorContractError) as error:
            if isinstance(error,SensorContractError): sensor_fault=str(error)
            prefix=evaluate_edge(spec,session.edge_arrays(),stop_reason=str(error),crossing=session.crossing(session.edge_arrays()))
            prefix.update(edge_index=0,geometry=spec['geometry'],controller_terminal_reason=None,
                terminal_global_sample_index=len(session.samples)-1)
        start_packet=session.capture_current(); start_index=len(session.samples)-1
        binding=prefix_binding(session,start_packet)
        branchable=bool(prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing'] and prefix['checks']['no_disallowed_contact'])
        def clock(): return int(round(float(session.samples[-1]['timestamp_s'])*1e9))
        def record_command(command,stage,index):
            tape.append({'tick':len(tape),'pre_sample_index':len(session.samples)-1,
                'timestamp_s':float(session.samples[-1]['timestamp_s']),'requested_command':list(command),
                'stage':stage,'decision_index':index})
        def step(command,release):
            session.edge_index=1; session.phase=2 if release else 1; session.observe_enabled=not release
            try: session.command_tick(command)
            finally: tape[-1]['post_sample_index']=len(session.samples)-1
        def record_selection(selection):
            event={'pre_sample_index':len(session.samples)-1,'observation_index':session.capture_current(),'selection':selection}
            write_json(output/f'choice_{len(events):02d}.json',event); events.append(event)
        def record_fault(message):
            nonlocal sensor_fault
            sensor_fault=message
        if branchable:
            try:
                session.observe_current()
                policy.begin_control(spec['intent_xy_body_start_m'],now_ns=clock())
            except SensorContractError as error: sensor_fault=str(error)
            if sensor_fault is None:
                try:
                    outcome=execute_control(policy,observe=session.observe_current,step=step,clock=clock,
                        record_selection=record_selection,record_command=record_command,record_fault=record_fault)
                    sensor_fault=outcome['sensor_fault']
                except PhysicalStop as error: stop_reason=str(error)
        else: stop_reason=prefix['stop_reason'] or 'PREFIX_NOT_CROSSED'
        # Initialization/prefix sensor faults have not entered execute_control.
        # Explicit zero release, with failed observation ingestion disabled.
        if sensor_fault is not None and not any(e['stage']=='fault_release' for e in tape):
            try:
                for _ in range(5):
                    record_command([0.,0.,0.],'fault_release',None); step([0.,0.,0.],True)
            except PhysicalStop as error: stop_reason=str(error)
        if sensor_fault is not None and stop_reason is None: stop_reason='SENSOR_CONTRACT_FAILURE'
        session.capture_current(); raw=session.persist(output); session.persist_observations(output)
        write_json(output/'prefix_decisions.json',session.decisions); write_json(output/'command_tape.json',tape)
        write_json(output/'selection_events.json',events); write_json(output/'observed_indices.json',session.observed_indices)
        terminal=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=actuator['effective']: raise ValueError('actuator gain drift')
        write_json(output/'terminal_actuator_gains.json',terminal)
        metrics=reduce_trial(raw,start_index=start_index,tape=tape,selections=[e['selection'] for e in events],
            direction=spec['intent_xy_body_start_m'],branchable=branchable,stop_reason=stop_reason,sensor_fault=sensor_fault)
        return {'scene_id':spec['scene_id'],'layout_id':spec['layout_id'],'data_role':spec['data_role'],
            'method':spec['method'],'intent_name':spec['intent_name'],'intent_xy_body_start_m':spec['intent_xy_body_start_m'],
            'branchable':branchable,'stop_reason':stop_reason,'sensor_fault':sensor_fault,'prefix_result':prefix,
            'prefix_binding':binding,'branch_start_observation_index':start_packet,'prefix_terminal_sample_index':start_index,
            'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),'rgb_packets':len(session.packet_rows),
            'metrics':metrics}
    except Exception:
        if session is not None:
            if not (output/'physics_trace.npz').exists(): session.persist(output); session.persist_observations(output)
            for name,value in (('prefix_decisions.json',session.decisions),('command_tape.json',tape),
                    ('selection_events.json',events),('observed_indices.json',session.observed_indices)):
                if not (output/name).exists(): write_json(output/name,value)
        raise
    finally:
        # Native emergency stops and unexpected infrastructure errors do not
        # advance physics again. Destroying this isolated simulated trial is
        # the terminal stop, not a deployable real-robot safety implementation.
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def prerequisites():
    inputs={}; sources={}; gait={}
    studies=(
        ('go2_online_choice_maze_pilot_development_v1_attempt_001','c8e25c20f52d03a6bc165757c2ebcdad1b620d996a72bb95610f7f5bde9c8045'),
        ('go2_temporal_rgb_body_learning_comparison_development_v1_attempt_001','64c878d4a7524712eaba1648a5f0b251bf8015ddeace1ba991b358b19abf30e5'),
        ('go2_online_rgb_history_replay_development_v1_attempt_001','587653dd19100711269df088c4bfd1ac0c3b6917844fb0038a0c86974b090edc'),
        ('go2_temporal_online_adapter_replay_development_v1_attempt_001','262fa3019560d2e5f1ed4e376b45d831bc97e3127d7adcf07b59eb274f9a8444'))
    for name,sha in studies:
        directory=ROOT/'.generated'/name; verify({str((directory/'result.json').relative_to(ROOT)):sha})
        result=json.loads((directory/'result.json').read_text()); launch=json.loads((directory/'launch.json').read_text())
        if result['status'] not in ('PASS','COMPLETE') or digest(directory/'launch.json')!=result['launch_sha256']:
            raise ValueError('completed prerequisite launch/result binding')
        if 'checks' in result and not all(result['checks'].values()): raise ValueError('prerequisite failed check')
        verify(launch['source_sha256']); verify(launch.get('gait_sha256',{}))
        for path,expected in launch['source_sha256'].items():
            if path in sources and sources[path]!=expected: raise ValueError('conflicting source bindings')
            sources[path]=expected
        gait.update(launch.get('gait_sha256',{}))
        for leaf in ('result.json','launch.json'): inputs[str((directory/leaf).relative_to(ROOT))]=digest(directory/leaf)
    sources.update({p:digest(ROOT/p) for p in NEW_SOURCES}); verify(sources)
    return sources,gait,inputs


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed one-shot fresh panel; no CLI overrides or resume')
    if shutil.disk_usage(OUTPUT.parent).free<10*1024**3: raise ValueError('less than ten GiB free')
    sources,gait,inputs=prerequisites(); specs=trials()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    templates={method:OnlineTemporalChoice.from_completed_study(method) for method in METHODS}
    launch={'schema':'successive_choice_maze_development.v1','trial_specs':specs,'source_sha256':sources,
        'gait_sha256':gait,'prerequisite_sha256':inputs,'model_bindings':{k:v.bindings for k,v in templates.items()},
        'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'fresh successive directional continuation; teacher initialization and ideal sensing; not maze exploration or hardware'}
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch); rows=[]; references={}
    try:
        for spec in specs:
            directory=OUTPUT/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event':'trial_started','scene_id':spec['scene_id'],'completed':len(rows),'total':144}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect(spec,directory,templates[spec['method']])
            leaves=['actuator_identity.json','terminal_actuator_gains.json','physics_trace.npz','native_contacts.npz',
                'contact_topology.json','contact_events.json','process.log','ideal_sensor_samples.npz','policy_histories.npz',
                'policy_observations.json','camera_audit.json','prefix_decisions.json','command_tape.json','selection_events.json','observed_indices.json']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            leaves.extend(f'choice_{i:02d}.json' for i in range(row['metrics']['decisions']))
            row['artifact_sha256']={name:digest(directory/name) for name in leaves}; write_json(directory/'result.json',row)
            reference=references.setdefault(spec['layout_id'],(directory,row))
            match=compare_prefix(reference[0],reference[1],directory,row)
            match={k:v for k,v in match.items() if not k.startswith('canonical_model_context_')}
            rows.append(row | {'result_sha256':digest(directory/'result.json'),'prefix_match':match,
                'prefix_reference_scene_id':reference[1]['scene_id']})
            print(json.dumps({'event':'trial_finished','scene_id':spec['scene_id'],'completed':len(rows),'total':144,
                'stop_reason':row['stop_reason'],'actions':row['metrics']['selected_actions'],
                'signed_displacement_m':row['metrics']['observed_signed_control_displacement_m']}),flush=True)
        verify(sources | gait | inputs)
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','planned_trials':144,'completed_trials':len(rows),
            'trials':rows,'reduction':paired_reduction(rows),'launch_sha256':digest(OUTPUT/'launch.json')})
        print(json.dumps({'status':'COMPLETE','trials':len(rows)}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'INFRASTRUCTURE_FAILURE','error':str(error),
            'planned_trials':144,'completed_trials':len(rows),'trials':rows,'launch_sha256':digest(OUTPUT/'launch.json')})
        raise


if __name__=='__main__': main()

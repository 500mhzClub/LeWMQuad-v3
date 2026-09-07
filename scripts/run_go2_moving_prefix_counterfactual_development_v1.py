#!/usr/bin/env python3
"""384 actual alternative suffixes after matched one-second moving prefixes."""
import contextlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.local_execution_controller_development import evaluate_edge
from lewm.moving_prefix_counterfactual_development import trials,evidence_cells
from lewm.moving_prefix_evidence_development import reference_context,suffix_targets
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import prefix_binding
from scripts.run_go2_multijunction_route_development_v1 import RouteSession,PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json

OUTPUT=ROOT/'.generated/go2_moving_prefix_counterfactual_development_v1_attempt_001'
PROTOCOL='docs/go2_moving_prefix_counterfactual_development_v1_2026-09-05.md'
NEW_SOURCES=(str(Path(__file__).relative_to(ROOT)),PROTOCOL,'lewm/moving_prefix_counterfactual_development.py',
    'lewm/moving_prefix_evidence_development.py','lewm/tests/test_moving_prefix_counterfactual_development.py',
    'lewm/tests/test_moving_prefix_evidence_development.py')


def load_references():
    dataset=AuditedSubtrajectoryDataset(DERIVATION_ROOT,'train')
    windows=json.loads((DERIVATION_ROOT/'windows.json').read_text())
    selected=[w for w in windows if w['offset_ns']==1_000_000_000 and w['action_index'] in range(1,5)]
    if len(selected)!=96 or len({w['scene_id'] for w in selected})!=96: raise ValueError('96 distinct existing moving contexts required')
    references={}; bindings={}
    for window in selected:
        scene=window['scene_id']; directory=dataset.corpus.paths[scene]; record=dataset.corpus.members[scene]
        member=record['result']; reference=reference_context(directory,member,window)
        references[scene]={'directory':str(directory.relative_to(ROOT)),'reference':reference,
            'member_result_sha256':record['result_sha256']}
        for leaf in ('result.json','physics_trace.npz','policy_histories.npz','camera_audit.json'):
            expected=record['result_sha256'] if leaf=='result.json' else member['artifact_sha256'][leaf]
            bindings[str((directory/leaf).relative_to(ROOT))]=expected
    verify(bindings)
    return references,bindings


def collect(spec,output):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    session=None; tape=[]; stop_reason=None; prefix=None; suffix_end=None
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
        teacher_packet=session.capture_current(); teacher_end=len(session.samples)-1
        teacher_available=bool(prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing'] and prefix['checks']['no_disallowed_contact'])
        def command(command,stage):
            session.edge_index=1 if stage=='moving_prefix' else 2; session.phase=2 if stage=='release' else 1
            entry={'tick':len(tape),'pre_sample_index':len(session.samples)-1,
                'timestamp_s':float(session.samples[-1]['timestamp_s']),'stage':stage,'requested_command':list(command)}
            tape.append(entry)
            try: session.command_tick(command)
            finally: entry['post_sample_index']=len(session.samples)-1
        if teacher_available:
            try:
                for _ in range(10): command(spec['prefix_command'],'moving_prefix')
            except PhysicalStop as error: stop_reason=str(error)
        else: stop_reason=prefix['stop_reason'] or 'PREFIX_NOT_CROSSED'
        start_packet=session.capture_current(); start=len(session.samples)-1
        branchable=bool(teacher_available and stop_reason is None and len(tape)==10 and start-teacher_end==500)
        binding=prefix_binding(session,start_packet); suffix_end=start
        if branchable:
            try:
                for _ in range(30):
                    try: command(spec['future_command'],'suffix')
                    finally: suffix_end=len(session.samples)-1
                session.capture_current()
                for _ in range(5): command([0.,0.,0.],'release')
            except PhysicalStop as error: stop_reason=str(error)
        session.capture_current(); raw=session.persist(output); session.persist_observations(output)
        write_json(output/'prefix_decisions.json',session.decisions); write_json(output/'command_tape.json',tape)
        terminal=read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal!=actuator['effective']: raise ValueError('actuator gain drift')
        write_json(output/'terminal_actuator_gains.json',terminal)
        window=suffix_targets(raw,start,suffix_end,session.model_manifest) if branchable else None
        write_json(output/'suffix_window.json',window)
        return {'scene_id':spec['scene_id'],'layout_id':spec['layout_id'],'data_role':spec['data_role'],
            'prefix_action_index':spec['prefix_action_index'],'future_action_index':spec['future_action_index'],
            'reference_scene_id':spec['reference_scene_id'],'teacher_available':teacher_available,'branchable':branchable,
            'stop_reason':stop_reason,'prefix_result':prefix,'teacher_terminal_sample_index':teacher_end,
            'teacher_terminal_observation_index':teacher_packet,'prefix_terminal_sample_index':start,
            'branch_start_observation_index':start_packet,'prefix_binding':binding,'suffix_terminal_sample_index':suffix_end,
            'suffix_window':window,'physics_samples':len(session.samples),'sensor_samples':len(session.sensor_rows),
            'rgb_packets':len(session.packet_rows),'any_contact':bool(raw['physics_contact'].any())}
    except Exception:
        if session is not None:
            if not (output/'physics_trace.npz').exists(): session.persist(output); session.persist_observations(output)
            for name,value in (('prefix_decisions.json',session.decisions),('command_tape.json',tape)):
                if not (output/name).exists(): write_json(output/name,value)
        raise
    finally:
        try:
            if session is not None: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def prerequisites():
    previous=ROOT/'.generated/go2_successive_choice_maze_development_v1_attempt_001'
    coverage=ROOT/'.generated/go2_successive_action_coverage_development_v1_attempt_001'
    inputs={str((previous/'raw_artifact_audit_clock_boundary_v2.json').relative_to(ROOT)):'659c7d552ef49e0e9328a341da60fb97f2160639c33ba8fc7495783d4dba6d1f',
        str((previous/'full_audit_source_dependency_witness_clock_boundary_v2.json').relative_to(ROOT)):'f91c7537ad8897fdf835545a05fe557d73acc56c4fe568ad8da8c73767e28398',
        str((coverage/'result.json').relative_to(ROOT)):'2d643ffb0a6c724c9c43b5bb593aa728f20df24b82edb8e9cbf9917e22951f8a',
        str((coverage/'launch.json').relative_to(ROOT)):'68ed1af2335301b081f7f6c88a7393bb2df2919e682da12d9dde946f20e80300',
        str((previous/'launch.json').relative_to(ROOT)):'e9a4bd281e631f06e01a134c3d68b969613d4599bf0554299a55d34bc20f7bf5'}
    verify(inputs)
    witness=json.loads((previous/'full_audit_source_dependency_witness_clock_boundary_v2.json').read_text())
    verify(witness['source_sha256'])
    launch=json.loads((previous/'launch.json').read_text()); verify(launch['gait_sha256'])
    coverage_launch=json.loads((coverage/'launch.json').read_text())
    verify(coverage_launch['source_sha256']|coverage_launch['input_sha256'])
    sources=launch['source_sha256']|coverage_launch['source_sha256']|{p:digest(ROOT/p) for p in NEW_SOURCES}
    inputs.update(coverage_launch['input_sha256'])
    for name in ('result.json','raw_artifact_audit.json','windows.json'):
        inputs[str((DERIVATION_ROOT/name).relative_to(ROOT))]=digest(DERIVATION_ROOT/name)
    references,reference_bindings=load_references(); inputs.update(reference_bindings)
    verify(sources|inputs|launch['gait_sha256'])
    return sources,inputs,launch['gait_sha256'],references


def main():
    if len(sys.argv)!=1 or OUTPUT.exists(): raise ValueError('fixed fresh collection; no override or resume')
    if shutil.disk_usage(OUTPUT.parent).free<10*1024**3: raise ValueError('less than ten GiB free')
    sources,inputs,gait,references=prerequisites(); specs=trials()
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',{'schema':'moving_prefix_counterfactual_development.v1',
        'trial_specs':specs,'planned_composite_cells':evidence_cells(),'source_sha256':sources,'input_sha256':inputs,
        'gait_sha256':gait,'references':references,
        'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'384 development action switches on24 existing layouts; source-matched moving prefixes; no fitting or final test'})
    rows=[]
    try:
        for spec in specs:
            directory=OUTPUT/spec['scene_id']; directory.mkdir()
            print(json.dumps({'event':'switch_trial_started','scene_id':spec['scene_id'],'completed':len(rows),'total':384}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream): row=collect(spec,directory)
            leaves=['physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
                'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
                'prefix_decisions.json','command_tape.json','suffix_window.json','process.log']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            row['artifact_sha256']={name:digest(directory/name) for name in leaves}; write_json(directory/'result.json',row)
            reference=references[spec['reference_scene_id']]
            match=compare_prefix(ROOT/reference['directory'],reference['reference'],directory,row) if row['branchable'] else None
            rows.append(row|{'result_sha256':digest(directory/'result.json'),'prefix_match':match})
            print(json.dumps({'event':'switch_trial_finished','scene_id':spec['scene_id'],'completed':len(rows),'total':384,
                'branchable':row['branchable'],'stop_reason':row['stop_reason']}),flush=True)
        verify(sources|inputs|gait)
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','planned_trials':384,'completed_trials':len(rows),'trials':rows,
            'branchable_trials':sum(r['branchable'] for r in rows),'contact_trials':sum(r['any_contact'] for r in rows),
            'launch_sha256':digest(OUTPUT/'launch.json')})
        print(json.dumps({'status':'COMPLETE','trials':len(rows)}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',{'status':'INFRASTRUCTURE_FAILURE','error':str(error),'planned_trials':384,
            'completed_trials':len(rows),'trials':rows,'launch_sha256':digest(OUTPUT/'launch.json')}); raise


if __name__=='__main__': main()

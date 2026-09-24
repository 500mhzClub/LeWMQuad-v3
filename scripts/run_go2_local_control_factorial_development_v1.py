#!/usr/bin/env python3
"""Fixed 2x2 local-control study with actual same-state corridor continuation."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.local_execution_controller_development import (ARMS,LocalController,continuation_geometry,
    evaluate_edge,motion_window_ok,trial_spec)
from lewm.physical_execution_development import KINDS,WIDTHS,rotation_xyzw
from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession,PhysicalStop
from scripts import run_physical_graph_edge_handoff_qualification_v1 as BASE


class FactorialSession(AttributedSession):
    def __init__(self,spec):
        self.edge_index=0
        self.decisions=[]
        super().__init__(spec)

    def _sample(self,requested,applied,timestamp_s):
        before=len(self.samples)
        try:
            return super()._sample(requested,applied,timestamp_s)
        finally:
            if len(self.samples)>before:
                self.samples[-1]['edge_index']=np.uint8(self.edge_index)

    def command_tick(self,requested):
        runner=self.ctx.runner
        block=runner._clip_block(np.asarray([[requested]*5],dtype=np.float32))
        command=np.asarray(block.executed[0,0],dtype=np.float32)
        for _ in range(runner._policy_steps_per_command_tick):
            observation=runner._build_observation(command[None,:])
            targets=self.ctx.policy.act(observation)
            runner._apply_joint_targets(targets)
            base_time=float(runner._sim_time_ns)/1e9
            for physics_step in range(runner._physics_steps_per_policy):
                self.ctx.build.scene.step()
                self._sample(requested,command,base_time+(physics_step+1)*.002)
            runner._sim_time_ns+=runner._policy_dt_ns
        runner._last_executed=command[None,:].copy()

    def edge_arrays(self):
        rows=[row for row in self.samples if row['edge_index']==self.edge_index]
        return {key:np.stack([row[key] for row in rows]) for key in rows[0]} if rows else {}

    def crossing(self,arrays):
        if not arrays:
            return None
        active=arrays['phase']!=0
        if not active.any():
            return None
        edge=self.geometry['selected_directed_edge']
        try:
            value=BASE.canonical_port_crossing(arrays['base_pose_world'][active],
                np.zeros(np.count_nonzero(active),dtype=np.uint8),edge['opening_segment_world'],
                edge['opening_normal_world'],[],sustained_samples=100)
            value['edge_id']=edge['edge_id']
            return value
        except BASE.ExperimentError as exc:
            if str(exc) not in ('teacher trace never crosses the canonical directed port',
                               'teacher enters a competing physical port first',
                               'teacher did not remain beyond the port for 100 physics samples'):
                raise
            return None

    def run_edge(self,spec):
        controller=LocalController(spec['arm'])
        stop_reason=None
        route=self.geometry['teacher_route_polyline_world']
        initial_heading=math.atan2(route[1][1]-route[0][1],route[1][0]-route[0][0])
        normal=self.geometry['selected_directed_edge']['opening_normal_world']
        desired=math.atan2(normal[1],normal[0])
        try:
            for tick in range(86):
                last=self.samples[-1]
                pose,twist=last['base_pose_world'],last['base_twist_world']
                yaw=BASE._pose_yaw_xyzw(pose)
                lookahead,_,_=BASE._lookahead_from_port([pose[0],pose[1],yaw],route,.18)
                heading=math.atan2(lookahead[1]-pose[1],lookahead[0]-pose[0])
                arrays=self.edge_arrays()
                crossing=self.crossing(arrays)
                arrived_rows=[row for row in self.samples if row['edge_index']==self.edge_index and row['phase']==2]
                stable=motion_window_ok([row['base_pose_world'] for row in arrived_rows],
                    [row['base_twist_world'] for row in arrived_rows],self.geometry,spec['width_m'])
                values={'tick':tick,'alignment_error':BASE._wrap_angle(initial_heading-yaw),
                    'pursuit_error':BASE._wrap_angle(heading-yaw),'arrival_error':BASE._wrap_angle(desired-yaw),
                    'body_forward_velocity':float(rotation_xyzw(pose[3:])[:,0]@twist[:3]),
                    'angular_velocity':float(twist[5]),'crossed':crossing is not None,'stable_arrival':stable}
                requested=controller.decide(**values)
                self.decisions.append({'edge_index':self.edge_index,'stage':controller.stage,
                    'pre_sample_index':len(self.samples)-1,'timestamp_s':float(last['timestamp_s']),
                    'inputs':values,'requested_command':requested})
                if requested is None:
                    break
                self.phase=2 if controller.stage=='ARRIVE' else 1
                self.command_tick(requested)
        except PhysicalStop as exc:
            stop_reason=str(exc)
        arrays=self.edge_arrays()
        edge_spec={**spec,'geometry':copy.deepcopy(self.geometry)}
        result=evaluate_edge(edge_spec,arrays,stop_reason=stop_reason,crossing=self.crossing(arrays))
        result.update(edge_index=self.edge_index,geometry=edge_spec['geometry'],
                      controller_terminal_reason=controller.terminal_reason,
                      terminal_global_sample_index=len(self.samples)-1)
        return result


def collect_trial(spec,output):
    from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
    session=None
    edges=[]
    images={}
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=FactorialSession(spec)
        session.install_contact_identity()
        try:
            session.settle_recorded()
        except PhysicalStop as exc:
            result=evaluate_edge(spec,session.edge_arrays(),stop_reason=str(exc),crossing=None)
            result.update(edge_index=0,geometry=copy.deepcopy(session.geometry),controller_terminal_reason=None,
                          terminal_global_sample_index=len(session.samples)-1)
            edges.append(result)
        else:
            images['initial']=session.capture_fixed_rgb(output,'initial_rgb')
            for index in range(2):
                session.edge_index=index
                if index==1:
                    session.geometry=continuation_geometry(spec['geometry'])
                result=session.run_edge(spec)
                edges.append(result)
                images[f'edge{index}']=session.capture_fixed_rgb(output,f'edge{index}_rgb')
                if result['stop_reason'] is not None:
                    break
        images['final']=session.capture_fixed_rgb(output,'final_rgb')
        session.persist(output)
        (output/'decisions.json').write_text(json.dumps(session.decisions,indent=2,allow_nan=False)+'\n')
        completed_crossings=sum(row['checks']['sustained_correct_crossing'] and row['checks']['no_disallowed_contact'] for row in edges)
        overall=bool(len(edges)==2 and completed_crossings==2 and edges[-1]['status']=='SUCCESS')
        return {'scene_id':spec['scene_id'],'arm':spec['arm'],'case_index':spec['case_index'],
                'status':'SUCCESS' if overall else 'TASK_FAILURE','edges':edges,'images':images,
                'two_contact_free_crossings':bool(len(edges)==2 and completed_crossings==2),
                'two_usable_arrivals':bool(len(edges)==2 and all(row['status']=='SUCCESS' for row in edges)),
                'physics_samples':len(session.samples),
                'first_disallowed_contact':next((row for row in session.contact_events if row['disallowed_contacts']),None)}
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.output_dir.absolute()
    if any(part=='sealed' or part=='sealed_test.json' or part.startswith('sealed_') for part in output.parts):
        parser.error('protected path forbidden')
    output.mkdir(exist_ok=False)
    specs=[trial_spec(kind,width,arm) for kind in KINDS for width in WIDTHS for arm in ARMS]
    paths=('scripts/run_go2_local_control_factorial_development_v1.py',
        'lewm/local_execution_controller_development.py','lewm/physical_execution_development.py',
        'scripts/run_go2_contact_attributed_execution_development_v1.py','lewm/physical_semantics.py',
        'lewm/safety/contact_attribution.py','lewm/safety/contact_hazard_ontology_v1.py',
        'scripts/run_physical_graph_edge_handoff_qualification_v1.py',
        'lewm_genesis/lewm_genesis/rollout.py','lewm_genesis/lewm_genesis/scene_builder.py',
        'lewm_genesis/lewm_genesis/scene_loader.py','config/go2_platform_manifest.yaml',
        'config/go2_primitive_registry.yaml','docs/go2_local_control_factorial_development_v1_2026-09-05.md')
    platform=yaml.safe_load((ROOT/'config/go2_platform_manifest.yaml').read_text())
    policy=platform['locomotion']['policy_artifact']
    gait={}
    for key,digest_key in (('path','sha256'),('cfg_path','cfg_sha256')):
        digest=hashlib.sha256((ROOT/policy[key]).read_bytes()).hexdigest()
        if digest!=policy[digest_key]:
            raise ValueError('gait binding mismatch')
        gait[policy[key]]=digest
    launch={'schema':'go2_local_control_factorial_development.v1','trial_specs':specs,
        'source_sha256':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
        'gait_sha256':gait,'versions':{name:importlib.metadata.version(name) for name in ('genesis-world','torch','numpy')},
        'scope':'oracle local-control development with corridor continuation; no JEPA/novel-maze/real-platform claim'}
    (output/'launch.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    rows=[]
    status='COMPLETE'
    try:
        for spec in specs:
            trial_dir=output/spec['scene_id']
            trial_dir.mkdir(exist_ok=False)
            print(json.dumps({'event':'trial_started','trial':spec['scene_id'],'completed':len(rows),'total':32}),flush=True)
            start=time.monotonic()
            with (trial_dir/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect_trial(spec,trial_dir)
            leaves=('physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','decisions.json',
                    'initial_rgb.png','edge0_rgb.png','edge1_rgb.png','final_rgb.png','process.log')
            row['artifact_sha256']={leaf:hashlib.sha256((trial_dir/leaf).read_bytes()).hexdigest() for leaf in leaves if (trial_dir/leaf).exists()}
            (trial_dir/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            rows.append(row)
            print(json.dumps({'event':'trial_finished','trial':row['scene_id'],'status':row['status'],
                'edges':[edge['status'] for edge in row['edges']],'two_crossings':row['two_contact_free_crossings'],
                'elapsed_s':round(time.monotonic()-start,3),'completed':len(rows),'total':32}),flush=True)
    except Exception as exc:
        traceback.print_exc()
        status='INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}',
            'traceback':traceback.format_exc(),'completed_trials':len(rows)},indent=2)+'\n')
    report={'status':status,'planned_trials':32,'completed_trials':len(rows),'trials':rows,
        'by_arm':{arm:{'trials':sum(row['arm']==arm for row in rows),
                       'task_successes':sum(row['arm']==arm and row['status']=='SUCCESS' for row in rows),
                       'two_crossings':sum(row['arm']==arm and row['two_contact_free_crossings'] for row in rows),
                       'two_usable_arrivals':sum(row['arm']==arm and row['two_usable_arrivals'] for row in rows)} for arm in ARMS},
        'launch_sha256':hashlib.sha256((output/'launch.json').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='trials'},indent=2))
    return 0 if status=='COMPLETE' else 1


if __name__=='__main__':
    raise SystemExit(main())

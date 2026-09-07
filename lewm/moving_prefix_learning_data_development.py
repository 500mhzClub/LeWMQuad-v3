"""Policy-only 600-cell composite; requires all 384 physical trials raw-audited.

The 120 original initial cells and 96 original moving continuations are reused,
not simulated again. All unavailable new cells are retained in accounting, but
cannot supply a training item. This selected panel does not silently include
the other 698 original temporal windows. No raw physics or camera-world file is
opened by this module. Shared context is a training construction only.
"""
import hashlib
import json
from pathlib import Path

import torch

from lewm.causal_subtrajectory_learning_development import (
    AuditedSubtrajectoryDataset,DERIVATION_ROOT,causal_history_tensors,remaining_plan)
from lewm.counterfactual_learning_data_development import ROOT
from lewm.counterfactual_maze_development import ACTIONS,HORIZONS_NS
from lewm.moving_prefix_counterfactual_development import evidence_cells,trials
from lewm.rgb_body_tensor_interface_development import observation_tensors
from lewm.route_rgb_dataset_development import load_route_observation

OUTPUT=ROOT/'.generated/go2_moving_prefix_counterfactual_development_v1_attempt_001'
LAUNCH_SHA='f8f1c6e01017aa08363bd8ad6997145e9df1a4fe52f115d1a88536608a4f3ae7'


def checked_policy_leaf(directory,name,expected):
    path=directory/name
    if path.resolve()!=path or path.parent!=directory or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('policy artifact identity or path changed')
    return path


def switch_tensors(window,command,context_packets,future_packet):
    """future_packet is target-only; never consulted to construct the plan/history."""
    if window['remaining_ticks']!=30: raise ValueError('fixed three-second switch suffix required')
    history=causal_history_tensors(context_packets,window['decision_ns'])
    prior=context_packets[-1]['sensor_state']['control']['applied_command']
    if not prior['valid'][-1].all(): raise ValueError('previous applied command unavailable')
    plan=remaining_plan(command,prior['values'][-1],30)
    context={k:v[-1] for k,v in history.items()}
    future={k:torch.zeros((8,*v.shape),dtype=v.dtype) for k,v in context.items()}
    motion=torch.full((8,3),float('nan')); contact=torch.full((8,),float('nan'))
    mv=torch.zeros(8,dtype=torch.bool); cv=mv.clone()
    for i,(horizon,target) in enumerate(zip(HORIZONS_NS,window['targets'],strict=True)):
        inside=i<6; valid_motion=target['motion_valid']; valid_contact=target['contact_valid']
        if (target['horizon_ns']!=horizon or target['in_plan']!=inside
                or type(valid_motion) is not bool or type(valid_contact) is not bool
                or (not inside and (valid_motion or valid_contact)) or (valid_motion and not valid_contact)):
            raise ValueError('target horizon or censoring mismatch')
        mv[i]=valid_motion; cv[i]=valid_contact
        if valid_motion:
            if target['contact_by_horizon'] is not False: raise ValueError('motion at/after contact')
            packet=future_packet(target['future_observation_index'])
            if packet['image']['measured_ns']!=window['decision_ns']+horizon: raise ValueError('future packet clock')
            if packet['sensor_state']['identity']!=context_packets[-1]['sensor_state']['identity']:
                raise ValueError('future episode mismatch')
            values=observation_tensors(packet)
            for k,v in values.items(): future[k][i]=v
            motion[i]=torch.tensor(target['delta_xy_yaw_current_body'])
            if not torch.isfinite(motion[i]).all(): raise ValueError('valid motion nonfinite')
        elif target['delta_xy_yaw_current_body'] is not None or target['future_observation_index'] is not None:
            raise ValueError('censored motion populated')
        if valid_contact:
            if type(target['contact_by_horizon']) is not bool: raise ValueError('explicit contact target required')
            contact[i]=float(target['contact_by_horizon'])
        elif target['contact_by_horizon'] is not None: raise ValueError('unknown contact populated')
    return {'observation_history':history,**plan,'targets':{'future_observations':future,'future_valid':mv.clone(),
        'motion':motion,'motion_valid':mv,'contact':contact,'contact_valid':cv}}


class AuditedMovingPrefixDataset:
    """All siblings stay in one layout role; missing cells are never imputed."""
    def __init__(self,directory,role):
        directory=Path(directory).absolute()
        if directory!=OUTPUT or directory.resolve()!=directory: raise ValueError('exact non-symlink development root required')
        if role not in ('train','validation'): raise ValueError('explicit development role required')
        self.role=role
        launch_path=checked_policy_leaf(directory,'launch.json',LAUNCH_SHA)
        launch=json.loads(launch_path.read_text())
        report=json.loads((directory/'result.json').read_text()); audit=json.loads((directory/'raw_artifact_audit.json').read_text())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if (report['status']!='COMPLETE' or report['completed_trials']!=384 or report['planned_trials']!=384
                or audit['status']!='PASS' or audit['audited_trials']!=384
                or audit['study_result_sha256']!=digest(directory/'result.json')
                or audit['launch_sha256']!=LAUNCH_SHA or report['launch_sha256']!=LAUNCH_SHA):
            raise ValueError('complete full raw audit required before fitting')
        if launch['trial_specs']!=trials() or launch['planned_composite_cells']!=evidence_cells():
            raise ValueError('fixed development population changed')
        if len(report['trials'])!=384 or len(audit['trials'])!=384: raise ValueError('complete member population required')
        self.old=AuditedSubtrajectoryDataset(DERIVATION_ROOT,role)
        self.old_indices={(r['scene_id'],r['offset_ns']):i for i,r in enumerate(self.old.rows)}
        self.new={}; self.paths={}
        for spec,row,checked in zip(trials(),report['trials'],audit['trials'],strict=True):
            scene=spec['scene_id']; path=directory/scene
            for key in ('scene_id','layout_id','data_role','prefix_action_index','future_action_index','reference_scene_id'):
                if row[key]!=spec[key]: raise ValueError('member identity mismatch')
            if checked['scene_id']!=scene or checked['status']!='PASS' or checked['result_sha256']!=row['result_sha256']:
                raise ValueError('member audit identity changed')
            member=json.loads(checked_policy_leaf(path,'result.json',row['result_sha256']).read_text())
            if row!=member|{'result_sha256':row['result_sha256'],'prefix_match':checked['prefix_match']}:
                raise ValueError('member/root audit disagreement')
            names=['policy_observations.json','policy_histories.npz']+[f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])]
            for name in names: checked_policy_leaf(path,name,row['artifact_sha256'][name])
            if row['branchable']:
                match=row['prefix_match']; reference=launch['references'][spec['reference_scene_id']]['reference']
                expected_index=reference['branch_start_observation_index']
                if (match['status']!='MATCH' or match['physical_and_history_exact'] is not True
                        or match['camera_geometry_exact'] is not True
                        or match['canonical_model_context_scene_id']!=spec['reference_scene_id']
                        or match['canonical_model_context_observation_index']!=expected_index
                        or row['suffix_window']['history_observation_indices']!=reference['history_observation_indices']):
                    raise ValueError('moving shared-context evidence changed')
            elif row['suffix_window'] is not None or row['prefix_match'] is not None:
                raise ValueError('unavailable prefix has fabricated evidence')
            self.new[scene]=row; self.paths[scene]=path
        self.cells=[c for c in evidence_cells() if c['data_role']==role]; self.rows=[]; self.unavailable=[]
        for cell in self.cells:
            if cell['existing_offset_ns'] is not None:
                key=(cell['scene_id'],cell['existing_offset_ns'])
                if key not in self.old_indices: raise ValueError('required existing evidence cell absent')
                self.rows.append(cell)
            elif self.new[cell['scene_id']]['branchable']: self.rows.append(cell)
            else: self.unavailable.append(cell)

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        cell=self.rows[index]
        if cell['existing_offset_ns'] is not None:
            item=self.old[self.old_indices[(cell['scene_id'],cell['existing_offset_ns'])]]
        else:
            row=self.new[cell['scene_id']]; scene=row['reference_scene_id']; reference=self.old.corpus.members[scene]['result']
            if reference['layout_id']!=cell['layout_id'] or reference['data_role']!=self.role: raise ValueError('shared context crossed layout role')
            packets=[load_route_observation(self.old.corpus.paths[scene],i) for i in row['suffix_window']['history_observation_indices']]
            item=switch_tensors(row['suffix_window'],ACTIONS[cell['future_action_index']][1],packets,
                lambda i:load_route_observation(self.paths[cell['scene_id']],i))
            item['metadata']={'window_id':cell['scene_id']+'-moving-suffix','scene_id':cell['scene_id'],
                'layout_id':cell['layout_id'],'data_role':self.role,'action_index':cell['future_action_index'],'offset_ns':1_000_000_000}
        item['metadata'].update(prefix_action_index=cell['prefix_action_index'],provenance=cell['provenance'],
            conditioning_group=cell['layout_id']+'-past-'+str(cell['prefix_action_index']))
        return item

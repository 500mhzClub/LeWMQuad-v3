"""Audited development corpus -> exact shared contexts, known plans, separate targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lewm.counterfactual_maze_development import ACTIONS,HORIZONS_NS
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.rgb_body_tensor_interface_development import observation_tensors

ROOT=Path(__file__).resolve().parents[1]


def prospective_plan(command,previous):
    command,previous=np.asarray(command,dtype=np.float32),np.asarray(previous,dtype=np.float32)
    if command.shape!=(3,) or previous.shape!=(3,) or not np.isfinite(command).all() or not np.isfinite(previous).all():
        raise ValueError('invalid prospective command state')
    limits=np.array([.3,0.,.5],dtype=np.float32)
    if np.any(np.abs(command)>limits+1e-7) or np.any(np.abs(previous)>limits+1e-7):
        raise ValueError('command outside declared action contract')
    delta=np.array([.25,0.,.35],dtype=np.float32)
    rows=[]
    for _ in range(40):
        previous=(previous+np.clip(command-previous,-delta,delta)).astype(np.float32)
        rows.append(previous.copy())
    return np.stack(rows)


class AuditedCounterfactualDataset:
    """No raw physics or camera-world artifacts are opened by the learning path."""
    def __init__(self,directory,role):
        if role not in ('train','validation'): raise ValueError('explicit development role required')
        self.directory=Path(directory).absolute()
        if self.directory.parent!=ROOT/'.generated' or any(p=='sealed' or p.startswith('sealed_') for p in self.directory.parts):
            raise ValueError('explicit development corpus root required')
        launch=json.loads((self.directory/'launch.json').read_text())
        result_path=self.directory/'result.json'
        report=json.loads(result_path.read_text())
        audit=json.loads((self.directory/'raw_artifact_audit.json').read_text())
        if report['status']!='COMPLETE' or audit['status']!='PASS' or audit['audited_trials']!=120:
            raise ValueError('complete raw-audited corpus required before fitting')
        if hashlib.sha256(result_path.read_bytes()).hexdigest()!=audit['study_result_sha256']:
            raise ValueError('audited corpus result binding changed')
        expected_roots={'original':ROOT/'.generated/go2_counterfactual_maze_dataset_development_v1_attempt_001',
                        'recovery':self.directory}
        if launch['roots']!={k:str(v.relative_to(ROOT)) for k,v in expected_roots.items()}:
            raise ValueError('corpus material roots changed')
        self.members={m['result']['scene_id']:m for m in report['members']}
        if len(self.members)!=120: raise ValueError('duplicate/missing corpus member')
        self.paths={}
        for name,member in self.members.items():
            if Path(name).name!=name or not name.startswith('counterfactual-maze-development-v1-'):
                raise ValueError('invalid branch identity')
            path=expected_roots[member['member_root']]/name
            if path.resolve().parent!=expected_roots[member['member_root']].resolve(): raise ValueError('branch path escape')
            if hashlib.sha256((path/'result.json').read_bytes()).hexdigest()!=member['result_sha256']:
                raise ValueError('member result binding changed')
            # Audit evidence must still refer to the bytes consumed for learning.
            # Verify only explicit policy files; never traverse physics/oracle data.
            policy_names=['policy_observations.json','policy_histories.npz']
            policy_names.extend(f'rgb_{i:04d}.png' for i in range(member['result']['rgb_packets']))
            for filename in policy_names:
                artifact=path/filename
                if artifact.resolve().parent!=path.resolve(): raise ValueError('policy artifact path escape')
                expected=member['result']['artifact_sha256'][filename]
                if hashlib.sha256(artifact.read_bytes()).hexdigest()!=expected:
                    raise ValueError('audited policy artifact binding changed')
            self.paths[name]=path
        self.rows=[m for m in report['members'] if m['result']['data_role']==role and m['result']['branchable']]
        self.role=role

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        member=self.rows[index]
        row=member['result']
        matching=member['prefix_match']
        reference=self.members[matching['canonical_model_context_scene_id']]['result']
        if reference['layout_id']!=row['layout_id'] or reference['data_role']!=self.role:
            raise ValueError('canonical context crossed a scene/data role')
        packet=load_route_observation(self.paths[reference['scene_id']],matching['canonical_model_context_observation_index'])
        context=observation_tensors(packet)
        prior=packet['sensor_state']['control']['applied_command']
        if not prior['valid'][-1].all(): raise ValueError('previous applied command unavailable')
        action_name,command=ACTIONS[row['action_index']]
        if action_name!=row['action_name']: raise ValueError('action identity mismatch')
        plan=prospective_plan(command,prior['values'][-1])
        plan=torch.from_numpy((plan/np.array([.3,1.,.5],dtype=np.float32)).reshape(8,5,3))
        observation_manifest=json.loads((self.paths[row['scene_id']]/'policy_observations.json').read_text())
        lookup={f['image_ns']:i for i,f in enumerate(observation_manifest['frames'])}
        t0=packet['image']['measured_ns']
        future={key:torch.zeros((8,*value.shape),dtype=value.dtype) for key,value in context.items()}
        motion=torch.full((8,3),float('nan'))
        contact=torch.full((8,),float('nan'))
        motion_valid=torch.zeros(8,dtype=torch.bool)
        contact_valid=torch.zeros(8,dtype=torch.bool)
        for i,(horizon,label) in enumerate(zip(HORIZONS_NS,row['horizon_labels'],strict=True)):
            if label['horizon_ns']!=horizon: raise ValueError('horizon identity changed')
            motion_valid[i]=label['motion_valid']; contact_valid[i]=label['contact_valid']
            if label['motion_valid']:
                motion[i]=torch.tensor(label['delta_xy_yaw_start_body'])
                if t0+horizon not in lookup: raise ValueError('valid future motion lacks its observation')
                observed=observation_tensors(load_route_observation(self.paths[row['scene_id']],lookup[t0+horizon]))
                for key,value in observed.items(): future[key][i]=value
            if label['contact_valid']: contact[i]=float(label['contact_by_horizon'])
        return {'observation':context,'known_action_blocks':plan,
            'targets':{'future_observations':future,'future_valid':motion_valid.clone(),'motion':motion,
                'contact':contact,'motion_valid':motion_valid,'contact_valid':contact_valid},
            'metadata':{'layout_id':row['layout_id'],'scene_id':row['scene_id'],'data_role':self.role,
                'action_index':row['action_index'],'canonical_context_scene_id':reference['scene_id']}}

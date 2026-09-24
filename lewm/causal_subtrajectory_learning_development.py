"""Strict policy-only causal windows and prospective remaining fixed-action plans."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lewm.causal_subtrajectory_development import HISTORY_OFFSETS_NS,OFFSETS_NS
from lewm.counterfactual_maze_development import ACTIONS,HORIZONS_NS
from lewm.counterfactual_learning_data_development import AuditedCounterfactualDataset,prospective_plan,ROOT
from lewm.rgb_body_tensor_interface_development import observation_tensors
from lewm.route_rgb_dataset_development import load_route_observation

DERIVATION_ROOT=ROOT/'.generated/go2_causal_subtrajectory_development_v1_attempt_001'
CORPUS_ROOT=ROOT/'.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001'


def remaining_plan(command,previous,remaining_ticks):
    if isinstance(remaining_ticks,bool) or not isinstance(remaining_ticks,int) or remaining_ticks not in range(5,41,5):
        raise ValueError('remaining plan must be a fixed half-second branch suffix')
    plan=prospective_plan(command,previous)
    plan[remaining_ticks:]=0.
    mask=np.arange(40)<remaining_ticks
    return {'known_action_blocks':torch.from_numpy((plan/np.array([.3,1.,.5],dtype=np.float32)).reshape(8,5,3)),
        'known_action_valid':torch.from_numpy(mask.reshape(8,5))}


def causal_history_tensors(packets,decision_ns):
    if len(packets)!=4: raise ValueError('exact four-frame causal history required')
    identity=packets[-1]['sensor_state']['identity']; values=[]
    for packet,offset in zip(packets,HISTORY_OFFSETS_NS,strict=True):
        if (packet['image']['measured_ns']!=decision_ns+offset
                or packet['sensor_state']['decision_ns']!=decision_ns+offset
                or packet['sensor_state']['identity']!=identity):
            raise ValueError('past-image clock or episode mismatch')
        values.append(observation_tensors(packet))
    return {k:torch.stack([v[k] for v in values]) for k in values[0]}


class AuditedSubtrajectoryDataset:
    """World-state artifacts are not opened; target labels remain segregated."""
    def __init__(self,directory,role):
        directory=Path(directory).absolute()
        if directory!=DERIVATION_ROOT or directory.resolve()!=directory:
            raise ValueError('exact non-symlink development derivation required')
        if role not in ('train','validation'): raise ValueError('explicit development role required')
        report=json.loads((directory/'result.json').read_text())
        audit=json.loads((directory/'raw_artifact_audit.json').read_text())
        if (report['status']!='COMPLETE' or audit['status']!='PASS'
                or audit['study_result_sha256']!=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest()
                or report['windows_sha256']!=hashlib.sha256((directory/'windows.json').read_bytes()).hexdigest()):
            raise ValueError('audited derivation binding required')
        if report['corpus_result_sha256']!=hashlib.sha256((CORPUS_ROOT/'result.json').read_bytes()).hexdigest():
            raise ValueError('derivation corpus changed')
        # Verifies every exact original/recovery policy artifact, not raw world state.
        self.corpus=AuditedCounterfactualDataset(CORPUS_ROOT,role)
        rows=json.loads((directory/'windows.json').read_text())
        if len(rows)!=report['window_count'] or len({r['window_id'] for r in rows})!=len(rows):
            raise ValueError('window population changed')
        self.rows=[r for r in rows if r['data_role']==role]; self.role=role

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]; member=self.corpus.members[row['scene_id']]['result']
        source=self.corpus.members[row['context_scene_id']]['result']
        if (row['offset_ns'] not in OFFSETS_NS or row['remaining_ticks']!=40-row['offset_ns']//100_000_000
                or source['layout_id']!=member['layout_id'] or row['layout_id']!=member['layout_id']
                or row['data_role']!=source['data_role'] or row['data_role']!=member['data_role']
                or row['action_index']!=member['action_index']):
            raise ValueError('window/source role or identity mismatch')
        canonical=self.corpus.members[row['scene_id']]['prefix_match']['canonical_model_context_scene_id']
        if row['context_scene_id']!=(canonical if row['offset_ns']==0 else row['scene_id']):
            raise ValueError('later context cannot borrow counterfactual sibling')
        packets=[load_route_observation(self.corpus.paths[row['context_scene_id']],i) for i in row['history_observation_indices']]
        history=causal_history_tensors(packets,row['decision_ns'])
        context={k:v[-1] for k,v in history.items()}
        prior=packets[-1]['sensor_state']['control']['applied_command']
        if not prior['valid'][-1].all(): raise ValueError('current past command unavailable')
        plan=remaining_plan(ACTIONS[row['action_index']][1],prior['values'][-1],row['remaining_ticks'])
        future={k:torch.zeros((8,*v.shape),dtype=v.dtype) for k,v in context.items()}
        motion=torch.full((8,3),float('nan')); contact=torch.full((8,),float('nan'))
        motion_valid=torch.zeros(8,dtype=torch.bool); contact_valid=motion_valid.clone()
        for i,(horizon,target) in enumerate(zip(HORIZONS_NS,row['targets'],strict=True)):
            in_plan=horizon<=row['remaining_ticks']*100_000_000
            if target['horizon_ns']!=horizon or target['in_plan']!=in_plan:
                raise ValueError('target plan/horizon mismatch')
            mv,cv=target['motion_valid'],target['contact_valid']
            if (not in_plan and (mv or cv)) or (mv and not cv): raise ValueError('invalid target censoring')
            motion_valid[i]=mv; contact_valid[i]=cv
            if mv:
                if target['contact_by_horizon'] is not False: raise ValueError('motion at/after contact')
                packet=load_route_observation(self.corpus.paths[row['scene_id']],target['future_observation_index'])
                if packet['image']['measured_ns']!=row['decision_ns']+horizon: raise ValueError('future target clock mismatch')
                values=observation_tensors(packet)
                for k,v in values.items(): future[k][i]=v
                motion[i]=torch.tensor(target['delta_xy_yaw_current_body'])
                if not torch.isfinite(motion[i]).all(): raise ValueError('nonfinite valid motion')
            elif target['delta_xy_yaw_current_body'] is not None or target['future_observation_index'] is not None:
                raise ValueError('censored motion/image populated')
            if cv: contact[i]=float(target['contact_by_horizon'])
            elif target['contact_by_horizon'] is not None: raise ValueError('unknown contact populated')
        return {'observation_history':history,**plan,
            'targets':{'future_observations':future,'future_valid':motion_valid.clone(),
                'motion':motion,'motion_valid':motion_valid,'contact':contact,'contact_valid':contact_valid},
            'metadata':{k:row[k] for k in ('window_id','scene_id','layout_id','data_role','action_index','offset_ns')}}

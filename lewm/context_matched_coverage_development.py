"""Keep current-context exposure fixed while expanding observed future actions.

The augmented view retains all914 original temporal windows plus384 new switch
suffixes. A schedule chooses the same context and action quantile for both data
conditions. Only contexts with newly observed alternatives can change targets.
This module does not fit models or sample validation for training.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm.moving_prefix_learning_data_development import AuditedMovingPrefixDataset,OUTPUT as PHYSICAL,ROOT

TENSOR_CHECK=ROOT/'.generated/go2_moving_prefix_tensor_check_development_v1_attempt_001'


def context_table(rows):
    groups={}; ids=set(); layout_roles={}
    for index,row in enumerate(rows):
        if row['source_kind'] not in ('old','switch') or row['data_role'] not in ('train','validation'):
            raise ValueError('explicit original/switch development source required')
        if row['window_id'] in ids: raise ValueError('duplicate window identity')
        ids.add(row['window_id']); layout=row['layout_id']; role=row['data_role']; key=row['context_id']
        if not all(isinstance(x,str) and x for x in (layout,key,row['window_id'])): raise ValueError('nonempty source identities required')
        if layout in layout_roles and layout_roles[layout]!=role: raise ValueError('layout crossed role')
        layout_roles[layout]=role
        if key not in groups: groups[key]={'layout_id':layout,'data_role':role,'old':{},'expanded':{}}
        g=groups[key]; action=row['action_index']
        if type(action) is not int or not 0<=action<5: raise ValueError('fixed action bank required')
        if g['layout_id']!=layout or g['data_role']!=role: raise ValueError('context crossed layout or role')
        if action in g['expanded']: raise ValueError('duplicate context/action observation')
        g['expanded'][action]=index
        if row['source_kind']=='old': g['old'][action]=index
    if not groups or any(not g['old'] for g in groups.values()): raise ValueError('switch cannot introduce a new current context')
    return groups


def matched_schedule(rows,*,updates,seed):
    if type(updates) is not int or updates<=0 or type(seed) is not int or seed<0: raise ValueError('positive updates and nonnegative integer seed required')
    groups=context_table(rows)
    if any(g['data_role']!='train' for g in groups.values()): raise ValueError('training schedule cannot use validation contexts')
    layouts=sorted({g['layout_id'] for g in groups.values()}); by_layout={p:sorted(k for k,g in groups.items() if g['layout_id']==p) for p in layouts}
    rng=np.random.default_rng(seed); schedule=[]
    for update in range(updates):
        batch=[]
        for layout in layouts:
            keys=by_layout[layout]; key=keys[int(rng.integers(len(keys)))]; quantile=float(rng.random()); group=groups[key]
            selected={}
            for condition,field in (('coverage_limited','old'),('expanded','expanded')):
                actions=sorted(group[field]); action=actions[int(quantile*len(actions))]
                selected[condition]={'action_index':action,'dataset_index':group[field][action]}
            batch.append({'layout_id':layout,'context_id':key,'action_quantile':quantile,**selected})
        schedule.append({'update':update,'batch':batch})
    return schedule


class AugmentedCausalDataset:
    """All original windows plus new switches; no replacement of old time strata."""
    def __init__(self,role):
        if role not in ('train','validation'): raise ValueError('explicit development role required')
        report=json.loads((TENSOR_CHECK/'result.json').read_text()); launch=json.loads((TENSOR_CHECK/'launch.json').read_text())
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if (report['status']!='PASS' or report['checked_cells']!=600 or report['conditioning_groups']!=120
                or report['launch_sha256']!=digest(TENSOR_CHECK/'launch.json')
                or launch['input_sha256'][str((PHYSICAL/'result.json').relative_to(ROOT))]!=digest(PHYSICAL/'result.json')
                or launch['input_sha256'][str((PHYSICAL/'raw_artifact_audit.json').relative_to(ROOT))]!=digest(PHYSICAL/'raw_artifact_audit.json')):
            raise ValueError('complete actual composite tensor qualification required')
        self.composite=AuditedMovingPrefixDataset(PHYSICAL,role); self.old=self.composite.old; self.role=role; self.rows=[]
        for index,row in enumerate(self.old.rows):
            self.rows.append({'window_id':row['window_id'],'layout_id':row['layout_id'],'data_role':role,
                'action_index':row['action_index'],'context_id':row['context_scene_id']+':image-'+str(row['history_observation_indices'][-1]),
                'source_kind':'old','source_index':index})
        for index,cell in enumerate(self.composite.rows):
            if cell['existing_offset_ns'] is not None: continue
            row=self.composite.new[cell['scene_id']]
            self.rows.append({'window_id':cell['scene_id']+'-moving-suffix','layout_id':cell['layout_id'],'data_role':role,
                'action_index':cell['future_action_index'],'context_id':row['reference_scene_id']+':image-'+str(row['suffix_window']['history_observation_indices'][-1]),
                'source_kind':'switch','source_index':index})
        self.contexts=context_table(self.rows)
        if len(self.rows)!=(866 if role=='train' else 432): raise ValueError('full original-plus-switch population changed')

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]; item=self.old[row['source_index']] if row['source_kind']=='old' else self.composite[row['source_index']]
        item['metadata'].update(context_id=row['context_id'],coverage_source=row['source_kind'])
        return item

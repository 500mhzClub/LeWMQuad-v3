"""Two authenticated development populations and a fixed equal-batch mixture."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView


class AugmentedFamilySwitchView:
    def __init__(self,family,switch):
        if not isinstance(family,FamilyWindowView) or not isinstance(switch,MovingActionSwitchView):
            raise ValueError('both complete source-specific views required')
        self.family=FamilyWindowView(family.windows);self.switch=MovingActionSwitchView(switch.reports);self.rows=[]
        for i,w in enumerate(self.family.windows):
            self.rows.append(deepcopy(w)|dict(source='family',local_index=i,
                sample_id='family/'+w['window_id'],stratum='initial' if w['offset_ticks']==0 else 'moving'))
        self.switch_offset=len(self.rows)
        for i,r in enumerate(self.switch.reports):
            self.rows.append(dict(source='switch',local_index=i,sample_id='switch/'+r['trial']+'/branch_13',
                trial=r['trial'],cluster=r['cluster'],data_role=r['data_role'],action=r['suffix_action'],
                prefix_action=r['prefix_action'],available=r['outcome']['branch_available'],
                stratum='repeat' if r['prefix_action']==r['suffix_action'] else 'switch',
                targets=[deepcopy(t)|dict(in_plan=True) for t in r['targets']['targets']] if r['targets'] is not None else None))
        if len(self.rows)!=912:raise ValueError('complete 768-window plus144-branch population required')

    def indices(self,role,*,source=None):
        if role not in ('train','geometry_transfer') or source not in (None,'family','switch'):raise ValueError('explicit source/role required')
        return [i for i,r in enumerate(self.rows) if r['data_role']==role and r['available'] and (source is None or r['source']==source)]

    def schedule(self,*,updates,batch_size,seed):
        if type(updates) is not int or updates!=1200 or type(batch_size) is not int or batch_size!=6 or type(seed) is not int or seed<0:
            raise ValueError('fixed 1200-update equal-source schedule and seed required')
        old=self.family.schedule(updates=1200,batch_size=6,seed=seed)
        new=self.switch.schedule(updates=1200,batch_size=6,seed=seed)
        rng=np.random.default_rng(seed+1_000_000_000);batches=[]
        for i in range(600):
            pair=[old['batches'][i],[self.switch_offset+j for j in new['batches'][i]]]
            batches.extend(pair[int(j)] for j in rng.permutation(2))
        row=dict(role='train',updates=updates,batch_size=batch_size,seed=seed,batches=batches,
            sample_ids=[r['sample_id'] for r in self.rows],source_schedule_sha256=dict(family=old['schedule_sha256'],switch=new['schedule_sha256']),
            source_batches=dict(family=600,switch=600),weighting='75 draws per original train episode;50 per new train branch cell',
            outcome_conditioned_sampling=False)
        return row|dict(schedule_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest())

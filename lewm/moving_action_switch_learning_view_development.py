"""Complete branch view and balanced six-suffix training schedules."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from lewm.moving_action_switch_family_development import TRIALS,assignments,ACTIONS
from lewm.moving_action_switch_learning_sample_development import validate_assignment


class MovingActionSwitchView:
    def __init__(self,reports):
        if [r['trial'] for r in reports]!=list(TRIALS):raise ValueError('complete ordered 144-cell view required')
        self.reports=deepcopy(reports)
        for r in self.reports:
            validate_assignment(r)
            if type(r['prefix']['complete']) is not bool:raise ValueError('explicit prefix availability required')
            if r['outcome']['branch_available']!=(r['targets'] is not None):raise ValueError('exact branch target availability required')
            if r['targets'] is not None and not r['prefix']['complete']:raise ValueError('available target requires complete actual prefix')

    def indices(self,role):
        if role not in ('train','geometry_transfer'):raise ValueError('explicit development role required')
        return [i for i,r in enumerate(self.reports) if r['data_role']==role and r['outcome']['branch_available']]

    def schedule(self,*,updates,batch_size,seed):
        if type(updates) is not int or updates!=1200 or type(batch_size) is not int or batch_size!=6 or type(seed) is not int or seed<0:
            raise ValueError('fixed 1200-update six-suffix schedule and explicit seed required')
        indices=self.indices('train')
        if len(indices)!=72:raise ValueError('no planned training cell can disappear')
        groups={}
        for i in indices:
            r=self.reports[i];groups.setdefault((r['cluster'],r['prefix_action']),{})[r['suffix_action']]=i
        if len(groups)!=12 or any(set(g)!=set(ACTIONS) for g in groups.values()):raise ValueError('all twelve complete six-suffix contexts required')
        rng=np.random.default_rng(seed);order=[];batches=[];keys=sorted(groups)
        for _ in range(updates):
            if not order:order=[keys[int(i)] for i in rng.permutation(len(keys))]
            group=groups[order.pop()];batches.append([group[str(a)] for a in rng.permutation(ACTIONS)])
        row=dict(role='train',updates=updates,batch_size=batch_size,seed=seed,batches=batches,
            episode_assignment=assignments(),weighting='each train cluster/prefix context100 batches; each suffix cell100 draws',
            outcome_conditioned_sampling=False,matched_training_contexts=12,training_geometry_clusters=2)
        return row|dict(schedule_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest())

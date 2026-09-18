"""Pure complete-window view and matched role/episode/action-balanced schedules.

Caller authenticates the separate causal derivation and complete measurement
gates. This helper does not read files, grant eligibility, fit, or select a model.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np
from lewm.geometry_progress_layout_family_development import TRIALS,assignments,ACTIONS
from lewm.geometry_progress_family_causal_windows_development import OFFSETS_TICKS,assignment


class FamilyWindowView:
    def __init__(self,windows):
        if [(w['trial'],w['offset_ticks']) for w in windows]!=[(t,o) for t in TRIALS for o in OFFSETS_TICKS]:
            raise ValueError('complete ordered family causal window population required')
        self.windows=deepcopy(windows);self.groups={r:{} for r in ('train','geometry_transfer')}
        self.unavailable=[];self.by_episode={t:[] for t in TRIALS}
        for i,w in enumerate(self.windows):
            assignment(w)
            if type(w['available']) is not bool:raise ValueError('explicit context availability required')
            if w['window_id']!=f"{w['trial']}/offset_{w['offset_ticks']:02d}":raise ValueError('exact window identity required')
            if not w['available']:
                if w['targets'] is not None or w['reason'] not in ('MISSING_ACTUAL_CONTEXT','POST_CONTACT_CONTEXT'):
                    raise ValueError('unavailable context must not fabricate targets')
                self.unavailable.append(dict(index=i,window_id=w['window_id'],reason=w['reason']));continue
            if w['reason'] is not None or len(w['targets'])!=8:raise ValueError('complete available target slots required')
            self.by_episode[w['trial']].append(i)
            self.groups[w['data_role']].setdefault(w['geometry'],{}).setdefault(w['action'],{}).setdefault(w['trial'],[]).append(i)

    def indices(self,role,*,initial_only=False):
        if role not in self.groups or type(initial_only) is not bool:raise ValueError('explicit family role and inference scope required')
        return [i for i,w in enumerate(self.windows) if w['data_role']==role and w['available']
            and (not initial_only or w['offset_ticks']==0)]

    def schedule(self,*,updates,batch_size,seed):
        if type(updates) is not int or not 1<=updates<=1200 or type(batch_size) is not int or batch_size!=6 or type(seed) is not int or seed<0:
            raise ValueError('bounded explicit six-sample training schedule required')
        metadata=assignments();groups=self.groups['train']
        train_trials=[t for t,c in metadata.items() if c['data_role']=='train']
        if any(not self.by_episode[t] for t in train_trials):raise ValueError('no declared training episode can disappear')
        if len(groups)!=4 or any(set(actions)!=set(ACTIONS) for actions in groups.values()):
            raise ValueError('complete four-layout six-action training population required')
        rng=np.random.default_rng(seed);layout_order=[];action_orders={};episode_orders={};window_orders={};draws=[]
        for _ in range(updates*batch_size):
            if not layout_order:layout_order=list(rng.permutation(sorted(groups)))
            layout=str(layout_order.pop())
            if not action_orders.get(layout):action_orders[layout]=list(rng.permutation(sorted(groups[layout])))
            action=str(action_orders[layout].pop());key=(layout,action)
            if not episode_orders.get(key):episode_orders[key]=list(rng.permutation(sorted(groups[layout][action])))
            trial=str(episode_orders[key].pop())
            if not window_orders.get(trial):window_orders[trial]=list(rng.permutation(self.by_episode[trial]))
            draws.append(int(window_orders[trial].pop()))
        row=dict(role='train',updates=updates,batch_size=batch_size,seed=seed,
            batches=[draws[i:i+batch_size] for i in range(0,len(draws),batch_size)],
            window_ids=[w['window_id'] for w in self.windows],episode_assignment=metadata,
            weighting='balanced layout, action, episode; balanced available offsets within each episode',
            outcome_conditioned_sampling=False,missing_contexts=deepcopy(self.unavailable))
        h=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return row|dict(schedule_sha256=h)

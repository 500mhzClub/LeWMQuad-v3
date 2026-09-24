"""Explicit pulse dataset joins and layout/action-balanced development schedules.

No directory discovery, fitting, role assignment or final-test access. Callers
must bind/audit packets, indices, labels and a prospective layout-role manifest.
A layout identity must group topology-equivalent trajectories across appearance,
start and friction changes; this interface cannot verify a publisher's geometry.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
from lewm.coupled_pulse_rollout_development import COMMANDS
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan,validate_timed_plan
from lewm.pulse_timed_observation_pairing_development import observation_pair_tensors
from lewm.pulse_timed_learning_development import join_sample

ROLES=('train','selection','development_eval')


def key(row):
    return row['condition'],row['departure_tick'],row['decision_ns']


def decode_targets(window,row):
    if row.get('target_only') is not True or key(row)!=key(window) or row['action_index']!=window['action_index']:
        raise ValueError('exact target-only pulse identity required')
    blocks,mask=pulse_brake_plan(tuple(window['command']),window['pulse_ticks'])
    active,offsets=validate_timed_plan(blocks[None],mask[None],1)
    if len(row['targets'])!=8 or len(window['targets'])!=8:raise ValueError('eight target slots required')
    motion=torch.full((8,3),float('nan'));contact=torch.full((8,),float('nan'))
    mv=torch.zeros(8,dtype=torch.bool);cv=mv.clone()
    for i,(t,w) in enumerate(zip(row['targets'],window['targets'],strict=True)):
        if (type(t['motion_valid']) is not bool or type(t['contact_valid']) is not bool
                or type(t['image_target_valid']) is not bool
                or t['offset_ns']!=int(offsets[0,i]) or w['offset_ns']!=t['offset_ns']
                or t['image_target_valid']!=w['future_valid']):
            raise ValueError('exact target clocks and independent masks required')
        m,c=t['motion_valid'],t['contact_valid']
        if (m or c or t['image_target_valid']) and not active[0,i]:raise ValueError('target beyond known plan')
        if m and (not c or t['contact']!=0):raise ValueError('motion must be observed and collision-free')
        if m:
            value=np.asarray(t['motion'],float)
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('finite motion label required')
            motion[i]=torch.tensor(value,dtype=torch.float32)
        elif t['motion'] is not None:raise ValueError('unobserved motion must remain null')
        if c:
            if isinstance(t['contact'],bool) or t['contact'] not in (0.,1.):raise ValueError('binary observed contact required')
            contact[i]=float(t['contact'])
        elif t['contact'] is not None:raise ValueError('unobserved contact must remain null')
        mv[i],cv[i]=m,c
    return dict(motion=motion,contact=contact,motion_valid=mv,contact_valid=cv,
        target_offsets_ns=offsets[0],target_only=True)


class PulseTimedDataset:
    def __init__(self,windows,targets,episode_roles):
        if not isinstance(windows,list) or not windows or not isinstance(targets,list):
            raise ValueError('nonempty explicit window and target lists required')
        self.windows=deepcopy(windows);self.episode_roles=deepcopy(episode_roles)
        labels={key(r):r for r in targets}
        if len(labels)!=len(targets) or len({key(w) for w in windows})!=len(windows):
            raise ValueError('duplicate pulse identities')
        if set(labels)!={key(w) for w in windows}:raise ValueError('complete one-to-one labels required')
        if set(episode_roles)!={w['condition'] for w in windows}:raise ValueError('exact episode-role inventory required')
        layout_roles={}
        for meta in self.episode_roles.values():
            if set(meta)!={'layout_id','role'} or meta['role'] not in ROLES:
                raise ValueError('explicit development role required')
            layout=meta['layout_id']
            if not isinstance(layout,str) or not layout:raise ValueError('nonempty publisher-bound layout identity required')
            if layout in layout_roles and layout_roles[layout]!=meta['role']:
                raise ValueError('one layout cannot cross train/selection/evaluation roles')
            layout_roles[layout]=meta['role']
        self._targets=[];self.groups={r:{} for r in ROLES};self.excluded=[]
        for layout,role in layout_roles.items():self.groups[role][layout]={}
        for i,w in enumerate(self.windows):
            action=w['action_index']
            if type(action) is not int or not 0<=action<6:raise ValueError('six measured action-duration cells required')
            if tuple(w['command'])!=COMMANDS[action//2] or w['pulse_ticks']!=(2,5)[action%2]:
                raise ValueError('action index does not identify pulse command/duration')
            if (type(w['departure_tick']) is not int or w['departure_tick']<0
                    or w['decision_ns']!=1_500_000_000+100_000_000*w['departure_tick']):
                raise ValueError('actual co-timed pulse departure required')
            indices=w['history_observation_indices']
            if (type(w['history_ready']) is not bool or len(indices)!=4
                    or w['history_ready']!=all(v is not None for v in indices)
                    or any(v is not None and (type(v) is not int or v<0) for v in indices)):
                raise ValueError('explicit complete or missing history required')
            self._targets.append(decode_targets(w,labels[key(w)]))
            if not w['history_ready']:
                self.excluded.append(dict(index=i,reason='MISSING_PAST_OBSERVATION'));continue
            meta=self.episode_roles[w['condition']]
            self.groups[meta['role']].setdefault(meta['layout_id'],{}).setdefault(action,[]).append(i)

    def __len__(self):return len(self.windows)

    def coverage(self,role):
        if role not in ROLES:raise ValueError('explicit development role required')
        return {layout:{str(a):len(cells.get(a,[])) for a in range(6)}
                for layout,cells in sorted(self.groups[role].items())}

    def schedule(self,role,*,updates,batch_size,seed,require_all_actions=True):
        if (role not in ROLES or any(type(v) is not int or v<=0 for v in (updates,batch_size))
                or type(seed) is not int or seed<0 or type(require_all_actions) is not bool):
            raise ValueError('explicit positive schedule dimensions and seed required')
        groups=self.groups[role]
        if not groups or any(not cells for cells in groups.values()):
            raise ValueError('no eligible samples for a declared layout in requested role')
        if require_all_actions and any(set(c)!=set(range(6)) for c in groups.values()):
            raise ValueError('missing action-duration cells; balanced collection required')
        rng=np.random.default_rng(seed);layout_order=[];cell_orders={};index_orders={};flat=[]
        for _ in range(updates*batch_size):
            if not layout_order:layout_order=list(rng.permutation(sorted(groups)))
            layout=layout_order.pop();cells=groups[layout]
            if not cell_orders.get(layout):cell_orders[layout]=list(rng.permutation(sorted(cells)))
            action=int(cell_orders[layout].pop());identity=(layout,action)
            if not index_orders.get(identity):index_orders[identity]=list(rng.permutation(cells[action]))
            flat.append(int(index_orders[identity].pop()))
        batches=[flat[i:i+batch_size] for i in range(0,len(flat),batch_size)]
        identity=dict(role=role,seed=seed,updates=updates,batch_size=batch_size,
            require_all_actions=require_all_actions,batches=batches,
            pulse_identities=[list(key(w)) for w in self.windows],episode_roles=self.episode_roles)
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return identity|dict(schedule_sha256=digest,outcome_conditioned_sampling=False,
            layout_geometry_verified_by_interface=False)

    def sample(self,index,readers):
        if type(index) is not int or not 0<=index<len(self):raise ValueError('explicit dataset index required')
        window=self.windows[index]
        pair=observation_pair_tensors(readers[window['condition']],window)
        # Never expose cached target tensors to mutation through a returned batch.
        native={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in self._targets[index].items()}
        return join_sample(pair,native)


def stack_samples(samples):
    if not samples:raise ValueError('nonempty batch required')
    def stack(rows):
        if isinstance(rows[0],dict):
            if any(set(r)!=set(rows[0]) for r in rows):raise ValueError('identical batch schemas required')
            return {k:stack([r[k] for r in rows]) for k in rows[0]}
        return torch.stack(rows)
    return stack(samples)

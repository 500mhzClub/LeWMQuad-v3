"""Exact inventory/role joins and paired layout-first pulse prediction scoring.

No file access, inference, training schedule choice, checkpoint selection or
navigation claim. Callers must supply terminal-audited, visibility-valid data.
Missing planned episodes remain explicit; frame counts are not independent N.
"""
from collections import Counter
from copy import deepcopy
import re
import numpy as np
from lewm.independent_layout_collection_development import CollectionInventory
from lewm.pulse_timed_dataset_development import PulseTimedDataset,ROLES
from lewm.pulse_action_time_baseline_development import ActionTimeMean
from lewm.temporal_prediction_metrics_development import reduce_predictions


class IndependentPulseEvaluation:
    def __init__(self,inventory,dataset):
        if not isinstance(inventory,CollectionInventory) or not isinstance(dataset,PulseTimedDataset):
            raise ValueError('validated frozen inventory and pulse dataset required')
        self.inventory=deepcopy(inventory);self.dataset=deepcopy(dataset)
        conditions=[w['condition'] for w in dataset.windows]
        if len(conditions)!=len(set(conditions)):
            raise ValueError('one prescribed departure per inventory episode required')
        for w in dataset.windows:
            c=w['condition']
            if c not in inventory.episodes:raise ValueError('episode outside prospective inventory')
            e=inventory.episodes[c]
            if dataset.episode_roles[c]!=dict(layout_id=e['layout_id'],role=e['role']):
                raise ValueError('dataset cannot reassign frozen layout/role')
            if w['departure_tick']!=8 or w['decision_ns']!=2_300_000_000 or w['action_index']!=e['action_index']:
                raise ValueError('exact prescribed action/departure required')

    def population(self,role):
        if role not in ROLES:raise ValueError('explicit development role required')
        planned=[e for e in self.inventory.episodes.values() if e['role']==role]
        planned_ids={e['episode_id'] for e in planned}
        present={w['condition'] for w in self.dataset.windows if w['condition'] in planned_ids}
        eligible={w['condition'] for w in self.dataset.windows if w['condition'] in planned_ids and w['history_ready']}
        layouts=sorted({e['layout_id'] for e in planned})
        counts={k:dict(planned=sum(e['layout_id']==k for e in planned),
            present=sum(self.inventory.episodes[c]['layout_id']==k for c in present),
            eligible=sum(self.inventory.episodes[c]['layout_id']==k for c in eligible)) for k in layouts}
        return dict(role=role,planned_episodes=len(planned_ids),present_episodes=len(present),eligible_episodes=len(eligible),
            absent_from_dataset=sorted(planned_ids-present),missing_history=sorted(present-eligible),layouts=counts,
            all_planned_episodes_eligible=planned_ids==eligible,
            absent_episode_outcomes='unknown here; consult terminal collection audit, never assume collision-free',
            source_artifacts_verified_by_interface=False,visibility_verified_by_interface=False)

    def arrays(self,role):
        self.population(role)
        indices=[i for i,w in enumerate(self.dataset.windows)
            if self.dataset.episode_roles[w['condition']]['role']==role and w['history_ready']]
        if not indices:raise ValueError('no eligible samples in requested role')
        windows=[self.dataset.windows[i] for i in indices]
        metadata=[dict(condition=w['condition'],**{k:self.inventory.episodes[w['condition']][k]
            for k in ('layout_id','role','action_index','context_kind','history_kind','support')}) for w in windows]
        targets={k:np.stack([self.dataset._targets[i][k].detach().cpu().numpy().copy() for i in indices])
            for k in ('motion','motion_valid','contact','contact_valid')}
        offsets=np.stack([self.dataset._targets[i]['target_offsets_ns'].detach().cpu().numpy().copy() for i in indices])
        return dict(indices=np.asarray(indices,dtype=np.int64),metadata=metadata,targets=targets,
            actions=np.asarray([w['action_index'] for w in windows],dtype=np.int64),offsets_ns=offsets,active=offsets>0)

    def fit_action_time(self,training_draw_indices):
        """Fit the baseline on exactly the model's train-role exposure schedule."""
        draws=np.asarray(training_draw_indices)
        if draws.ndim!=1 or draws.dtype.kind not in 'iu' or not len(draws):
            raise ValueError('nonempty explicit integer training exposure indices required')
        train=self.arrays('train');lookup={int(v):i for i,v in enumerate(train['indices'])}
        if any(int(i) not in lookup for i in draws):
            raise ValueError('every draw must be an eligible train-role window; no evaluation-label fit')
        rows=np.asarray([lookup[int(i)] for i in draws],dtype=np.int64)
        model=ActionTimeMean.fit(train['actions'][rows],train['offsets_ns'][rows],train['active'][rows],
            {k:v[rows] for k,v in train['targets'].items()},roles=['train']*len(rows))
        return model,dict(draws=len(draws),distinct_windows=len(set(draws.tolist())),
            layout_draw_counts=dict(Counter(train['metadata'][i]['layout_id'] for i in rows)),
            training_draw_indices=draws.tolist(),evaluation_targets_used=False)

    def compare(self,heads,*,role):
        """All heads must cover identical ordered rows and every known plan slot.

        Unavailable head outputs are reported, never compared on a smaller
        favorable population. Truth/mask corruption remains a hard error.
        Per-layout paired differences are descriptive; no frame-level CI/p-value.
        """
        if not isinstance(heads,dict) or not heads or any(not isinstance(k,str) or re.fullmatch('[a-z][a-z0-9_]*',k) is None for k in heads):
            raise ValueError('named matched prediction heads required')
        data=self.arrays(role);active=data['active'];meta=data['metadata'];y=data['targets'];scores={};unavailable={}
        for name,entry in heads.items():
            if not isinstance(entry,dict) or set(entry)!={'indices','prediction'}:
                raise ValueError('explicit row-index binding and prediction array required')
            indices=np.asarray(entry['indices'])
            if indices.dtype.kind not in 'iu' or not np.array_equal(indices,data['indices']):
                raise ValueError('same exact ordered evaluation row identities required')
            p=np.asarray(entry['prediction'],dtype=float)
            if p.shape!=(*active.shape,5):raise ValueError('five-component outcome on eight exact-time slots required')
            missing=active&~np.isfinite(p).all(-1)
            if missing.any():
                unavailable[name]=dict(reason='MISSING_PREDICTIONS_NO_PAIRED_SCORE',
                    missing_known_cells=int(missing.sum()),cells=np.argwhere(missing).tolist())
                continue
            def score(rows=None,slots=None):return reduce_predictions(p,y,meta,active,row_selection=rows,horizon_selection=slots)
            scores[name]=dict(all=score(),
                by_actual_offset_ns={str(int(t)):score(slots=data['offsets_ns']==t) for t in sorted(set(data['offsets_ns'][active]))},
                strata={field:{str(value):score(rows=np.array([r[field]==value for r in meta]))
                    for value in sorted({r[field] for r in meta})}
                    for field in ('action_index','context_kind','history_kind','support')})
        pairs={}
        for left in sorted(scores):
            for right in sorted(scores):
                if left>=right:continue
                a=scores[left]['all'];b=scores[right]['all'];rows=[]
                for x,z in zip(a['layouts'],b['layouts'],strict=True):
                    if x['layout_id']!=z['layout_id']:raise ValueError('paired layout identity mismatch')
                    rows.append(dict(layout_id=x['layout_id'],**{metric:
                        x[metric]-z[metric] if x[metric] is not None and z[metric] is not None else None
                        for metric in ('position_error_m','yaw_error_rad','contact_brier')}))
                pairs[left+' minus '+right]=dict(layout_differences=rows,
                    macro_difference={metric:float(np.mean([r[metric] for r in rows if r[metric] is not None]))
                        if any(r[metric] is not None for r in rows) else None
                        for metric in ('position_error_m','yaw_error_rad','contact_brier')},
                    paired_layouts=len(rows),lower_is_better=True,confidence_interval=None,
                    uncertainty_note='layout-clustered descriptive comparison; correlated windows/horizons are not independent replicates')
        return dict(role=role,resubstitution=role=='train',population=self.population(role),
            metrics=scores,unavailable_heads=unavailable,paired_comparisons=pairs,
            all_requested_heads_comparable=not unavailable,scored_row_indices=data['indices'].tolist(),
            independent_layout_units=len({m['layout_id'] for m in meta}),final_evaluation=False,
            checkpoint_selection_performed=False,navigation_qualified=False,goal_achieved=False)

"""Mask-aware, layout-first temporal evaluation and training-only simple controls."""
import math

import numpy as np

from lewm.counterfactual_decision_diagnostic_development import decision_rows


def _mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def reduce_predictions(prediction,targets,metadata,active,row_selection=None,horizon_selection=None):
    prediction=np.asarray(prediction,dtype=float); active=np.asarray(active)
    n=len(metadata); shape=(n,8)
    if prediction.shape!=(*shape,5) or active.shape!=shape or active.dtype!=bool:
        raise ValueError('prediction/mask shape or type')
    mv=np.asarray(targets['motion_valid']); cv=np.asarray(targets['contact_valid'])
    motion=np.asarray(targets['motion']); contact=np.asarray(targets['contact'])
    if (mv.shape!=shape or cv.shape!=shape or mv.dtype!=bool or cv.dtype!=bool
            or motion.shape!=(*shape,3) or contact.shape!=shape or ((mv|cv)&~active).any()):
        raise ValueError('target/mask contract')
    if not np.isfinite(prediction[active]).all() or not np.isfinite(motion[mv]).all() or not np.isin(contact[cv],[0,1]).all():
        raise ValueError('nonfinite or invalid observed value')
    selected=active.copy()
    if row_selection is not None:
        rows=np.asarray(row_selection)
        if rows.shape!=(n,) or rows.dtype!=bool: raise ValueError('row selection shape/type')
        selected&=rows[:,None]
    if horizon_selection is not None:
        mask=np.asarray(horizon_selection)
        if mask.shape!=shape or mask.dtype!=bool: raise ValueError('horizon selection shape/type')
        selected&=mask
    mv=mv&selected; cv=cv&selected
    p=1/(1+np.exp(-np.clip(prediction[...,4],-60,60)))
    angle=np.arctan2(prediction[...,2],prediction[...,3]); rows=[]
    for layout in sorted({r['layout_id'] for r in metadata}):
        indices=np.array([i for i,r in enumerate(metadata) if r['layout_id']==layout],dtype=int)
        m,c=mv[indices],cv[indices]; difference=angle[indices][m]-motion[indices,:,2][m]
        paired=selected[indices,1:]&selected[indices,:-1]
        rows.append({'layout_id':layout,'windows':int(selected[indices].any(1).sum()),
            'motion_count':int(m.sum()),'contact_count':int(c.sum()),'contact_positives':int(contact[indices][c].sum()),
            'position_error_m':float(np.linalg.norm(prediction[indices,:,:2][m]-motion[indices,:,:2][m],axis=-1).mean()) if m.any() else None,
            'yaw_error_rad':float(np.abs(np.arctan2(np.sin(difference),np.cos(difference))).mean()) if m.any() else None,
            'contact_brier':float(((p[indices][c]-contact[indices][c])**2).mean()) if c.any() else None,
            'contact_accuracy_at_half':float(((p[indices][c]>=.5)==contact[indices][c]).mean()) if c.any() else None,
            'monotonicity_pair_count':int(paired.sum()),
            'contact_monotonicity_violation_fraction':float((np.diff(p[indices],axis=1)[paired]<-1e-6).mean()) if paired.any() else None})
    metrics=('position_error_m','yaw_error_rad','contact_brier','contact_accuracy_at_half','contact_monotonicity_violation_fraction')
    return {'layout_macro':{k:_mean([r[k] for r in rows]) for k in metrics},'layouts':rows,
        'windows':int(selected.any(1).sum()),'known_horizons':int(selected.sum()),'motion_valid':int(mv.sum()),
        'contact_valid':int(cv.sum()),'contact_positives':int(contact[cv].sum()),
        'contributing_layouts':{k:sum(r[k] is not None for r in rows) for k in metrics}}


def prediction_report(prediction,batch,eligible=None):
    metadata=batch['metadata']; active=np.asarray(batch['known_action_valid']).all(-1)
    offsets=np.array([m['offset_ns'] for m in metadata]); n=len(metadata)
    base=np.ones(n,dtype=bool) if eligible is None else np.asarray(eligible,dtype=bool)
    def score(rows=None,horizons=None):
        return reduce_predictions(prediction,batch['targets'],metadata,active,
            base if rows is None else base&rows,horizons)
    last=np.zeros_like(active)
    if n: last[np.arange(n),active.sum(1)-1]=True
    return {'all':score(),'later':score(offsets>0),'initial':score(offsets==0),
        'last_known':score(horizons=last),
        'by_offset_seconds':{str(offset/1e9):score(offsets==offset) for offset in range(0,4_000_000_000,500_000_000)},
        'by_horizon_seconds':{str((h+1)*.5):score(horizons=np.broadcast_to(np.arange(8)==h,(n,8))) for h in range(8)},
        'diagnostic_eligible_windows':int(base.sum()),'diagnostic_omitted_windows':int((~base).sum())}


def shuffle_population(metadata):
    """All-layout complete action/offset cells only; cyclic sorted-layout donors."""
    layouts=sorted({m['layout_id'] for m in metadata})
    lookup={(m['layout_id'],m['action_index'],m['offset_ns']):i for i,m in enumerate(metadata)}
    if len(lookup)!=len(metadata): raise ValueError('duplicate shuffle identity')
    donors=np.arange(len(metadata)); eligible=np.zeros(len(metadata),dtype=bool)
    if len(layouts)<2: return donors,eligible
    for i,m in enumerate(metadata):
        if not all((layout,m['action_index'],m['offset_ns']) in lookup for layout in layouts): continue
        next_layout=layouts[(layouts.index(m['layout_id'])+1)%len(layouts)]
        donors[i]=lookup[(next_layout,m['action_index'],m['offset_ns'])]; eligible[i]=True
    return donors,eligible


def initial_decisions(prediction,batch):
    indices=[i for i,m in enumerate(batch['metadata']) if m['offset_ns']==0]
    targets={k:np.asarray(batch['targets'][k])[indices] for k in ('motion','contact','motion_valid','contact_valid')}
    return decision_rows(np.asarray(prediction)[indices],targets,[batch['metadata'][i] for i in indices])


def simple_predictions(train,evaluation):
    """No evaluation outcomes used. Empirical fallback stays training-only."""
    if any(m['data_role']!='train' for m in train['metadata']): raise ValueError('empirical controls require training role')
    meta=evaluation['metadata']; n=len(meta); active=np.asarray(evaluation['known_action_valid']).all(-1)
    predictions={name:np.zeros((n,8,5)) for name in ('training_action_remaining_mean','zero_motion_no_contact','command_kinematics_no_contact')}
    for v in predictions.values(): v[...,3]=1.; v[...,4]=-30.
    targets=train['targets']; tm=np.asarray(targets['motion']); tc=np.asarray(targets['contact'])
    mv=np.asarray(targets['motion_valid']); cv=np.asarray(targets['contact_valid'])
    fallback={'motion_cells':0,'contact_cells':0,'evaluated_known_cells':int(active.sum())}
    def mean_by_layout(values,selection):
        return np.mean([values[[j for j,m in enumerate(train['metadata']) if selection[j] and m['layout_id']==layout]].mean(0)
            for layout in sorted({m['layout_id'] for j,m in enumerate(train['metadata']) if selection[j]})],axis=0)
    for i,m in enumerate(meta):
        action=np.array([t['action_index']==m['action_index'] for t in train['metadata']])
        cell=action&np.array([t['offset_ns']==m['offset_ns'] for t in train['metadata']])
        for h in np.flatnonzero(active[i]):
            motion_sel=cell&mv[:,h]; contact_sel=cell&cv[:,h]
            if not motion_sel.any(): motion_sel=action&mv[:,h]; fallback['motion_cells']+=1
            if not contact_sel.any(): contact_sel=action&cv[:,h]; fallback['contact_cells']+=1
            if not motion_sel.any() or not contact_sel.any(): raise ValueError('training action/horizon fallback unavailable')
            values=np.zeros((len(tm),4)); values[mv[:,h],:2]=tm[mv[:,h],h,:2]
            values[mv[:,h],2]=np.sin(tm[mv[:,h],h,2]); values[mv[:,h],3]=np.cos(tm[mv[:,h],h,2])
            predictions['training_action_remaining_mean'][i,h,:4]=mean_by_layout(values,motion_sel)
            probability=float(mean_by_layout(tc[:,h],contact_sel)); probability=np.clip(probability,1e-6,1-1e-6)
            predictions['training_action_remaining_mean'][i,h,4]=math.log(probability/(1-probability))
        xy=np.zeros(2); yaw=0.
        plan=np.asarray(evaluation['known_action_blocks'][i]).reshape(40,3)*[.3,1.,.5]
        valid=np.asarray(evaluation['known_action_valid'][i]).reshape(40)
        for tick,(vx,vy,w) in enumerate(plan):
            if not valid[tick]: break
            if abs(w)<1e-8: delta=np.array([vx,vy])*.1
            else:
                a=w*.1; delta=np.array([math.sin(a)*vx-(1-math.cos(a))*vy,(1-math.cos(a))*vx+math.sin(a)*vy])/w
            xy+=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])@delta; yaw+=w*.1
            if (tick+1)%5==0: predictions['command_kinematics_no_contact'][i,tick//5,:4]=[*xy,math.sin(yaw),math.cos(yaw)]
    for v in predictions.values(): v[~active]=0.
    return predictions,fallback

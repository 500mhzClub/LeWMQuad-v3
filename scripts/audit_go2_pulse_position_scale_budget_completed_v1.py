import json,math,hashlib
import numpy as np
import torch
from scripts.run_go2_pulse_position_scale_budget_v1 import OUTPUT,SEEDS,ARMS,OBJECTIVES,DATA,PAIRING,LABELS,PILOT,ROOT,digest
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from lewm.pulse_position_scale_learning_development import PositionScaleTrainer
from lewm.pulse_timed_training_runner_development import state_digest
torch.set_num_threads(1)
def read(p):return json.loads(p.read_text())
launch=read(OUTPUT/'launch.json');verify_bindings(launch['source_sha256']|launch['input_sha256'])
result=read(OUTPUT/'result.json');assert result['status']=='COMPLETE' and result['total_optimizer_steps']==2160
assert digest(OUTPUT/'launch.json')==result['launch_sha256']
assert digest(OUTPUT/'schedule.json')==result['schedule_sha256']
schedule=read(OUTPUT/'schedule.json');assert schedule['repetitions']==10 and schedule['total_draws']==720
old_schedules=read(DATA/'schedules.json')
assert all(schedule['base']==old_schedules[a] for a in ARMS)
windows=read(PAIRING/'windows.json');raw=read(LABELS/'targets.json')
lookup={(r['condition'],r['departure_tick'],r['decision_ns']):r for r in raw}
targets=[lookup[(w['condition'],w['departure_tick'],w['decision_ns'])] for w in windows]
mv=np.array([[t['motion_valid'] for t in r['targets']] for r in targets])
cv=np.array([[t['contact_valid'] for t in r['targets']] for r in targets])
motion=np.array([[t['motion'] if t['motion_valid'] else [np.nan]*3 for t in r['targets']] for r in targets],np.float32)
contact=np.array([[t['contact'] if t['contact_valid'] else np.nan for t in r['targets']] for r in targets],np.float32)
offsets=np.array([[min((h+1)*5,w['pulse_ticks']+20)*100_000_000 if h*5<w['pulse_ticks']+20 else 0 for h in range(8)] for w in windows],np.int64)
active=offsets>0
actions=np.array([w['action_index'] for w in windows]);conditions=np.array([w['condition'] for w in windows])
assert len(windows)==185 and mv.sum()==917 and cv.sum()==917 and contact[cv].sum()==0
prior=read(PILOT/'result.json')
expected_names={f'{s}_{a}_{o}' for s in SEEDS for a in ARMS for o in OBJECTIVES}
assert set(result['fits'])==expected_names and set(result['result_sha256'])==expected_names
bindings={};inventory={};summary=[]
def bind(path,expected=None):
    h=digest(path)
    if expected is not None:assert h==expected,str(path)
    name=str(path.relative_to(ROOT));inventory[name]=h
    return h
for n in ('launch.json','result.json','schedule.json'):bindings[str((OUTPUT/n).relative_to(ROOT))]=bind(OUTPUT/n)
def independent_score(p,rows=None,horizons=None):
    p=np.asarray(p,float);selected=active.copy()
    if rows is not None:selected &= rows[:,None]
    if horizons is not None:selected &= horizons
    m=mv&selected;c=cv&selected
    angle=np.arctan2(p[...,2],p[...,3]);diff=angle[m]-motion[...,2][m]
    probability=1/(1+np.exp(-np.clip(p[...,4],-60,60)))
    paired=selected[:,1:]&selected[:,:-1]
    metrics=dict(position_error_m=float(np.linalg.norm(p[...,:2][m]-motion[...,:2][m],axis=-1).mean()) if m.any() else None,
      yaw_error_rad=float(np.abs(np.arctan2(np.sin(diff),np.cos(diff))).mean()) if m.any() else None,
      contact_brier=float(((probability[c]-contact[c])**2).mean()) if c.any() else None,
      contact_accuracy_at_half=float(((probability[c]>=.5)==contact[c]).mean()) if c.any() else None,
      contact_monotonicity_violation_fraction=float((np.diff(probability,axis=1)[paired]<-1e-6).mean()) if paired.any() else None)
    return metrics,dict(windows=int(selected.any(1).sum()),known_horizons=int(selected.sum()),motion_valid=int(m.sum()),contact_valid=int(c.sum()),contact_positives=int(contact[c].sum()))
def check_score(p,reported,rows=None,horizons=None):
    calculated,counts=independent_score(p,rows,horizons)
    for k,x in calculated.items():
        y=reported['layout_macro'][k]
        if x is None:assert y is None
        else:assert math.isclose(x,y,rel_tol=1e-11,abs_tol=1e-12),(k,x,y)
    for k,x in counts.items():assert reported[k]==x,(k,x,reported[k])
    assert len(reported['layouts'])==1
for name in sorted(expected_names):
    fit=result['fits'][name];directory=OUTPUT/name
    bind(OUTPUT/(name+'_result.json'),result['result_sha256'][name])
    assert read(OUTPUT/(name+'_result.json'))==fit
    assert fit['optimizer_steps']==120 and set(fit['snapshots'])=={'12','120'}
    old=prior['fits'][str(fit['seed'])+'_'+fit['arm']]
    assert fit['initial_model_sha256']==old['initial_model_sha256']
    updates=read(directory/'updates.json');bind(directory/'updates.json',fit['updates_sha256'])
    assert len(updates)==120
    for j,row in enumerate(updates):
        assert row['update']==j+1 and row['sample_indices']==schedule['base']['batches'][j%12]
        assert row['schedule_cycle']==j//12 and row['schedule_sha256']==schedule['base']['schedule_sha256']
        assert row['objective']==fit['objective']
        path=directory/('update_%04d.json'%(j+1));assert read(path)==row;bind(path)
    for step,snap in fit['snapshots'].items():
        n=int(step);prefix='snapshot_%04d'%n
        for file,h in snap['artifact_sha256'].items():bind(directory/file,h)
        checkpoint=torch.load(directory/(prefix+'.pt'),map_location='cpu',weights_only=True)
        assert checkpoint['updates']==n and not checkpoint['failed']
        assert checkpoint['condition']==fit['arm'] and checkpoint['objective']==fit['objective'] and checkpoint['seed']==fit['seed']
        assert checkpoint['position_loss_scale_m']==(1. if fit['objective']=='raw' else .06)
        assert checkpoint['schedule_sha256']==schedule['base']['schedule_sha256']
        trainer=PositionScaleTrainer(fit['arm'],objective=fit['objective'],seed=fit['seed'])
        assert trainer.initial_sha256==fit['initial_model_sha256']
        trainer.model.load_state_dict(checkpoint['model_state']);trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        assert state_digest(trainer.model.state_dict())==checkpoint['model_sha256']==snap['model_sha256']==updates[n-1]['model_sha256']
        assert len(trainer.optimizer.state)==len(trainer.parameters)
        assert all(int(s['step'])==n for s in trainer.optimizer.state.values())
        if fit['objective']=='raw' and n==12:assert checkpoint['model_sha256']==old['final_model_sha256']
        metrics=read(directory/(prefix+'_metrics.json'));assert metrics==snap['metrics']
        assert metrics['model_sha256']==checkpoint['model_sha256']
        with np.load(directory/(prefix+'_predictions.npz'),allow_pickle=False) as saved:
            arrays={k:saved[k] for k in saved.files}
        for k,value in dict(motion=motion,contact=contact,motion_valid=mv,contact_valid=cv,active=active,offsets_ns=offsets,actions=actions,indices=np.arange(185)).items():
            np.testing.assert_array_equal(arrays[k],value)
        heads=['direct_outcomes'] if fit['arm']=='direct' else ['direct_outcomes','rollout_outcomes']
        assert set(metrics['metrics'])==set(heads)
        record=dict(seed=fit['seed'],arm=fit['arm'],objective=fit['objective'],update=n,heads={})
        for head in heads:
            p=arrays[head];assert p.shape==(185,8,5) and np.isfinite(p[active]).all()
            report=metrics['metrics'][head];check_score(p,report['all'])
            for action in range(6):check_score(p,report['by_action'][str(action)],rows=actions==action)
            for condition in sorted(set(conditions)):check_score(p,report['by_condition'][condition],rows=conditions==condition)
            for ns in sorted(set(offsets[active])):check_score(p,report['by_actual_offset_ns'][str(ns)],horizons=offsets==ns)
            record['heads'][head]=dict(all=report['all']['layout_macro'],
                by_condition={c:r['layout_macro'] for c,r in report['by_condition'].items()})
        zero=np.zeros((185,8,5));zero[...,3]=1.;zero[...,4]=-30.
        check_score(zero,metrics['zero_motion_no_contact']['all'])
        summary.append(record)
verify_bindings(launch['source_sha256']|launch['input_sha256'])
size=sum((ROOT/p).stat().st_size for p in inventory)
assert len(inventory)==2307
print(json.dumps(dict(status='VERIFIED',fits=18,optimizer_steps=2160,snapshots=36,raw12_predecessor_tensor_matches=9,
    explicit_output_files_verified=len(inventory),output_bytes=size,
    inventory_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    root_bindings=bindings,independent_prediction_metric_reconstruction=True,full_training_replay=False,
    summary=summary,goal_achieved=False)))

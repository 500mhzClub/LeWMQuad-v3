"""Eighteen fresh paired fits, two fixed snapshots; not independent navigation."""
import shutil
import time
import numpy as np
import torch
from lewm.pulse_position_scale_learning_development import PositionScaleTrainer,OBJECTIVES,POSITION_SCALE_M
from lewm.pulse_position_scale_evaluation_development import evaluate
from lewm.pulse_timed_training_runner_development import state_digest
from lewm.pulse_timed_dataset_development import PulseTimedDataset,stack_samples
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_pulse_training_pilot_v2 import OUTPUT as PILOT,SEEDS,ARMS
from scripts.check_go2_pulse_action_time_baseline_v1 import OUTPUT as BASELINE
from scripts.check_go2_pulse_loss_scale_v1 import OUTPUT as GRADIENT
from scripts.check_go2_pulse_dataset_v1 import OUTPUT as DATA
from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT as PAIRING,TRIALS
from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json

OUTPUT=ROOT/'.generated/go2_pulse_position_scale_budget_v1_attempt_001'
PROTOCOL='docs/go2_pulse_position_scale_budget_v1_2026-09-06.md'
RESERVE=10*1024**3
UPDATES=120
SNAPSHOTS=(12,120)


def preflight():
    ids={}
    for directory,values in (
        (PILOT,{'launch.json':'5059f97d20817d8781f1ecdabe10d811ad5e0741a2ca96861985a58a84e21984',
            'result.json':'ecfbd5b84ac65821134c28b7983230c8f9196dce92325c1e4b2f0c683e15e032'}),
        (BASELINE,{'launch.json':'cbed7efff0cc98643b232b86f13e113a9f855ede35941002dc649dcaaa6211f6',
            'result.json':'a03459e6b59f8bae3ac1ca1acd2c3964e6abb1484bd3949a60438e9888389d63'}),
        (GRADIENT,{'launch.json':'751c31fd4d82db77bbaf5f006b447d0dfe525ceca2faa2c06dc995aadb3fc5e8',
            'result.json':'2d39a27cc83ff461e08424a8092a75545b2bdf501474002627ec90afc5978a71'})):
        ids.update({str((directory/n).relative_to(ROOT)):h for n,h in values.items()})
    verify_bindings(ids);inherited={};inputs=ids.copy()
    for directory in (PILOT,BASELINE,GRADIENT):
        old=read_json(directory,'launch.json');verify_bindings(old['source_sha256']|old['input_sha256'])
        inherited.update(old['source_sha256']);inputs.update(old['input_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/run_go2_pulse_position_scale_budget_v1.py',
        'lewm/tests/test_pulse_position_scale_learning_development.py',
        'lewm/tests/test_pulse_position_scale_runner_development.py'),inherited)
    verify_bindings(sources|inputs)
    if shutil.disk_usage(ROOT).free<RESERVE+1024**3:raise ValueError('1GiB budget plus10GiB reserve required')
    return dict(source_sha256=sources,input_sha256=inputs,device='cpu',torch_version=str(torch.__version__),
        seeds=list(SEEDS),arms=list(ARMS),objectives=list(OBJECTIVES),position_scale_m=POSITION_SCALE_M,
        updates=UPDATES,snapshots=list(SNAPSHOTS),schedule_repetitions=10,batch_size=6,
        learning_rate=.001,weight_decay=0.,gradient_clip=1.,ema_momentum=.99,latent_dim=32,
        resubstitution=True,new_physics=False,navigation_qualified=False,goal_achieved=False)


def snapshot(directory,trainer,dataset,readers,schedule):
    step=trainer.updates;prefix='snapshot_%04d'%step
    checkpoint=trainer.checkpoint() | dict(schedule_sha256=schedule['schedule_sha256'],schedule_repetitions=10)
    with (directory/(prefix+'.pt')).open('xb') as stream:torch.save(checkpoint,stream)
    saved=torch.load(directory/(prefix+'.pt'),map_location='cpu',weights_only=True)
    clone=PositionScaleTrainer(trainer.condition,objective=trainer.objective,seed=trainer.seed,latent_dim=trainer.latent_dim)
    clone.model.load_state_dict(saved['model_state']);clone.optimizer.load_state_dict(saved['optimizer_state'])
    if state_digest(clone.model.state_dict())!=checkpoint['model_sha256']:raise ValueError('snapshot model identity')
    if len(clone.optimizer.state)!=len(clone.parameters) or any(int(s['step'])!=step for s in clone.optimizer.state.values()):
        raise ValueError('exact active optimizer step counts required')
    metrics,arrays=evaluate(clone,dataset,readers)
    write_json(directory/(prefix+'_metrics.json'),metrics)
    with (directory/(prefix+'_predictions.npz')).open('xb') as stream:np.savez_compressed(stream,**arrays)
    return dict(update=step,model_sha256=checkpoint['model_sha256'],
        artifact_sha256={prefix+suffix:digest(directory/(prefix+suffix))
            for suffix in ('.pt','_metrics.json','_predictions.npz')},metrics=metrics)


def run_fit(directory,seed,arm,objective,dataset,readers,schedule,batches,prior):
    directory.mkdir();trainer=PositionScaleTrainer(arm,objective=objective,seed=seed)
    if trainer.initial_sha256!=prior['initial_model_sha256']:raise ValueError('paired initial identity')
    logs=[];snapshots={};started=time.perf_counter()
    for step in range(UPDATES):
        if shutil.disk_usage(ROOT).free<RESERVE:raise ValueError('training storage reserve stop')
        index=step%len(batches);ids=schedule['batches'][index]
        row=trainer.step(batches[index]) | dict(sample_indices=ids,schedule_sha256=schedule['schedule_sha256'],
            schedule_cycle=step//len(batches),objective=objective)
        logs.append(row);write_json(directory/('update_%04d.json'%trainer.updates),row)
        if trainer.updates%12==0:print('SCALE_BUDGET',seed,arm,objective,trainer.updates,row['loss'],flush=True)
        if trainer.updates in SNAPSHOTS:
            if objective=='raw' and trainer.updates==12 and row['model_sha256']!=prior['final_model_sha256']:
                raise ValueError('raw12 did not reproduce frozen pilot tensor identity')
            snapshots[str(trainer.updates)]=snapshot(directory,trainer,dataset,readers,schedule)
    write_json(directory/'updates.json',logs)
    return dict(seed=seed,arm=arm,objective=objective,optimizer_steps=trainer.updates,
        initial_model_sha256=trainer.initial_sha256,snapshots=snapshots,
        updates_sha256=digest(directory/'updates.json'),wall_seconds=time.perf_counter()-started,
        raw12_predecessor_identity_verified=objective=='raw',resubstitution=True,navigation_qualified=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive fresh comparison; no implicit retry/resume')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);reports={}
    try:
        data=read_json(DATA,'result.json');schedules=read_json(DATA,'schedules.json');prior=read_json(PILOT,'result.json')
        dataset=PulseTimedDataset(read_json(PAIRING,'windows.json'),read_json(LABELS,'targets.json'),data['episode_roles'])
        schedule=dataset.schedule('train',updates=12,batch_size=6,seed=2026090711)
        if any(schedules[a]!=schedule for a in ARMS):raise ValueError('frozen matched schedule required')
        write_json(OUTPUT/'schedule.json',dict(base=schedule,repetitions=10,total_draws=720))
        readers={c:IntentReturnRGBDReplay(INPUT/c) for c in TRIALS}
        batches=[stack_samples([dataset.sample(i,readers) for i in ids]) for ids in schedule['batches']]
        for seed in SEEDS:
            for arm in ARMS:
                for objective in OBJECTIVES:
                    name=str(seed)+'_'+arm+'_'+objective
                    reports[name]=run_fit(OUTPUT/name,seed,arm,objective,dataset,readers,schedule,batches,prior['fits'][str(seed)+'_'+arm])
                    write_json(OUTPUT/(name+'_result.json'),reports[name])
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        write_json(OUTPUT/'result.json',dict(status='COMPLETE',fits=reports,total_optimizer_steps=sum(r['optimizer_steps'] for r in reports.values()),
            launch_sha256=digest(OUTPUT/'launch.json'),schedule_sha256=digest(OUTPUT/'schedule.json'),
            result_sha256={n:digest(OUTPUT/(n+'_result.json')) for n in reports},
            empirical_baseline=read_json(BASELINE,'result.json')['metrics'],posthoc=True,resubstitution=True,
            independent_layouts=1,evaluation_layouts=0,trained_navigation_policy=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAILURE',reason=repr(error),completed_fits=list(reports)))
        raise


if __name__=='__main__':main()

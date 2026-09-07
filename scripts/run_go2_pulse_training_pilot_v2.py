"""Nine fixed CPU model fits on one development room; no navigation promotion."""
import json
import shutil
import time
import torch
from lewm.pulse_timed_training_runner_development import PulseTrainer,evaluate,state_digest
from lewm.pulse_timed_dataset_development import PulseTimedDataset,stack_samples
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.check_go2_pulse_dataset_v1 import OUTPUT as DATA
from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT as PAIRING,TRIALS
from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json

OUTPUT=ROOT/'.generated/go2_pulse_training_pilot_v2_attempt_001'
PROTOCOL='docs/go2_pulse_training_pilot_v2_2026-09-06.md'
SEEDS=(2026090721,2026090722,2026090723)
ARMS=('direct','supervised_rollout','jepa')
RESERVE=10*1024**3


def preflight():
    ids={str((DATA/n).relative_to(ROOT)):h for n,h in {
        'launch.json':'6b21ead70f5bf98291ccc57640859843bfdfd6ac48ff1ffc75012862eff6b31e',
        'result.json':'293870ecbb04615029aef389a76fcc4c62477f0d1ffbb8ca2b649664a48cc02b',
        'schedules.json':'d6bd0ce92bfe31963abd92c215fe1908876ea6e006aac84dca582d1da93dad19'}.items()}
    verify_bindings(ids);old=read_json(DATA,'launch.json');verify_bindings(old['source_sha256']|old['input_sha256'])
    first=ROOT/'.generated/go2_pulse_training_pilot_v1_attempt_001'
    failure_ids={str((first/n).relative_to(ROOT)):h for n,h in {
        'launch.json':'0123e2c16588abfcf8c6c3cdf5f77161c75da3b2f2bc77d104887ff3933904bd',
        'failure.json':'68f4848b18a8569f9e811f71251addf131f78ae716fc59ac812fa8d817119070',
        '2026090721_direct/updates.json':'e4badfb60747b0cd7540480e76c3afd750bae0f43310c59bc7970f79e8480a50'}.items()}
    verify_bindings(failure_ids);first_launch=read_json(first,'launch.json')
    verify_bindings(first_launch['source_sha256']|first_launch['input_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/run_go2_pulse_training_pilot_v2.py',
        'lewm/tests/test_pulse_timed_training_runner_development.py',
        'lewm/tests/test_pulse_training_persistence_development.py'),first_launch['source_sha256'])
    inputs=old['input_sha256']|ids|failure_ids;verify_bindings(sources|inputs)
    if shutil.disk_usage(ROOT).free<RESERVE+1024**3:raise ValueError('1GiB pilot budget plus10GiB reserve required')
    return dict(source_sha256=sources,input_sha256=inputs,seeds=list(SEEDS),arms=list(ARMS),
        latent_dim=32,updates_per_fit=12,batch_size=6,learning_rate=.001,weight_decay=0.,
        gradient_clip_norm=1.,ema_momentum=.99,device='cpu',torch_version=str(torch.__version__),
        deterministic_algorithms=True,schedule_seed=2026090711,minimum_free_bytes=RESERVE,
        evaluation_role='train',checkpoint_selection=False,new_physics=False,navigation_qualified=False,goal_achieved=False)


def run_fit(seed,arm,dataset,readers,schedule):
    directory=OUTPUT/(str(seed)+'_'+arm);directory.mkdir()
    trainer=PulseTrainer(arm,seed=seed);initial=evaluate(trainer,dataset,readers,role='train')
    write_json(directory/'initial_metrics.json',initial);logs=[];started=time.perf_counter()
    for ids in schedule['batches']:
        if shutil.disk_usage(ROOT).free<RESERVE:raise ValueError('training storage reserve stop')
        batch=stack_samples([dataset.sample(i,readers) for i in ids])
        row=trainer.step(batch)|dict(sample_indices=ids,schedule_sha256=schedule['schedule_sha256'])
        logs.append(row);write_json(directory/('update_%04d.json'%trainer.updates),row)
        print('PULSE_TRAIN',seed,arm,row['update'],row['loss'],flush=True)
    elapsed=time.perf_counter()-started
    if trainer.updates!=12:raise ValueError('exact fixed optimizer budget required')
    write_json(directory/'updates.json',logs)
    checkpoint=trainer.checkpoint();checkpoint['schedule_sha256']=schedule['schedule_sha256']
    torch.save(checkpoint,directory/'final_checkpoint.pt')
    restored=torch.load(directory/'final_checkpoint.pt',map_location='cpu',weights_only=True)
    clone=PulseTrainer(arm,seed=seed);clone.model.load_state_dict(restored['model_state'])
    clone.optimizer.load_state_dict(restored['optimizer_state'])
    if state_digest(clone.model.state_dict())!=checkpoint['model_sha256']:raise ValueError('checkpoint model identity mismatch')
    # Score restored weights, not an unsaved transient model. This is not resume.
    final=evaluate(clone,dataset,readers,role='train');write_json(directory/'final_metrics.json',final)
    return dict(seed=seed,arm=arm,optimizer_steps=trainer.updates,initial_model_sha256=trainer.initial_sha256,
        final_model_sha256=checkpoint['model_sha256'],checkpoint_sha256=digest(directory/'final_checkpoint.pt'),
        initial_metrics_sha256=digest(directory/'initial_metrics.json'),final_metrics_sha256=digest(directory/'final_metrics.json'),
        updates_sha256=digest(directory/'updates.json'),schedule_sha256=schedule['schedule_sha256'],
        training_wall_seconds=elapsed,checkpoint_roundtrip_verified=True,resubstitution=True,
        selection_performed=False,navigation_qualified=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive training pilot; no implicit resume/retry')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);reports={}
    try:
        result=read_json(DATA,'result.json');schedules=read_json(DATA,'schedules.json')
        dataset=PulseTimedDataset(read_json(PAIRING,'windows.json'),read_json(LABELS,'targets.json'),result['episode_roles'])
        for arm in ARMS:
            actual=dataset.schedule('train',updates=12,batch_size=6,seed=2026090711)
            if actual!=schedules[arm]:raise ValueError('exact frozen matched schedule required')
        readers={c:IntentReturnRGBDReplay(INPUT/c) for c in TRIALS}
        for seed in SEEDS:
            initial=None
            for arm in ARMS:
                name=str(seed)+'_'+arm;reports[name]=run_fit(seed,arm,dataset,readers,schedules[arm])
                write_json(OUTPUT/(name+'_result.json'),reports[name])
                current=reports[name]['initial_model_sha256']
                if initial is None:initial=current
                elif current!=initial:raise ValueError('paired initial weights disagree')
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        write_json(OUTPUT/'result.json',dict(status='PULSE_TRAINING_PILOT_COMPLETE',fits=reports,
            result_sha256={name:digest(OUTPUT/(name+'_result.json')) for name in reports},
            total_optimizer_steps=sum(r['optimizer_steps'] for r in reports.values()),
            independent_layouts=1,selection_layouts=0,evaluation_layouts=0,
            trained_navigation_policy=False,independent_generalization_established=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_PULSE_TRAINING_PILOT_FAILURE',reason=repr(error),completed_fits=reports));raise


if __name__=='__main__':main()

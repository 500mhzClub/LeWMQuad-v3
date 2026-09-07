"""Read-only first-batch gradient diagnosis of all nine frozen pulse fits."""
import shutil
import torch
from lewm.pulse_loss_scale_diagnostic_development import diagnose
from lewm.pulse_timed_training_runner_development import PulseTrainer,state_digest
from lewm.pulse_timed_dataset_development import PulseTimedDataset,stack_samples
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_pulse_training_pilot_v2 import OUTPUT as PILOT,SEEDS,ARMS
from scripts.check_go2_pulse_dataset_v1 import OUTPUT as DATA
from scripts.check_go2_pulse_timed_pairing_v1 import INPUT,OUTPUT as PAIRING,TRIALS
from scripts.derive_go2_recorded_pulse_native_targets_v1 import OUTPUT as LABELS
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json

OUTPUT = ROOT/'.generated/go2_pulse_loss_scale_diagnostic_v1_attempt_001'
PROTOCOL = 'docs/go2_pulse_loss_scale_diagnostic_v1_2026-09-06.md'
RESERVE = 10*1024**3


def preflight():
    ids = {str((PILOT/n).relative_to(ROOT)):h for n,h in {
        'launch.json':'5059f97d20817d8781f1ecdabe10d811ad5e0741a2ca96861985a58a84e21984',
        'result.json':'ecfbd5b84ac65821134c28b7983230c8f9196dce92325c1e4b2f0c683e15e032'}.items()}
    verify_bindings(ids); old=read_json(PILOT,'launch.json'); result=read_json(PILOT,'result.json')
    verify_bindings(old['source_sha256'] | old['input_sha256'])
    expected={str(seed)+'_'+arm for seed in SEEDS for arm in ARMS}
    if set(result['fits'])!=expected: raise ValueError('exact nine frozen fits required')
    ids.update({str((PILOT/name/'final_checkpoint.pt').relative_to(ROOT)):r['checkpoint_sha256']
        for name,r in result['fits'].items()})
    sources=discover_sources((PROTOCOL,'scripts/check_go2_pulse_loss_scale_v1.py',
        'lewm/tests/test_pulse_loss_scale_diagnostic_development.py'),old['source_sha256'])
    inputs=old['input_sha256'] | ids; verify_bindings(sources | inputs)
    if shutil.disk_usage(ROOT).free<RESERVE+10*1024**2: raise ValueError('diagnostic storage reserve')
    return dict(source_sha256=sources,input_sha256=inputs,device='cpu',torch_version=str(torch.__version__),
        posthoc=True,first_batch_only=True,optimizer_steps=0,ema_updates=0,
        navigation_qualified=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive diagnostic; no overwrite/retry')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    reports={}
    try:
        data=read_json(DATA,'result.json'); schedules=read_json(DATA,'schedules.json')
        dataset=PulseTimedDataset(read_json(PAIRING,'windows.json'),read_json(LABELS,'targets.json'),data['episode_roles'])
        actual=dataset.schedule('train',updates=12,batch_size=6,seed=2026090711)
        if any(schedules[a]!=actual for a in ARMS): raise ValueError('exact matched schedule required')
        indices=actual['batches'][0]
        readers={c:IntentReturnRGBDReplay(INPUT/c) for c in TRIALS}
        batch=stack_samples([dataset.sample(i,readers) for i in indices])
        for seed in SEEDS:
            for arm in ARMS:
                name=str(seed)+'_'+arm
                saved=torch.load(PILOT/name/'final_checkpoint.pt',map_location='cpu',weights_only=True)
                if saved['failed'] or saved['updates']!=12 or saved['seed']!=seed or saved['condition']!=arm:
                    raise ValueError('frozen completed checkpoint required')
                for stage in ('initial','final'):
                    trainer=PulseTrainer(arm,seed=seed)
                    if trainer.initial_sha256!=saved['initial_sha256']: raise ValueError('initial identity')
                    if stage=='final': trainer.model.load_state_dict(saved['model_state'])
                    before=state_digest(trainer.model.state_dict())
                    expected=saved['initial_sha256'] if stage=='initial' else saved['model_sha256']
                    if before!=expected: raise ValueError('checkpoint identity')
                    row=diagnose(trainer.model,batch,arm)
                    if state_digest(trainer.model.state_dict())!=before or any(p.grad is not None for p in trainer.model.parameters()):
                        raise ValueError('diagnostic mutated model or parameter gradients')
                    row.update(seed=seed,arm=arm,stage=stage,model_sha256=before,
                        sample_indices=indices,schedule_sha256=actual['schedule_sha256'])
                    label=name+'_'+stage;write_json(OUTPUT/(label+'.json'),row);reports[label]=row
                    print('LOSS_SCALE',label,row['total_loss'],flush=True)
        verify_bindings(launch['source_sha256'] | launch['input_sha256'])
        write_json(OUTPUT/'result.json',dict(status='COMPLETE',diagnostics=reports,
            artifact_sha256={n+'.json':digest(OUTPUT/(n+'.json')) for n in reports},
            launch_sha256=digest(OUTPUT/'launch.json'),posthoc=True,first_batch_only=True,
            optimizer_steps=0,ema_updates=0,independent_generalization_established=False,
            navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FAILURE',reason=repr(error),completed=list(reports)))
        raise


if __name__=='__main__':main()

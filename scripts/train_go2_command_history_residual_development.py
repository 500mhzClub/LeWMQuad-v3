"""Fixed 1200-update matched JEPA/supervised corrections around command dynamics."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time
import cv2
import torch
from lewm.command_history_residual_learning_development import CommandHistoryResidualTrainer,REFERENCE
from scripts import command_history_residual_snapshot_development as snapshot
from scripts import train_go2_longer_residual_fit_development as previous

data=previous.data
BASE=previous.BASE
OUTPUT=BASE/'go2_command_history_residual_matched_fits_v1_attempt_001'
PLAN=Path('docs/go2_command_history_residual_plan_2026-09-17.json')
SCHEDULE=previous.SCHEDULE
CONDITIONS=('jepa','supervised_rollout')
UPDATES=1200
write=previous.write


def prepare():
    old=json.loads(previous.PLAN.read_text());schedule=json.loads(SCHEDULE.read_text())
    reference=json.loads((REFERENCE.parent/'result.json').read_text())
    assert reference['status']=='COMPLETE' and reference['training_contexts']==4694
    assert reference['schedule_sha256']==hashlib.sha256(SCHEDULE.read_bytes()).hexdigest()
    assert schedule['updates']==1200 and len(schedule['batches'])==1200 and schedule['batch_size']==6
    if OUTPUT.exists():raise ValueError('preserve every attempt')
    plan=dict(schema='command_history_residual_plan.v1',conditions=CONDITIONS,updates=UPDATES,
        seed=schedule['seed'],batch_size=6,training_contexts=4694,training_draws_per_condition=7200,
        schedule_path=str(SCHEDULE),schedule_sha256=hashlib.sha256(SCHEDULE.read_bytes()).hexdigest(),
        reference_path=str(REFERENCE),reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        old_models=old['old_models'],evaluation_roots=old['evaluation_roots'],
        change='replace ideal command integration reference with frozen training-only command-history fit',
        fixed_reference=True,reference_and_neural_fit_use_same_training_contexts=True,
        fresh_initialization=True,new_navigation_data_in_training=False,
        original_RGB_body_control_inputs_and_JEPA_objective_retained=True,
        model_conditions=['reference_only','jepa_reference_residual','supervised_reference_residual','jepa_1200','supervised_rollout_1200'],
        only_fixed_final_update_evaluated=True,checkpoint_selection=False,
        primary_evaluation='all 3726 matched executed windows from the same six exposed recordings; prefix/action/whole XY and yaw errors',
        next_navigation_criterion='consistent improvement over reference-only and original neural predictions, especially turn prefix/action errors; inspect failures before native execution',
        parallel_cpu_fits=True,threads_per_fit=1,native_jobs_during_training=False,
        limits=['hybrid motion reference plus learned residual, not pure JEPA dynamics',
                'exposed development trajectories, not independent generalization',
                'prediction improvements alone do not establish navigation benefit'],
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            'scripts/train_go2_command_history_residual_development.py',
            'scripts/command_history_residual_snapshot_development.py',
            'lewm/command_history_residual_learning_development.py',
            'lewm/observation_horizon_learning_development.py',
            'scripts/prepare_go2_short_pulse_training_development.py',
            'scripts/pre_switch_training_data_development.py')})
    write(PLAN,plan);OUTPUT.mkdir();print('PREPARED command-history residual comparison',flush=True)


def fit(condition):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    plan=json.loads(PLAN.read_text());raw=SCHEDULE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==plan['schedule_sha256']
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==plan['reference_sha256']
    for name,sha in plan['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    schedule=json.loads(raw);rows=data.load_training_rows()
    assert len(rows)==4694 and all(r['data_role']=='train' for r in rows)
    for name,sha in schedule['input_sha256'].items():assert hashlib.sha256((data.BASE/name).read_bytes()).hexdigest()==sha
    output=OUTPUT/condition;output.mkdir(exist_ok=False);started=time.monotonic();trainer=None
    write(output/'launch.json',dict(plan=str(PLAN),condition=condition,pid=os.getpid(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=1))
    try:
        cache,identities=data.prepare(rows);write(output/'consumed_policy_sha256.json',identities)
        trainer=CommandHistoryResidualTrainer(condition,seed=plan['seed'],latent_dim=32)
        with (output/'updates.jsonl').open('x') as ledger:
            for step,ids in enumerate(schedule['batches'],1):
                record=trainer.step(previous.batch(cache,ids));ledger.write(json.dumps(record)+'\n')
                if step==1 or step%300==0:
                    ledger.flush();print('FIT',condition,step,'loss',round(record['loss'],5),
                        'elapsed_s',round(time.monotonic()-started,1),flush=True)
        binding=dict(experiment_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            dataset_sha256=plan['old_models'][condition]['binding']['dataset_sha256'],
            schedule_sha256=plan['schedule_sha256'],reference_sha256=plan['reference_sha256'],input_variant='full')
        payload=dict(schema=snapshot.SCHEMA,binding=binding,trainer=trainer.checkpoint());config=snapshot.config(trainer)
        snapshot.validate_payload(payload,binding,config)
        buffer=io.BytesIO();torch.save(payload,buffer);checkpoint=buffer.getvalue();filename=condition+'.pt'
        with (output/filename).open('xb') as stream:stream.write(checkpoint);stream.flush();os.fsync(stream.fileno())
        digest=hashlib.sha256(checkpoint).hexdigest()
        snapshot.load_snapshot(output,filename,sha256=digest,expected_binding=binding,expected_config=config)
        write(output/'fit.json',dict(condition=condition,filename=filename,sha256=digest,
            model_sha256=payload['trainer']['model_sha256'],initial_sha256=trainer.initial_sha256,
            configuration=config,binding=binding,wall_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            prediction_evaluated=False,navigation_tested=False))
        print('COMMAND_REFERENCE_FIT_COMPLETE',condition,flush=True)
    except Exception as error:
        write(output/'failure.json',dict(reason=repr(error),updates=None if trainer is None else trainer.updates));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--condition',choices=CONDITIONS);args=parser.parse_args()
    if args.prepare and args.condition is None:prepare()
    elif not args.prepare and args.condition is not None:fit(args.condition)
    else:raise ValueError('prepare or fit one condition')

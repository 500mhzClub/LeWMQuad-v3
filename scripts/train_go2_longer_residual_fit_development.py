"""Fixed 6000-update J/S fits on the unchanged training-only schedule."""
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

from lewm.nominal_motion_residual_learning_development import NominalResidualTrainer
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import nominal_motion_residual_snapshot_development as snapshot
from scripts import prepare_go2_short_pulse_training_development as data
from scripts.pre_switch_training_data_development import batch
from scripts import run_go2_view_replan_repeatability_development as navigation

BASE=navigation.BASE
OUTPUT=BASE/'go2_longer_residual_matched_fits_v1_attempt_001'
PLAN=Path('docs/go2_longer_residual_fit_plan_2026-09-17.json')
SCHEDULE=data.OUTPUT/'schedule.json'
CONDITIONS=('jepa','supervised_rollout')
UPDATES=6000


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def prepare():
    frozen=json.loads(navigation.PLAN.read_text());schedule=json.loads(SCHEDULE.read_text())
    assert schedule['updates']==1200 and schedule['batch_size']==6 and len(schedule['batches'])==1200
    if OUTPUT.exists():raise ValueError('preserve any fit attempt')
    plan=dict(schema='longer_residual_fit_plan.v1',conditions=CONDITIONS,updates=UPDATES,
        seed=schedule['seed'],batch_size=6,training_contexts=4694,
        schedule_path=str(SCHEDULE),schedule_sha256=hashlib.sha256(SCHEDULE.read_bytes()).hexdigest(),
        schedule_repetitions=5,training_draws_per_condition=36000,
        old_models={c:frozen['models'][c] for c in CONDITIONS},
        training_inputs_and_architecture_unchanged=True,new_navigation_data_in_training=False,
        fresh_initialization=True,checkpoint_selection=False,only_final_update_evaluated=True,
        original_1200_update_state_must_reproduce=True,
        parallel_cpu_fits=True,threads_per_fit=1,estimated_peak_memory_per_fit_gib=10,
        observed_available_memory_gib=64,native_jobs_during_training=False,
        primary_evaluation='prediction errors on the same matched executed windows in all six recent development missions',
        evaluation_roots=[navigation.root_name(n) for n in range(1,5)]+[
            f'go2_earlier_visual_recovery_{n:02d}_jepa_noise_2mm_native_layout01_4800_v1_attempt_001' for n in (1,2)],
        limits=['exposed development recordings, not final evaluation',
            'same training seed and fixed final update only','prediction improvement is not navigation improvement'],
        source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__,'lewm/nominal_motion_residual_learning_development.py',
             'lewm/observation_horizon_learning_development.py','scripts/prepare_go2_short_pulse_training_development.py',
             'scripts/pre_switch_training_data_development.py')})
    write(PLAN,plan);OUTPUT.mkdir();print('PREPARED fixed 6000-update JEPA and supervised fits',flush=True)


def fit(condition):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    plan=json.loads(PLAN.read_text());raw=SCHEDULE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==plan['schedule_sha256']
    for name,sha in plan['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    schedule=json.loads(raw);rows=data.load_training_rows()
    assert len(rows)==4694 and all(r['data_role']=='train' for r in rows)
    for name,sha in schedule['input_sha256'].items():
        assert hashlib.sha256((data.BASE/name).read_bytes()).hexdigest()==sha
    output=OUTPUT/condition;output.mkdir(exist_ok=False);started=time.monotonic();trainer=None
    write(output/'launch.json',dict(plan=str(PLAN),condition=condition,pid=os.getpid(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),torch_threads=1))
    try:
        cache,identities=data.prepare(rows)
        write(output/'consumed_policy_sha256.json',identities)
        trainer=NominalResidualTrainer(condition,seed=plan['seed'],latent_dim=32)
        original_matched=False
        with (output/'updates.jsonl').open('x') as ledger:
            for step in range(1,UPDATES+1):
                record=trainer.step(batch(cache,schedule['batches'][(step-1)%1200]))
                ledger.write(json.dumps(record)+'\n')
                if step==1200:
                    assert state_digest(trainer.model.state_dict())==plan['old_models'][condition]['model_sha256']
                    original_matched=True
                if step==1 or step%500==0:
                    ledger.flush();print('FIT',condition,step,'loss',round(record['loss'],5),
                        'elapsed_s',round(time.monotonic()-started,1),flush=True)
        binding=dict(experiment_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
            dataset_sha256=plan['old_models'][condition]['binding']['dataset_sha256'],
            schedule_sha256=hashlib.sha256(raw+b'\nrepeat_exact_schedule_five_times').hexdigest(),input_variant='full')
        payload=dict(schema=snapshot.SCHEMA,binding=binding,trainer=trainer.checkpoint())
        config=snapshot.config(trainer);snapshot.validate_payload(payload,binding,config)
        buffer=io.BytesIO();torch.save(payload,buffer);checkpoint=buffer.getvalue()
        filename=condition+'.pt'
        with (output/filename).open('xb') as stream:stream.write(checkpoint);stream.flush();os.fsync(stream.fileno())
        digest=hashlib.sha256(checkpoint).hexdigest()
        snapshot.load_snapshot(output,filename,sha256=digest,expected_binding=binding,expected_config=config)
        write(output/'fit.json',dict(condition=condition,filename=filename,sha256=digest,
            model_sha256=payload['trainer']['model_sha256'],initial_sha256=trainer.initial_sha256,
            configuration=config,binding=binding,original_1200_update_model_reproduced=original_matched,
            wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            prediction_evaluated=False,navigation_tested=False))
        print('LONGER_FIT_COMPLETE',condition,flush=True)
    except Exception as error:
        write(output/'failure.json',dict(reason=repr(error),updates=None if trainer is None else trainer.updates))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--condition',choices=CONDITIONS);args=parser.parse_args()
    if args.prepare:
        if args.condition is not None:raise ValueError('prepare separately')
        return prepare()
    if args.condition is None:raise ValueError('condition required')
    fit(args.condition)


if __name__=='__main__':main()

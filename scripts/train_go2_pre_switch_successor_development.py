"""Three fixed-budget data successors; same model, seed and training objectives."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import time
import cv2
import torch
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
from scripts.pre_switch_training_data_development import BASE, prepare, load_training_rows, batch
from scripts.observation_horizon_snapshot_development import SCHEMA, validate_payload, config, load_snapshot

OUTPUT = BASE/'go2_pre_switch_matched_fits_v1_attempt_001'
SCHEDULE = BASE/'go2_pre_switch_training_schedule_v1_attempt_001/schedule.json'
CONDITIONS = ('jepa', 'direct', 'supervised_rollout')
MODEL_DEFINITION = dict(motion_parameterization='absolute', motion_head_initialization='original')
MODEL_SOURCES = ()
EXPECTED_CONTEXTS = 4514


def write(name, value):
    with (OUTPUT/name).open('x') as f:
        json.dump(value, f, indent=2)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    if OUTPUT.exists():
        raise ValueError('preserve any completed or incomplete fit')
    if shutil.disk_usage(BASE).free < 1024**3:
        raise ValueError('one GiB available for small checkpoint outputs required')
    schedule_bytes = SCHEDULE.read_bytes(); schedule = json.loads(schedule_bytes)
    for name, sha in schedule['input_sha256'].items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != sha:
            raise ValueError('scheduled target population changed')
    rows = load_training_rows()
    if (len(rows) != EXPECTED_CONTEXTS or schedule['updates'] != 1200 or schedule['batch_size'] != 6
            or set(i for b in schedule['batches'] for i in b) != {r['sample_id'] for r in rows}):
        raise ValueError('fixed complete training schedule required')
    OUTPUT.mkdir(); started = time.monotonic(); trainer = None; completed = []
    launch = dict(seed=schedule['seed'], conditions=list(CONDITIONS), input_variant='full',
        updates=1200, batch_size=6, training_contexts=len(rows),
        schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
        dataset_sha256=hashlib.sha256(json.dumps(schedule['input_sha256'],sort_keys=True).encode()).hexdigest(),
        model_definition=MODEL_DEFINITION, learned_motion_input_added=False,
        training_only_future_rgb=True, native_state_is_target_only=True,
        checkpoint_selection=False, checkpoint_resume=False, native_execution=False,
        pid=os.getpid(), workers=1, torch_threads=1,
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in (
            __file__, 'scripts/pre_switch_training_data_development.py',
            'lewm/observation_horizon_learning_development.py',
            'lewm/observation_horizon_sample_development.py',
            'lewm/rgb_body_tensor_interface_development.py', *MODEL_SOURCES)})
    write('launch.json', launch)
    try:
        cache, identities = prepare(rows)
        write('consumed_policy_sha256.json', identities)
        warm_s = time.monotonic()-started
        print('CACHE_COMPLETE', len(cache), 'seconds', warm_s, flush=True)
        binding = dict(experiment_sha256=hashlib.sha256((OUTPUT/'launch.json').read_bytes()).hexdigest(),
            dataset_sha256=launch['dataset_sha256'], schedule_sha256=launch['schedule_sha256'], input_variant='full')
        for condition in CONDITIONS:
            trainer = ObservationHorizonTrainer(condition, seed=schedule['seed'], latent_dim=32)
            began = time.monotonic()
            with (OUTPUT/(condition+'_updates.jsonl')).open('x') as ledger:
                for step, identifiers in enumerate(schedule['batches'], 1):
                    record = trainer.step(batch(cache, identifiers))
                    ledger.write(json.dumps(record)+'\n')
                    if step == 1 or step % 100 == 0:
                        ledger.flush()
                        print('FIT_UPDATE', condition, step, 'seconds', round(time.monotonic()-began, 2), flush=True)
            if trainer.updates != 1200:
                raise ValueError('complete fixed-budget fit required')
            payload = dict(schema=SCHEMA, binding=binding, trainer=trainer.checkpoint())
            expected = config(trainer)
            validate_payload(payload, binding, expected)
            buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue()
            if len(raw) > 64*1024**2 or shutil.disk_usage(OUTPUT).free < 1024**3+len(raw):
                raise ValueError('bounded checkpoint storage required')
            filename = condition+'.pt'
            with (OUTPUT/filename).open('xb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            sha = hashlib.sha256(raw).hexdigest()
            clone = load_snapshot(OUTPUT, filename, sha256=sha, expected_binding=binding, expected_config=expected)
            record = dict(condition=condition, filename=filename, sha256=sha, bytes=len(raw),
                configuration=expected, binding=binding, initial_sha256=trainer.initial_sha256,
                model_sha256=payload['trainer']['model_sha256'],
                wall_s=time.monotonic()-began, reload_model_sha256=clone.checkpoint()['model_sha256'])
            write(condition+'_fit.json', record); completed.append(record)
            del clone, payload, buffer, raw, trainer
            trainer = None
        write('result.json', dict(status='COMPLETE', models=completed, total_updates=3600,
            cache_warm_s=warm_s, wall_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            prediction_evaluation_complete=False, navigation_tested=False, goal_complete=False))
        print('THREE_PRE_SWITCH_FITS_COMPLETE', flush=True)
    except Exception as error:
        write('failure.json', dict(reason=repr(error), completed_models=completed,
            incomplete_model_updates=trainer.updates if trainer is not None else None))
        raise


if __name__ == '__main__':
    main()

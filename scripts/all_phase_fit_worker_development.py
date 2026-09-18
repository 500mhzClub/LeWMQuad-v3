"""Direct-subprocess worker with a private reusable cache and fresh fit state."""
import hashlib
import os
import resource
import time
import numpy as np
import psutil
import torch
import cv2
from lewm.observation_horizon_learning_development import ObservationHorizonTrainer
from lewm.observation_horizon_fit_development import score
from lewm.all_phase_training_fit_development import train, predict
from lewm.all_phase_training_schedule_development import schedule as build_schedule, SEEDS
from scripts.all_phase_study_inputs_development import stream, corrected_inputs
from scripts.all_phase_fit_execution_development import (BENCH, OUTPUT, WORKER_RAM, RESERVE,
    CACHE_BYTES, canonical, make_request, science)
from scripts.observation_horizon_snapshot_development import save_snapshot, load_snapshot
from scripts.navigation_artifact_root_development import verify_artifacts, validate_root
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json


def resource_check(output):
    import shutil
    if psutil.Process().memory_info().rss > WORKER_RAM:
        raise ValueError('actual worker RSS exceeds10GiB; partial evidence retained')
    if shutil.disk_usage(output).free < RESERVE:
        raise ValueError('artifact storage reserve exhausted; partial evidence retained')


def worker(*, phase, request_name, request_sha256):
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output = BENCH if phase == 'benchmark' else OUTPUT
    validate_root(output); verify_artifacts(output, {request_name: request_sha256})
    request = read_json(output, request_name)
    if request != make_request(request['phase'], request['slot'], request['workers'], request['launch_sha256']):
        raise ValueError('exact assigned worker request required')
    if request_name != request['name']+'_request.json' or (request['phase']=='fits') != (phase=='fits'):
        raise ValueError('exact request filename and execution phase required')
    name = request['name']; terminal = dict(name=name, status='ALL_PHASE_FIT_WORKER_FAILED', jobs=[],
        artifact_sha256={request_name:request_sha256}, actual_updates=0, warmed_contexts=0, cache_bytes=0)
    started = time.perf_counter(); trainer = None
    try:
        verify_artifacts(output, {'launch.json':request['launch_sha256']})
        launch = read_json(output, 'launch.json'); verify_ordered_launch(launch)
        verify_artifacts(output, {'training_schedules.json':launch['training_schedules_file_sha256']})
        schedules = read_json(output, 'training_schedules.json')
        if (launch['phase'] != phase or launch['output_root'] != str(output)
                or phase=='fits' and request['workers'] != launch['selected_workers']):
            raise ValueError('worker assignment must match the bound launch phase and concurrency')
        if launch['science'] != science(schedules): raise ValueError('exact scientific definition required')
        data = stream(maximum_cache_bytes=8*1024**3)
        for seed in SEEDS:
            if schedules[str(seed)] != build_schedule(data.view, seed=seed):
                raise ValueError('all exact complete expanded training schedules required')
        warm_started = time.perf_counter(); ids = data.view.indices('train')
        for at in range(0, len(ids), 6):
            data.training_batch(ids[at:at+6]); resource_check(output)
            terminal['warmed_contexts'] = min(at+6, len(ids))
            if at%300 == 0: print('ALL_PHASE_CACHE_WARM', name, terminal['warmed_contexts'], flush=True)
        if len(ids) != 4010 or data.training.cache_bytes != CACHE_BYTES or len(data.training._cache) != 4010:
            raise ValueError('complete exact4010-context private tensor cache required')
        terminal.update(cache_bytes=data.training.cache_bytes, cache_warm_wall_s=time.perf_counter()-warm_started)
        benchmark = phase == 'benchmark'
        for job in request['jobs']:
            trainer = None; prefix = name+'_'+job['name']; fit_started = time.perf_counter()
            schedule = schedules[str(SEEDS[0] if benchmark else job['seed'])]
            trainer = ObservationHorizonTrainer(job['condition'], seed=job['seed'], latent_dim=32)
            ledger_name = prefix+'_updates.jsonl'; content = hashlib.sha256(); count = 0
            with (output/ledger_name).open('xb') as ledger:
                def record(row):
                    nonlocal count
                    count += 1; resource_check(output)
                    if row['update'] != count: raise ValueError('exact optimizer accounting required')
                    raw = canonical(row)+b'\n'; ledger.write(raw); ledger.flush(); os.fsync(ledger.fileno()); content.update(raw)
                    if count == 1 or count%100 == 0: print('ALL_PHASE_FIT_UPDATE', prefix, count, flush=True)
                fit = train(trainer, data, schedule, input_variant=job['variant'], on_update=record, benchmark=benchmark)
            if count != (20 if benchmark else 1200) or digest(output/ledger_name) != content.hexdigest():
                raise ValueError('complete durable fit ledger required')
            names = [ledger_name]; fit_wall = time.perf_counter()-fit_started
            binding = dict(experiment_sha256=request['launch_sha256'], dataset_sha256=launch['science']['dataset_pair_sha256'],
                schedule_sha256=schedule['schedule_sha256'], input_variant=job['variant'])
            saved = None
            if not benchmark:
                saved = save_snapshot(output, prefix+'.pt', trainer, binding); names.append(prefix+'.pt')
                model = load_snapshot(output, prefix+'.pt', sha256=saved['sha256'], expected_binding=binding,
                    expected_config=saved['configuration'])
                prediction_files = {}
                for role in ('train', 'geometry_transfer'):
                    arrays = predict(model, data, role=role, input_variant=job['variant'])
                    file = prefix+'_'+role+'.npz'
                    with (output/file).open('xb') as destination: np.savez_compressed(destination, **arrays)
                    prediction_files[role] = file; names.append(file)
                complete = prefix+'_prediction_phase_complete.json'
                write_json(output/complete, dict(model_sha256=fit['model_sha256'],
                    prediction_sha256={role:digest(output/file) for role,file in prediction_files.items()}))
                names.append(complete)
                for role, file in prediction_files.items():
                    with np.load(output/file, allow_pickle=False) as archive: arrays = {k:archive[k] for k in archive.files}
                    scores = score(data.view, arrays, role=role, head='direct_outcomes' if job['condition']=='direct' else 'rollout_outcomes')
                    scored = prefix+'_'+role+'_scores.json'; write_json(output/scored, scores); names.append(scored)
            fit_file = prefix+'_fit.json'
            write_json(output/fit_file, dict(fit=fit, snapshot=saved, ledger_sha256=content.hexdigest(),
                fit_wall_s=fit_wall, cache_bytes=data.training.cache_bytes)); names.append(fit_file)
            if data.training.cache_bytes != CACHE_BYTES or data.training.failed or data.failed:
                raise ValueError('unchanged complete nonfailed training cache required')
            verify_ordered_launch(launch)
            bindings = {n:digest(output/n) for n in names}; verify_artifacts(output, bindings)
            terminal['artifact_sha256'].update(bindings)
            terminal['jobs'].append(dict(name=job['name'], actual_updates=count, fit=fit,
                ledger_sha256=content.hexdigest(), fit_wall_s=fit_wall, artifact_sha256=bindings))
            terminal['actual_updates'] += count
            trainer = None
        corrected_inputs(); verify_ordered_launch(launch)
        terminal['status'] = 'ALL_PHASE_FIT_WORKER_COMPLETE'
    except Exception as error:
        import traceback
        traceback.print_exc()
        terminal['failure'] = repr(error)
        terminal['incomplete_job_updates'] = trainer.updates if trainer is not None else 0
    terminal.update(wall_s=time.perf_counter()-started, worker_pid=os.getpid(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        benchmark_weights_reused=False, native_execution=False, navigation_qualified=False)
    write_json(output/(name+'_terminal.json'), terminal)
    print('ALL_PHASE_FIT_WORKER_TERMINAL', name, terminal['status'], flush=True)
    return 0 if terminal['status']=='ALL_PHASE_FIT_WORKER_COMPLETE' else 1

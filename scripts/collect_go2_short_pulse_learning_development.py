"""Targeted one-tick translations using the unchanged native training session."""
import contextlib
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import cv2
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.geometry_progress_layout_family_development import specification as geometry_specification
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.moving_action_switch_family_development import CANONICAL, decision as original_decision
from lewm.observation_horizon_targets_development import derive
from scripts.moving_action_switch_episode_development import collect as original_collect
from scripts.navigation_artifact_root_development import BASE, create_output
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_short_pulse_learning_v1_attempt_001'
PROTOCOL = Path('docs/go2_short_pulse_learning_2026-09-16.md')
RESERVE = 1024**3


def assignments():
    rows = [dict(cluster=c, geometry_trial=t, data_role='train' if c in ('cluster_00','cluster_01') else 'geometry_transfer',
        prefix_action=before, pulse_action=pulse) for c,t in CANONICAL.items()
        for before in ('hold','left_turn','right_turn') for pulse in ('forward','left_arc','right_arc')]
    random.Random(2026091601).shuffle(rows)
    return {f'pulse_episode_{i:03d}':r for i,r in enumerate(rows)}


def specification(trial):
    return geometry_specification(assignments()[trial]['geometry_trial'])


def schedule(trial):
    cell = assignments()[trial]
    phases = [(3,'quiet',None), (10,'preceding_action',cell['prefix_action']),
        (3,'committed_zero',None), (1,'translation_pulse',cell['pulse_action']), (8,'zero_drain',None)]
    return [dict(phase=2 if action else 3, role=role,
        requested_command=list(candidate_commands(action)[0]) if action else [0.,0.,0.])
        for count,role,action in phases for _ in range(count)]


decision = bind(original_decision, COMMAND_TICKS=25, schedule=schedule)


def branch_specification(trial):
    return dict(trial=trial, **assignments()[trial], branch_tick=13,
        branch_measured_ns=2_800_000_000, expected_complete_command_ticks=25,
        expected_complete_frames=26, expected_complete_physics_samples=2000,
        prospective_commands=[r['requested_command'] for r in schedule(trial)],
        navigation_qualified=False)


collect = bind(original_collect, specification=specification, schedule=schedule,
    decision=decision, branch_specification=branch_specification, RESERVE=RESERVE)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def run_episode(trial):
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    started = time.monotonic(); terminal = dict(trial=trial, status='FAILED', pid=os.getpid())
    with (OUTPUT/(trial+'_worker.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            result = collect(trial, hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), output=OUTPUT)
            terminal['collection'] = result
            directory = OUTPUT/trial
            with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
                raw = {k:archive[k].copy() for k in ('timestamp_s','base_pose_world','physics_contact','requested_command')}
            cameras = json.loads((directory/'camera_audit.json').read_text())
            commands = [r['requested_command'] for r in schedule(trial)]
            rows = []; cell = assignments()[trial]
            for frame in range(8,18):
                known = commands[frame:frame+8]
                labels = derive(raw, cameras, frame=frame, commands=known)
                rows.append(dict(sample_id=f'short_pulse/{trial}/frame_{frame:02d}', source='short_pulse',
                    trial=trial, **cell, action=cell['pulse_action'], observation_frame=frame,
                    decision_ns=labels['departure_ns'], history_observation_indices=labels['history_observation_indices'],
                    known_commands=known, available=labels['available'], reason=labels['reason'],
                    targets=labels['targets'], observation_horizon_receipt={k:v for k,v in labels.items() if k!='targets'},
                    native_labels_are_target_only=True))
            write(directory/'windows.json', rows)
            complete = (result['completed_ticks']==25 and result['rgbd_frames']==26 and result['physics_samples']==2000
                and result['physical_stop'] is None and result['acquisition_stop'] is None)
            terminal.update(status='COMPLETE' if complete else 'PHYSICAL_OR_ACQUISITION_STOP',
                available_contexts=sum(r['available'] for r in rows),
                windows_sha256=hashlib.sha256((directory/'windows.json').read_bytes()).hexdigest())
        except Exception as error:
            traceback.print_exc(); terminal['reason'] = repr(error)
    terminal['wall_s'] = time.monotonic()-started
    write(OUTPUT/(trial+'_terminal.json'), terminal)
    return terminal


def main():
    if OUTPUT.exists():
        raise ValueError('preserve every collection attempt')
    resources = hardware()
    if resources['artifact_free_bytes'] < 3*1024**3 or resources['memory_available_bytes'] < 16*1024**3:
        raise ValueError('bounded four-worker collection capacity unavailable')
    benchmark = json.loads((BASE/'go2_moving_action_switch_scaling_v1_attempt_001/result.json').read_text())
    if benchmark['selected_workers'] != 4 or not benchmark['all_execution_and_pixel_signatures_equal']:
        raise ValueError('established compatible native collection concurrency required')
    create_output(OUTPUT)
    write(OUTPUT/'launch.json', dict(assignments=assignments(), workers=4, hardware=resources,
        prior_collection_speedup=benchmark['measured_speedup'], physics_paused_during_compute=True,
        native_navigation=False, minimum_free_bytes=RESERVE, source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (str(PROTOCOL),__file__,'scripts/moving_action_switch_episode_development.py',
                'scripts/geometry_progress_family_session_development.py','lewm/observation_horizon_targets_development.py')}))
    trials = list(assignments()); results = []; started = time.monotonic()
    try:
        # Fresh processes per batch avoid retaining native scene/render state.
        for offset in range(0,len(trials),4):
            if shutil.disk_usage(OUTPUT).free < RESERVE+256*1024**2:
                raise ValueError('reserve for next four short episodes unavailable')
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
                for row in pool.map(run_episode, trials[offset:offset+4]):
                    results.append(row)
                    print('SHORT_PULSE_COLLECTION', len(results), row, flush=True)
            if any(r['status']=='FAILED' for r in results):
                raise ValueError('collection worker failed; preserve outputs and diagnose')
        windows = [r for t in trials for r in json.loads((OUTPUT/t/'windows.json').read_text())]
        write(OUTPUT/'windows.json', windows)
        write(OUTPUT/'result.json', dict(status='COMPLETE', episodes=results, contexts=len(windows),
            available_contexts=sum(r['available'] for r in windows), wall_s=time.monotonic()-started,
            windows_sha256=hashlib.sha256((OUTPUT/'windows.json').read_bytes()).hexdigest(),
            newly_independent_maze_evaluations=0, models_trained=False, navigation_tested=False))
        print('SHORT_PULSE_COLLECTION_COMPLETE', flush=True)
    except Exception as error:
        write(OUTPUT/'failure.json', dict(reason=repr(error), completed_episodes=results))
        raise


if __name__ == '__main__':
    main()

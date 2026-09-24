"""Four fixed, exposed-layout navigation trials of the matched readout fits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil
import torch

from scripts.navigation_artifact_root_development import BASE
from scripts import train_go2_horizon_dense_predictor_development as predictor
from scripts import run_go2_maze_view_readout_recovery_development as readout

ROOT = BASE/'go2_maze_view_readout_navigation_v1_attempt_001'
ASSIGNMENTS = ((0, 'old_data'), (0, 'maze_data'), (2, 'maze_data'), (2, 'old_data'))
SOURCES = (__file__, 'scripts/run_go2_dense_horizon_navigation_development.py',
    'scripts/read_go2_dense_horizon_navigation_development.py',
    'lewm/dense_horizon_navigation_development.py')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def output(index, arm):
    return BASE/f'go2_dense_world_model_maze_layout{index:02d}_action_maze_view_{arm}_v1_attempt_001'


def prepare():
    assert not ROOT.exists() and not any(output(*a).exists() for a in ASSIGNMENTS)
    trained = json.loads((readout.OUTPUT/'result.json').read_text())
    diagnostic = BASE/'go2_maze_view_transfer_v1_attempt_001/predictor_diagnostic/result.json'
    assert trained['status'] == 'COMPLETE'
    assert json.loads(diagnostic.read_text())['status'] == 'COMPLETE'
    assert torch.cuda.is_available() and psutil.virtual_memory().available > 32*1024**3
    assert shutil.disk_usage(BASE).free > 8*1024**3
    checkpoints = {arm:dict(path=str(readout.OUTPUT/f'{arm}_final.pt'),
        sha256=trained['checkpoint_sha256'][arm]) for arm in ('old_data','maze_data')}
    assert all(predictor.digest(v['path']) == v['sha256'] for v in checkpoints.values())
    ROOT.mkdir()
    save(ROOT/'plan.json', dict(assignments=ASSIGNMENTS,
        checkpoints=checkpoints, predictor_sha256=predictor.digest(predictor.OUTPUT/'action_final.pt'),
        source_sha256={p:predictor.digest(p) for p in SOURCES},
        component_result_sha256=predictor.digest(diagnostic),
        layout_inventory_sha256=predictor.digest('docs/go2_dense_world_model_maze_inventory_2026-09-18.json'),
        scientific_question='Does matched maze-data readout training change closed-loop behaviour with the encoder, predictor and controller held fixed?',
        selection='Layout 0 is the first prior action failure and has exposed readout diagnostics; layout 2 is the sole prior action success. Both selected before this follow-up.',
        primary_outcome='Physically verified goal and home arrivals without disallowed contact.',
        secondary_outcomes=['simulated duration', 'hold counts', 'same-executed-window motion error'],
        layout_data_used_for_training=False, exposed_development_evaluation=True,
        prospective_independent_cohort=False, new_jepa_objective_ablation=False,
        controller='unchanged action-conditioned dense predictor, six actions and eight horizons',
        navigation_ticks=4800, depth_retention='rgb_only',
        timing='untimed synchronous simulation', hardware_validated=False,
        retries=False, preserve_all_failures=True, automatic_production_promotion=False,
        execution='one native/GPU run at a time, fixed order, physical reader after each owner exits',
        resources=dict(gpu=torch.cuda.get_device_name(0), ram_available_bytes=psutil.virtual_memory().available,
            disk_free_bytes=shutil.disk_usage(BASE).free, affinity=sorted(os.sched_getaffinity(0)))))
    print('MAZE_READOUT_NAVIGATION_PREPARED',len(ASSIGNMENTS),flush=True)


def run():
    plan = json.loads((ROOT/'plan.json').read_text())
    assert plan['assignments'] == [list(a) for a in ASSIGNMENTS]
    assert not (ROOT/'process.json').exists()
    save(ROOT/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time()))
    started, results = time.monotonic(), []
    try:
        for index, arm in ASSIGNMENTS:
            assert all(predictor.digest(p)==h for p,h in plan['source_sha256'].items())
            assert predictor.digest(predictor.OUTPUT/'action_final.pt') == plan['predictor_sha256']
            assert predictor.digest('docs/go2_dense_world_model_maze_inventory_2026-09-18.json') == plan['layout_inventory_sha256']
            checkpoint = plan['checkpoints'][arm]
            assert predictor.digest(checkpoint['path']) == checkpoint['sha256']
            root = output(index,arm)
            assert not root.exists()
            print('MAZE_READOUT_NAVIGATION_START',index,arm,flush=True)
            with (ROOT/f'layout{index:02d}_{arm}.log').open('x') as log:
                owner = subprocess.run([sys.executable,'scripts/run_go2_dense_horizon_navigation_development.py',
                    '--full-mission','--arm','action','--readout-arm','maze_view_'+arm,
                    '--prospective-layout',str(index),'--depth-retention','rgb_only'],stdout=log,stderr=subprocess.STDOUT)
                if (root/'result.json').exists() or (root/'failure.json').exists():
                    reader = subprocess.run([sys.executable,'scripts/read_go2_dense_horizon_navigation_development.py',
                        '--root-name',root.name],stdout=log,stderr=subprocess.STDOUT)
                    if reader.returncode:
                        raise RuntimeError(f'Physical reader exited {reader.returncode}: {root}')
            if owner.returncode:
                raise RuntimeError(f'Native owner exited {owner.returncode}: {root}')
            result = json.loads((root/'dense_navigation_readout.json').read_text())
            assert not result['new_independent_development_layout']
            assert result['motion_readout']['sha256'] == checkpoint['sha256']
            row = dict(layout=index, readout=arm, root=str(root), physical=result['physical'],
                pipeline_faults=result['pipeline_faults'], decisions={k:v for k,v in result['decisions'].items() if k!='rows'})
            results.append(row)
            save(ROOT/f'layout{index:02d}_{arm}_result.json',row)
            print('MAZE_READOUT_NAVIGATION_COMPLETE',index,arm,result['physical']['round_trip_arrival_checks_passed'],flush=True)
        save(ROOT/'result.json',dict(status='COMPLETE',rows=results,wall_s=time.monotonic()-started,
            plan_sha256=predictor.digest(ROOT/'plan.json'),prospective_independent_cohort=False))
    except BaseException as error:
        save(ROOT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc(),completed=results))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()

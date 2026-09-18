"""Extend the original 48 training-only balanced starts to eight native ticks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.geometry_progress_layout_family_development import assignments, specification
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.geometry_progress_family_session_development import GeometryProgressFamilySession
from scripts.rgb_only_goal_session_development import RGBOnlyGoalCapture
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts import train_go2_frozen_vjepa_native_adaptation_development as fit
from scripts import collect_go2_balanced_start_actions_development as original

OUTPUT = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_balanced_start_horizon_actions_v1_attempt_001')
PLAN = Path('docs/go2_balanced_start_horizon_actions_plan_2026-09-18.json')
RESULT = Path('docs/go2_balanced_start_horizon_actions_result_2026-09-18.json')
CASES = tuple((trial, cell['action']) for trial, cell in sorted(assignments().items()) if cell['data_role'] == 'train')


class TrainingSession(RGBOnlyGoalCapture, GeometryProgressFamilySession):
    pass


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists() and len(CASES) == 48
    free = shutil.disk_usage(OUTPUT.parent).free
    assert free > (512 + 100) * 1024**2, free
    plan = dict(cases=CASES, specifications=[specification(t) for t, _ in CASES],
                source_sha256=fit.digest(__file__), training_only=True,
                original_collection_sha256=fit.digest(original.RESULT),
                mesh_retention='After each completed case, hard-link byte-identical meshes to the corresponding preserved original case.',
                context_ticks=[0, 5, 10], future_tick=18, quiet_prefix_ticks=10,
                command_suffix_ticks=8, actions_per_environment=6, environments=8,
                selection='All two training clusters x two openings x two appearances x six actions; no transfer outcomes used for selection.',
                fixed_tapes_are_training_not_navigation=True, depth_recorded=False,
                resources=dict(free_bytes=free, reserve_bytes=512*1024**2,
                               expected_collection_bytes=100*1024**2, workers=2,
                               cpu_groups=[[4,5,6,7],[8,9,10,11]]),
                limitations=['same procedural family and dynamics', 'one deterministic recording per cell',
                             'new data coverage is not JEPA-objective isolation'])
    OUTPUT.mkdir();fit.save(PLAN, plan);fit.save(OUTPUT/'plan.json', plan)
    print('BALANCED_HORIZON_START_COLLECTION_PREPARED', len(CASES), flush=True)


def run(case):
    plan = json.loads(PLAN.read_text());assert fit.digest(__file__) == plan['source_sha256']
    trial, action = CASES[case];spec = specification(trial)
    assert spec == plan['specifications'][case] and spec['data_role'] == 'train'
    directory = OUTPUT/f'case_{case:02d}';directory.mkdir()
    fit.save(directory/'launch.json', dict(pid=os.getpid(), case=case, trial=trial, action=action))
    fit.save(directory/'specification.json', spec)
    torch.set_num_threads(4);torch.manual_seed(2026091706)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    started=time.monotonic();session=None;stop=None
    try:
        initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
        session=TrainingSession(spec,directory);session.install_contact_identity()
        gains=configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        fit.save(directory/'actuator_identity.json',gains)
        try:
            session.settle_recorded();session.capture_current();admit_context_setup(session,fit.digest(PLAN))
            tape=[[0.,0.,0.]]*10+[candidate_commands(action)[0]]*8
            for tick in range(19):
                session.sensor_packets()
                if tick == 18:break
                if shutil.disk_usage(OUTPUT).free < 512*1024**2:raise RuntimeError('storage reserve reached')
                session.phase=1 if tick<10 else 2
                session.command_tick(tape[tick])
        except PhysicalStop as error:
            stop=str(error);session.capture_current()
        assert read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist()) == gains['effective']
        result=dict(status='COMPLETE',case=case,trial=trial,action=action,data_role='train',
                    frames=len(session.image_audit),physical_stop=stop,
                    disallowed_contact=any(bool(s['physics_contact']) for s in session.samples),
                    complete_800ms=stop is None and len(session.image_audit)==19,
                    wall_s=time.monotonic()-started,closed_loop_navigation=False)
    except Exception as error:
        fit.save(directory/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()));raise
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                fit.save(directory/'native_guard_rows.json',session.guard_rows)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()
    # Preserve the entire new record and verify its shared 500-ms prefix.
    predecessor=original.OUTPUT/f'case_{case:02d}'
    result['original_rgb_prefix_matches']=all(
        fit.digest(directory/f'rgb_{i:04d}.png')==fit.digest(predecessor/f'rgb_{i:04d}.png')
        for i in range(min(16,result['frames'])))
    result['mesh_links']=[]
    for name in ('ground_visual.ply','wall_union_visual.ply'):
        source=predecessor/'visual_meshes'/name
        target=directory/'visual_meshes'/name
        identity=fit.digest(target)
        if source.stat().st_dev==target.stat().st_dev and fit.digest(source)==identity:
            temporary=target.with_suffix('.hardlink_tmp')
            os.link(source,temporary);os.replace(temporary,target)
            assert fit.digest(target)==identity
            result['mesh_links'].append(dict(name=name,sha256=identity,source=str(source)))
    fit.save(directory/'result.json',result)
    print('BALANCED_HORIZON_START_CASE_COMPLETE',json.dumps(result),flush=True)


def worker(index):
    # Each native scene gets a fresh process; never relaunch an existing case.
    for case in range(index,len(CASES),2):
        assert not (OUTPUT/f'case_{case:02d}').exists()
        with (OUTPUT/f'case_{case:02d}_process.log').open('x') as log:
            subprocess.run([sys.executable,__file__,'--case',str(case)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('WORKER_COMPLETED',index,case,flush=True)


def summarize():
    assert not RESULT.exists()
    rows=[];groups={}
    for i,(trial,action) in enumerate(CASES):
        root=OUTPUT/f'case_{i:02d}'
        assert not (root/'failure.json').exists()
        r=json.loads((root/'result.json').read_text());assert r['status']=='COMPLETE'
        spec=specification(trial);key=(spec['layout_id'],spec['appearance_seed'])
        if r['complete_800ms'] and not r['disallowed_contact']:
            prefix=[fit.digest(root/f'rgb_{t:04d}.png') for t in (0,5,10)]
            if key in groups:assert prefix==groups[key]
            else:groups[key]=prefix
        rows.append(r|dict(result_sha256=fit.digest(root/'result.json')))
    result=dict(status='COMPLETE',cases=rows,training_only=True,
                eligible=sum(r['complete_800ms'] and not r['disallowed_contact'] for r in rows),
                contacts=sum(r['disallowed_contact'] for r in rows),
                matched_context_groups=len(groups),plan_sha256=fit.digest(PLAN),
                original_rgb_prefix_matches=all(r['original_rgb_prefix_matches'] for r in rows),
                new_navigation=False)
    fit.save(RESULT,result);fit.save(OUTPUT/'result.json',result)
    print('BALANCED_HORIZON_START_COLLECTION_COMPLETE',result['eligible'],result['contacts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--case',type=int,choices=range(48))
    g.add_argument('--worker',type=int,choices=(0,1));g.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.summarize:summarize()
    elif a.worker is not None:worker(a.worker)
    else:run(a.case)

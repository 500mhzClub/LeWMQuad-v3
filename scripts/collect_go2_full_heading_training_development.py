"""Compact visible-robot full-heading motion data on existing train geometries."""
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
import psutil
import torch

from lewm.actuator_gain_development import configure_gains
from lewm.eligible_floor_registration_development import bind
from lewm import geometry_progress_layout_family_development as family
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.physical_execution_development import rotation_xyzw
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.independent_round_trip_session_development import IndependentRoundTripPhysicalInit
from scripts.in_memory_paired_camera_session_development import LzmaRawDepthPairedCameraSession
from scripts.compact_depth_retention_session_development import CompactDepthRetentionMixin
from scripts.live_depth_noise_session_development import LiveDepthNoiseMixin
from scripts.nogil_drawing_session_development import NogilDrawingMixin
from scripts.rgb_navigation_retention_development import RGBNavigationRetentionMixin
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts import train_go2_horizon_dense_predictor_development as fit

OUTPUT = fit.OUTPUT.parent/'go2_full_heading_training_v1_attempt_001'


def training_trials():
    groups = {}
    for trial,cell in sorted(family.assignments().items()):
        if cell['data_role']=='train':
            groups.setdefault((cell['geometry'],cell['appearance_seed']),trial)
    assert len(groups)==8
    return tuple(groups[k] for k in sorted(groups))


TRIALS = training_trials()


def specification(index):
    original = family.specification(TRIALS[index])
    assert original['data_role']=='train'
    return original|dict(layout_index=index)


def pack(spec):
    assert spec==specification(spec['layout_index'])
    return family.pack({k:v for k,v in spec.items() if k!='layout_index'})


class VisibleTrainingPhysicalInit(IndependentRoundTripPhysicalInit):
    def __init__(self,spec,*,backend):
        bind(IndependentRoundTripPhysicalInit.__init__,specification=specification,pack=pack)(self,spec,backend=backend)
        self._runtime['high_level']='fixed training excitation; no navigation policy'


class TrainingSession(RGBNavigationRetentionMixin,NogilDrawingMixin,
        LiveDepthNoiseMixin,CompactDepthRetentionMixin,LzmaRawDepthPairedCameraSession,
        VisibleTrainingPhysicalInit):
    pass


def schedule():
    tape = []; phases = []
    def add(action,count,label):
        tape.extend([candidate_commands(action)[0]]*count)
        phases.extend([label]*count)
    for heading in range(6):
        add('hold',10,f'heading_{heading}_quiet')
        for action in ('hold','left_turn','right_turn','forward','left_arc','right_arc'):
            duration = 1 if action in ('forward','left_arc','right_arc') else 5
            add(action,duration,f'heading_{heading}_{action}')
            add('hold',5-duration,f'heading_{heading}_{action}_tail')
        if heading<5:
            add('left_turn',24,f'heading_{heading}_rotation')
    add('hold',8,'final_settling')
    assert len(tape)==368
    return tape,phases


def prepare():
    free = shutil.disk_usage(OUTPUT.parent).free
    assert free>1024**3 and psutil.virtual_memory().available>16*1024**3
    tape,phases = schedule()
    OUTPUT.mkdir(exist_ok=False)
    fit.save(OUTPUT/'plan.json',dict(trials=TRIALS,specifications=[specification(i) for i in range(8)],
        tape=tape,phases=phases,source_sha256=fit.digest(__file__),data_role='train',
        render_robot=True,old_training_render_robot=False,geometries_changed=False,
        intended_heading_stages=6,heading_increment_requested_rad=24*.1*.45,
        actual_headings_measured_after_collection=True,depth_arrays_retained=False,
        fixed_tape_training_not_navigation=True,existing_exposed_or_prospective_mazes_used=False,
        workers=2,cpu_groups=[[4,5,6,7],[8,9,10,11]],free_bytes=free,
        estimated_collection_bytes=512*1024**2,minimum_free_bytes=512*1024**2,
        sample_rule='All completed pre-contact 500-ms windows with departure frame >=10.',
        planned_learning='matched old-data versus half-old/half-new continuations of the same frozen-feature motion head; encoder and predictor unchanged',
        limitation='Added heading coverage and robot visibility are combined; their effects are not isolated.'))
    print('FULL_HEADING_PREPARED',len(TRIALS),len(tape)+1,flush=True)


def run(case):
    plan = json.loads((OUTPUT/'plan.json').read_text())
    assert fit.digest(__file__)==plan['source_sha256']
    spec = specification(case); assert spec==plan['specifications'][case]
    directory = OUTPUT/f'case_{case:02d}'
    directory.mkdir(exist_ok=False)
    fit.save(directory/'launch.json',dict(pid=os.getpid(),case=case,cpu_affinity=psutil.Process().cpu_affinity()))
    fit.save(directory/'specification.json',spec)
    torch.set_num_threads(4);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    session = None; stop = None; started = time.monotonic()
    try:
        initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        session = TrainingSession(spec,directory,noise_layout_index=case%4,noise_sigma_mm=2)
        session.install_contact_identity()
        fit.save(directory/'actuator_identity.json',configure_gains(session.ctx.build.robot,
            session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint'))
        try:
            session.settle_recorded();admit_context_setup(session,plan['source_sha256'])
            for tick in range(len(plan['tape'])*5+1):
                if tick%5==0:
                    if shutil.disk_usage(OUTPUT).free<plan['minimum_free_bytes']:
                        raise RuntimeError('recording reserve reached')
                    session.sensor_packets()
                    if tick%500==0:
                        print('FULL_HEADING_PROGRESS',case,tick//5,flush=True)
                if tick==len(plan['tape'])*5:
                    break
                session.phase = 2
                session.command_policy_step(plan['tape'][tick//5])
        except PhysicalStop as error:
            stop = str(error)
        result = dict(status='COMPLETE' if stop is None else 'PHYSICAL_STOP',case=case,
            trial=TRIALS[case],data_role='train',camera_frames=len(session.captured_pairs),
            physical_stop=stop,disallowed_contact=any(bool(s['physics_contact']) for s in session.samples),
            wall_s=time.monotonic()-started,closed_loop_navigation=False,robot_visible=True)
    except BaseException as error:
        fit.save(directory/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
            finally:
                session.ctx.build.scene.destroy()
        shutdown_genesis()
    fit.save(directory/'result.json',result)
    print('FULL_HEADING_CASE_COMPLETE',json.dumps(result),flush=True)


def worker(index):
    for case in range(index,8,2):
        with (OUTPUT/f'case_{case:02d}_process.log').open('x') as log:
            subprocess.run([sys.executable,__file__,'--case',str(case)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FULL_HEADING_WORKER_CASE',index,case,flush=True)


def summarize():
    samples = []; cases = []
    for case in range(8):
        directory = OUTPUT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text())
        assert result['data_role']=='train' and not (directory/'failure.json').exists()
        metadata = json.loads((directory/'in_memory_camera_observations.json').read_text())
        frames = metadata['frames']
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            poses = archive['base_pose_world'].copy(); contact = archive['physics_contact'].copy()
        headings = []
        for row in frames:
            R = rotation_xyzw(poses[row['physical_sample_index'],3:])
            headings.append(float(np.arctan2(R[1,0],R[0,0])))
        count = 0
        for frame in range(10,len(frames)-5):
            start,end = [frames[f]['physical_sample_index'] for f in (frame,frame+5)]
            if contact[:end+1].any():
                continue
            assert frames[frame+5]['measured_ns']-frames[frame]['measured_ns']==500_000_000
            R = rotation_xyzw(poses[start,3:]); future = rotation_xyzw(poses[end,3:]); relative = R.T@future
            delta = (poses[end,:3]-poses[start,:3])@R
            motion = [float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))]
            samples.append(dict(sample_id=f'full_heading/case_{case:02d}/frame_{frame}',case=case,frame=frame,
                current_rgb=str(directory/f'rgb_{frame:04d}.png'),future_rgb=str(directory/f'rgb_{frame+5:04d}.png'),
                motion=motion,horizon_ms=500,data_role='train'))
            count += 1
        occupied = sorted(set((np.floor((np.asarray(headings)+np.pi)/(2*np.pi)*12).astype(int)%12).tolist()))
        cases.append(result|dict(samples=count,occupied_30_degree_heading_bins=occupied,
            unwrapped_heading_span_rad=float(np.ptp(np.unwrap(headings))) if headings else 0.))
    fit.save(OUTPUT/'samples.json',samples)
    result = dict(status='COMPLETE',cases=cases,training_windows=len(samples),
        complete_recordings=sum(c['status']=='COMPLETE' for c in cases),
        physical_stops=sum(c['status']=='PHYSICAL_STOP' for c in cases),
        samples_sha256=fit.digest(OUTPUT/'samples.json'),training_only=True,no_model_fit_yet=True)
    fit.save(OUTPUT/'result.json',result)
    print('FULL_HEADING_COLLECTION_COMPLETE',len(samples),'windows',result['physical_stops'],'physical stops',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--case',type=int,choices=range(8))
    group.add_argument('--worker',type=int,choices=(0,1));group.add_argument('--summarize',action='store_true')
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.case is not None: run(args.case)
    elif args.worker is not None: worker(args.worker)
    else: summarize()

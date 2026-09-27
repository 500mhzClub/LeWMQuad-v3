"""Approved navigation capability owner: fresh, fixed episode assignments.

V0 inherits the deployed controller, sensing and execution implementations.
The oracle's physics requests are serviced on the native main thread while the
source controller and simulation clock are paused.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
from dataclasses import replace
from functools import partial
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
import yaml
from lewm_genesis.lewm_contract import SafetyLimits
from lewm.navigation_capability_direct_slot_development import load as load_direct
from lewm.navigation_capability_unused_workload_development import UnusedNeuralWorkload
from lewm.navigation_capability_sensor_retention_development import SensorHashRetentionMixin, FullSensorRetentionMixin
from lewm.navigation_capability_target_reference_development import settled_task_cues, install_task_cues
from lewm.physical_execution_development import rotation_xyzw
from lewm.navigation_capability_environment_development import verify_environment
from lewm.navigation_capability_live_turn_memory_development import LiveTurnMemoryRuntimeMixin as StartupRecoveryRuntimeMixin
from lewm.navigation_capability_paired_floor_start_development import initialize_mapping as initialize_startup_mapping
import shutil
import time
import traceback

import cv2
import numpy as np
import psutil
import torch

from lewm import decision_headroom_json_v42_development as output_json
from lewm import independent_round_trip_layouts_development as generator
from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_oracle_development import OracleMotionModel
from scripts import run_go2_dense_horizon_navigation_development as source
from scripts.run_go2_decision_headroom_branches_development import TRACE_FIELDS
from scripts.run_go2_headroom_v42_source_development import AuditSession

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO/'docs/go2_navigation_capability_live_turn_v3_2026-09-27.json'
PROTOCOL_SHA = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
FREEZE = REPO/'docs/go2_navigation_capability_harness_v3_live_turn_final_2026-09-27.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, separators=(',',':'))
        stream.write('\n')


class ResourceStop(RuntimeError):
    pass


class Budget:
    def __init__(self, root, protocol):
        self.root, self.protocol = root, protocol
        self.started = time.time()
        ledger = root/'wall_budget_origin.json'
        if not ledger.exists():
            save(ledger, dict(started_unix_s=self.started, cap_hours=160,
                basis='Elapsed programme run/video window; includes intervening time conservatively'))
        self.origin = json.loads(ledger.read_text())['started_unix_s']
        self.last = 0.
        self.peak_rss = 0
        self.peak_device_used = 0
        self.check(force=True)

    def check(self, force=False):
        now = time.monotonic()
        if not force and now-self.last < 1.:
            return
        self.last = now
        if time.time()-self.origin >= 160*3600-120:
            raise ResourceStop('160-hour run/video window closeout reserve reached')
        caps = self.protocol['caps']
        for path, reserve in ((self.root,caps['recovery_reserve_bytes']), (REPO,caps['workspace_reserve_bytes'])):
            if shutil.disk_usage(path).free < reserve+128*1024**2:
                raise ResourceStop('filesystem reserve closeout boundary: '+str(path))
        p = psutil.Process()
        self.peak_rss = max(self.peak_rss, sum(q.memory_info().rss for q in [p,*p.children(recursive=True)] if q.is_running()))
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(device)
                self.peak_device_used = max(self.peak_device_used,total-free)
                if free < caps['per_device_vram_reserve_bytes']:
                    raise ResourceStop('device VRAM reserve reached')

    def admit_persist(self, size):
        if shutil.disk_usage(self.root).free < self.protocol['caps']['recovery_reserve_bytes']+size+128*1024**2:
            raise ResourceStop('insufficient space for measured episode closeout')


def episode_inputs(root, maze, episode):
    if maze not in range(30):
        raise ValueError('Only registered development and validation roles admitted')
    role='dev_tune' if maze<10 else 'validation'
    if role=='validation':
        gate=json.loads((root/'cohorts/v3_live_turn_C0_gate/result.json').read_text())
        assert gate['passed'] and gate['harness_sha256']==sha(FREEZE)
        assert episode==0, 'Fixed reduced validation design'
    folder = root/'sets'/role
    packet = json.loads((folder/f'episode_{maze:02d}_{episode}.json').read_text())
    original = json.loads((folder/f'maze_{maze:02d}.json').read_text())
    assert packet['role'] == original['data_role'] == role
    spec = original | dict(procedural_seed=packet['simulation_seed'],
        geometry=original['geometry'] | dict(spawn_se2_world=packet['home_se2_world']))
    return spec, packet


def make_session(spec, directory, full_frames=False):
    retention = FullSensorRetentionMixin if full_frames else SensorHashRetentionMixin
    def specification(index):
        if index != spec['layout_index']:
            raise ValueError('episode identity mismatch')
        return spec
    def pack(value):
        definition = bind(generator.pack,specification=specification)(value)
        x,y,yaw = value['geometry']['spawn_se2_world']
        return replace(definition,robot=replace(definition.robot,spawn_xyz_m=(x,y,.375),
            spawn_quat_wxyz=(math.cos(yaw/2),0.,0.,math.sin(yaw/2))))
    class EpisodeInit(source.ProspectivePhysicalInit):
        __init__ = bind(source.ProspectivePhysicalInit.__init__,specification=specification,pack=pack)
    class EpisodeSession(retention,source.previous.native.NogilDrawingMixin,
            source.previous.native.study.cohort.LiveDepthNoiseMixin,
            source.previous.native.study.cohort.CompactDepthRetentionMixin,
            source.previous.native.study.cohort.LzmaRawDepthPairedCameraSession,EpisodeInit):
        pass
    return EpisodeSession(spec,directory,noise_layout_index=spec['layout_index']%4,noise_sigma_mm=2)


def load_model(arm,protocol,root):
    if arm=='C0':return OracleMotionModel()
    if arm=='C4':
        freeze=json.loads(FREEZE.read_text())
        assert sha(root/'c4_fit_attempt002/direct_final.pt')==freeze['C4_final_binding']['sha256']
        return load_direct(protocol)
    if arm in ('C1','C2'):
        path=root/({'C1':'videos/pipeline_test_attempt001/replay_verification.json',
            'C2':'equivalence/C2_unused_workload_serial_attempt001/result.json'}[arm])
        evidence=json.loads(path.read_text())
        assert evidence['unused_workload_equivalence_passed'] and evidence['exact_native_trace_values']
        limits=SafetyLimits.from_manifest(yaml.safe_load((REPO/'config/go2_platform_manifest.yaml').read_text()))
        return UnusedNeuralWorkload(arm,limits)
    return source.load_dense_navigation_model('action',readout_arm='maze_view_maze_data')


def drain(controller, model, session, budget):
    started = time.monotonic()
    oracle_wall = 0.
    while any(q.unfinished_tasks for q in controller.queues.values()):
        if isinstance(model,OracleMotionModel):
            before = time.monotonic()
            model.service(session,budget)
            oracle_wall += time.monotonic()-before
        budget.check()
        if controller.faults:
            raise RuntimeError(str(controller.faults))
        if time.monotonic()-started-oracle_wall > 120:
            raise TimeoutError('unchanged non-oracle controller drain timeout')
        time.sleep(.005)
    if controller.faults:
        raise RuntimeError(str(controller.faults))


def run(arm, maze, episode, assignment):
    assert sha(PROTOCOL) == PROTOCOL_SHA
    protocol = json.loads(PROTOCOL.read_text())
    root = Path(protocol['output_root'])
    output_json.install(root)
    freeze = json.loads(FREEZE.read_text())
    for name, binding in (freeze['original_source_bindings'] | freeze['implementation_bindings']).items():
        if sha(REPO/name) != binding['sha256']:
            raise ValueError('frozen implementation changed: '+name)
    for name, binding in freeze['refined_containment_evidence'].items():
        if sha(root/name) != binding['sha256']:
            raise ValueError('frozen containment cutoff changed: '+name)
    environment = verify_environment(REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    if arm=='C0':
        screen=json.loads((root/'cohorts/v3_live_turn_C1_screen/result.json').read_text())
        assert screen['passed'] and screen['harness_sha256']==sha(FREEZE)
        second=json.loads((root/'cohorts/v3_live_turn_C1_second_screen/result.json').read_text())
        assert second['passed'] and second['successes']>=9 and second['episodes']==10 and second['harness_sha256']==sha(FREEZE)
        if maze>=20:raise ValueError('C0 validation limited to ten lowest IDs')
    full_frames = arm=='C0' and assignment==protocol['first_corrected_C0_assignment']
    retention_evidence = None
    if arm=='C0' and not full_frames:
        receipt=root/'live_turn_v3_C0_sensor_replay/result.json'
        evidence=json.loads(receipt.read_text())
        assert evidence['status']=='PASS' and evidence['harness_sha256']==sha(FREEZE)
        retention_evidence=dict(path=str(receipt),sha256=sha(receipt))
    elif arm!='C0':
        receipt=root/'sensor_regeneration_2026-09-26'/arm/'result.json'
        assert json.loads(receipt.read_text())['status']=='PASS'
        retention_evidence=dict(path=str(receipt),sha256=sha(receipt))
    spec, packet = episode_inputs(root,maze,episode)
    destination = root/'runs'/assignment
    destination.mkdir(parents=True,exist_ok=False)
    directory = destination/'native';directory.mkdir()
    budget = Budget(root,protocol)
    allocation=(8 if full_frames else 1)*1024**3
    budget.admit_persist(allocation)
    config = dict(schema='navigation_capability_run.v1',controller=arm,maze=maze,episode=episode,
        assignment=assignment,protocol_sha256=PROTOCOL_SHA,harness='v3_live_turn',
        harness_sha256=sha(FREEZE),
        episode_packet_sha256=sha(root/'sets'/packet['role']/f'episode_{maze:02d}_{episode}.json'),
        policy_budget_s=480, policy_steps_cap=24000, settle_s=1.5,
        recording_allocation_bytes=allocation,implementation_check=False,
        retention='full_frames' if full_frames else 'hashes_only',retention_evidence=retention_evidence,
        environment=environment,task_cues='one_time_settled_start_frame_both_targets',
        physics_backend='cpu',device='cuda:0' if arm!='C0' else 'physics cpu; native renderer',
        original_source_stack_unchanged=False,shared_correctness_version=False, outcome_driven_mapping_version=True)
    save(destination/'config.json',config)
    save(destination/'process.json',dict(pid=psutil.Process().pid,created=psutil.Process().create_time()))
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    requests=[];acquisitions=[];published=[]
    model=controller=session=None
    clock=source.UntimedSimulationClock()
    started=time.monotonic();error=None
    try:
        model = load_model(arm,protocol,root)
        source.previous.warmup()
        source.previous.study.cohort.stable.floor.configure()
        save(destination/'specification.json',spec)
        save(destination/'episode.json',packet)
        with contextlib.ExitStack() as stack:
            def pool(initializer):
                return stack.enter_context(ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initializer))
            registration=pool(source.previous.study.previous.reference.previous.initialize_registration)
            mapping=pool(initialize_startup_mapping)
            pose=pool(partial(source.initialize_pose,str(destination)))
            obstacles=pool(source.previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(source.previous.native.baseline.registration_ready).result()
            assert mapping.submit(source.mapping_ready).result()
            assert pose.submit(source.pose_ready).result()
            assert obstacles.submit(source.previous.study.cohort.stable.obstacles_ready).result()
            log=stack.enter_context((destination/'worker.log').open('x'))
            stack.enter_context(contextlib.redirect_stdout(log));stack.enter_context(contextlib.redirect_stderr(log))
            source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
            budget.check(force=True)
            session=make_session(spec,directory,full_frames=full_frames)
            session.install_contact_identity()
            save(destination/'actuator_identity.json',source.configure_gains(session.ctx.build.robot,
                session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint'))
            session.settle_recorded()
            source.admit_context_setup(session,sha(__file__))
            pose_at_start=np.asarray(session.samples[-1]['base_pose_world'],dtype=float)
            cues=settled_task_cues(packet,pose_at_start[:3],rotation_xyzw(pose_at_start[3:]))
            save(destination/'task_reference_evaluator.json',dict(
                settled_pose_world=pose_at_start.tolist(),public_cues=cues,
                convention='Initial-body XY projection of world XY targets at settled body-origin height',
                true_pose_available_to_controller=False,one_time_only=True))
            save(destination/'public_task_cues.json',cues)
            def sink(frame,raw,registered):
                published.append(dict(frame=frame,raw_pose=raw['current_pose'],registered_pose=registered['current_pose']))
            base_runtime=source.DenseReactiveNavigationRuntime if arm=='C2' else source.DenseNavigationRuntime
            runtime=type('StartupRecoveredRuntime',(StartupRecoveryRuntimeMixin,base_runtime),{})
            controller=runtime(model,goal_initial_xy=cues['goal_initial_body_xy_m'],
                condition='jepa',variant='full',clock_ns=clock,evidence_sink=sink,
                planning_delay_ticks=3,maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history' if arm=='C1' else 'neural',
                registration_executor=registration,navigation_ticks=4800,arrival_radius_m=.02,
                mapping_executor=mapping,pose_executor=pose,obstacle_executor=obstacles)
            install_task_cues(controller,cues)
            session.physics_clock_callback=clock.advance
            history=deque(maxlen=4)
            with (destination/'progress.jsonl').open('x') as progress:
                for tick in range(24000):
                    budget.check()
                    sim_ns=int(session.ctx.runner._sim_time_ns);clock.advance(sim_ns)
                    if tick%5==0:
                        policy,depth,fast,auxiliary_depth,auxiliary_rgb,measured=session.sensor_packets()
                        history.append(policy)
                        acquired=source.AcquiredFrame(tick//5,measured,policy,depth,fast,auxiliary_rgb,auxiliary_depth,tuple(history))
                        controller.submit(acquired);drain(controller,model,session,budget)
                        acquisitions.append(dict(frame=acquired.frame,measured_ns=measured))
                    request=controller.request(now_ns=clock())
                    row=dict(request,simulator_ns=sim_ns,pre_sample_index=len(session.samples)-1)
                    requests.append(row);session.phase=2
                    row['applied_command']=session.command_policy_step(request['requested_command'])
                    row['post_sample_index']=len(session.samples)-1
                    if tick%500==0:
                        progress.write(json.dumps(dict(policy_steps=tick+1,simulated_s=(tick+1)*.02,
                            wall_s=time.monotonic()-started,plans=len(controller.planning)))+'\n')
                    if controller.faults:
                        raise RuntimeError(str(controller.faults))
                    if controller.mission_terminal is not None:
                        break
            controller.finish()
    except BaseException as exc:
        error=exc
        save(destination/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),
            policy_steps=len(requests),frames=len(acquisitions),controller_failure_is_retained_result=True))
    finally:
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:thread.join(timeout=2.)
            for name,value in (('planning',controller.planning),('poses',published),('pipeline_faults',controller.faults),
                               ('mission',controller.mission_rows),('stage_timings',clock.releases),('startup_recovery',controller.startup_rows),
                               ('exhausted_view_retirements',controller.exhausted_view_retirements)):
                save(destination/f'{name}.json',value)
        clock.close()
        save(destination/'requests.json',requests);save(destination/'acquisitions.json',acquisitions)
        if model is not None:save(destination/'model_calls.json',model.receipts)
        try:
            if session is not None:
                session.physics_clock_callback=None
                # Preserve complete logs and the selected qualified recording mode.
                budget.admit_persist(allocation)
                session.persist(directory)
                session.persist_observations(directory)
                if isinstance(model,OracleMotionModel) and requests:
                    check=model.verify_executed(requests,session.samples)
                    save(destination/'oracle_prefix_check.json',check)
                    if not check['passed'] and error is None:
                        error=RuntimeError('oracle executed-prefix fidelity check failed')
        except BaseException as exc:
            save(destination/'closeout_failure.json',dict(reason=repr(exc),traceback=traceback.format_exc()))
            if session is not None and not (directory/'physics_trace.npz').exists():
                with (directory/'emergency_physics_trace.npz').open('xb') as stream:
                    np.savez_compressed(stream,**{key:np.stack([r[key] for r in session.samples]) for key in TRACE_FIELDS})
            if error is None:error=exc
        finally:
            if session is not None:session.ctx.build.scene.destroy()
            source.shutdown_genesis()
        save(destination/'result.json',dict(status='EPISODE_RECORDED' if error is None else 'FAILED_ATTEMPT_PRESERVED',
            error=None if error is None else repr(error),frames=len(acquisitions),policy_steps=len(requests),
            simulated_s=len(requests)*.02,wall_s=time.monotonic()-started,peak_process_tree_rss_bytes=budget.peak_rss,
            peak_device_used_bytes=budget.peak_device_used,oracle_branches=6*len(model.receipts) if isinstance(model,OracleMotionModel) else 0,
            capability_classification_pending=True,automatic_retry=False))
    print(json.dumps(dict(assignment=assignment,status='RECORDED' if error is None else 'FAILURE',error=repr(error))))
    if error is not None:raise error


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',choices=['C0','C1','C2','C3','C4'],required=True)
    parser.add_argument('--maze',type=int,choices=range(30),default=0)
    parser.add_argument('--episode',type=int,choices=(0,1),default=0)
    parser.add_argument('--assignment',required=True)
    args=parser.parse_args()
    if '/' in args.assignment or args.assignment.startswith('.'):
        raise ValueError('single fresh assignment name required')
    run(args.controller,args.maze,args.episode,args.assignment)

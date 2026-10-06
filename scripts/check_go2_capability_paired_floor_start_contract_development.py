"""Startup-only floor-output contract, queued after the unchanged C1 screen."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
import psutil
import torch
from lewm import decision_headroom_json_v42_development as output
from lewm import process_mapped_runtime_development as tracking
from lewm import process_registered_round_trip_development as registration
from lewm.navigation_capability_startup_recovery_development import RecoverableStartupMap
from lewm.navigation_capability_paired_floor_start_development import PairedFloorStartupMap
from lewm.physical_execution_development import rotation_xyzw
from scripts import run_go2_navigation_capability_paired_floor_v1_development as owner

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
PLAN=owner.REPO/'docs/go2_navigation_capability_paired_floor_output_contract_plan_2026-09-26.json'
ROOT=BASE/'paired_floor_output_contract_attempt001'
NORMAL=(0,3,4,5,7)


def load(path):return json.loads(path.read_text())


def worker(stage,maze,episode):
    plan=load(PLAN);budget=owner.Budget(BASE,load(owner.PROTOCOL));budget.admit_persist(plan['retained_cap_bytes'])
    dest=ROOT/f'{stage}_{maze:02d}_{episode}';dest.mkdir()
    historical=BASE/f'runs/v0_grid_c3_screen_C1_dev{maze:02d}_ep0_attempt001'
    if episode==0:
        for p in (BASE/'depth_retention.json',historical/'depth_retention.json',historical/'native/depth_retention.json'):
            if p.exists():raise RuntimeError('Historical retention manifest requires review: '+str(p))
    spec,packet=owner.episode_inputs(BASE,maze,episode)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    session=None;started=time.monotonic()
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        owner.source.initialize_pose(str(dest))
        owner.source.previous.study.previous.reference.previous.initialize_registration()
        owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        native=dest/'native';native.mkdir()
        session=owner.make_session(spec,native,full_frames=True);session.install_contact_identity()
        owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(__file__))
        policy,depth,fast,aux,auxrgb,now=session.sensor_packets()
        acquisition=owner.source.AcquiredFrame(0,now,policy,depth,fast,auxrgb,aux,())
        raw=tracking.pose_update(acquisition)
        if raw.get('current_pose') is None:raise ValueError('Original startup tracker did not establish pose')
        evidence=registration.register(policy,depth,aux,raw,now)
        origin=np.asarray(session.samples[-1]['base_pose_world'])
        # Evaluator reads the native physical ground plane only after acquisition.
        # No true values are supplied to either mapper or the pose estimator.
        floor=session.ctx.build.collision_floor
        ground_position=np.asarray(floor.get_pos().cpu()).reshape(-1,3)[0]
        ground_quat=np.asarray(floor.get_quat().cpu()).reshape(-1,4)[0]
        np.testing.assert_array_equal(ground_position,np.zeros(3))
        assert np.array_equal(ground_quat,[1.,0.,0.,0.]) or np.array_equal(ground_quat,[-1.,0.,0.,0.])
        rows={}
        for label,cls in ([('c3',RecoverableStartupMap)] if stage=='calibration' else
                          [('c3',RecoverableStartupMap),('paired_floor_v1',PairedFloorStartupMap)]):
            mapper=cls()
            try:
                snapshot=mapper.update(policy,depth,evidence,auxiliary_depth=aux,measured_ns=now)
                normal_map=mapper.B@rotation_xyzw(origin[3:]).T@np.array([0.,0.,1.])
                true_height=float((ground_position[2]-origin[2])/normal_map[2])
                rows[label]=dict(initialised=True,floor_source=mapper.initial_floor_source,
                    floor_height_m=snapshot.floor_height,true_floor_height_at_map_origin_m=true_height,
                    signed_error_m=snapshot.floor_height-true_height,map_from_initial=mapper.B,
                    evaluator_true_floor_normal_map=normal_map)
            except ValueError as exc:
                rows[label]=dict(initialised=False,error=repr(exc),classification='UNRESOLVED_STARTUP_OUTPUT')
        if stage=='calibration':
            assert rows['c3']['initialised'] and rows['c3']['floor_source']=='primary'
        else:
            tolerance=load(ROOT/'frozen_tolerance.json')['tolerance_m']
            for row in rows.values():row['passed']=row['initialised'] and abs(row['signed_error_m'])<=tolerance
        hashes=session.captured_pairs[-1]['consumed_hash_record'];session.captured_pairs[-1]['images']=[]
        # The ten previously collected episode-zero starts must reproduce their
        # original consumed packets. New episode-one starts have no prior trace.
        historical_match=None
        if episode==0:
            old=load(historical/'native/in_memory_camera_observations.json')['frames'][0]
            for key in ('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index'):
                assert hashes[key]==old[key],(maze,key)
            historical_match=True
        owner.save(dest/'result.json',dict(status='RECORDED',stage=stage,episode_id=packet['episode_id'],rows=rows,
            sensor_hashes=hashes,historical_first_frame_hashes_match=historical_match,
            native_settled_pose_evaluator_only=origin,ground_position_evaluator_only=ground_position,
            registered_start_pose=evidence['current_pose'],wall_s=time.monotonic()-started,
            simulated_s=1.5,mission_commands=0,frames_retained=False,
            source_episode_sha256=owner.sha(BASE/'sets/dev_tune'/f'episode_{maze:02d}_{episode}.json')))
    except BaseException as exc:
        owner.save(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),stop_required=True,no_retry=True));raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()


def execute(stage,maze,episode):
    with (ROOT/f'{stage}_{maze:02d}_{episode}.log').open('x') as stream:
        p=subprocess.run([sys.executable,__file__,'--stage',stage,'--maze',str(maze),'--episode',str(episode)],stdout=stream,stderr=subprocess.STDOUT)
    if p.returncode:
        owner.save(ROOT/'stop.json',dict(stage=stage,maze=maze,episode=episode,returncode=p.returncode,no_retry=True))
        raise SystemExit(p.returncode)
    return load(ROOT/f'{stage}_{maze:02d}_{episode}/result.json')


def compare_recordings():
    from lewm.navigation_capability_refined_containment_development import first_difference
    results=[]
    for maze in NORMAL:
        old=BASE/f'runs/v0_grid_c3_screen_C1_dev{maze:02d}_ep0_attempt001'
        new=BASE/f'runs/v1_paired_floor_screen_C1_dev{maze:02d}_ep0_attempt001'
        if not (new/'result.json').exists():
            results.append(dict(maze=maze,compared=False,reason='Screen did not execute this assignment; no replacement run'))
            continue
        arrays={}
        with np.load(old/'native/physics_trace.npz',allow_pickle=False) as a,np.load(new/'native/physics_trace.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                x,y=a[key],b[key];n=min(len(x),len(y))
                diff=np.flatnonzero(~(x[:n]==y[:n]).reshape(n,-1).all(1))
                arrays[key]=dict(exact=x.dtype==y.dtype and np.array_equal(x,y),
                    first_difference_sample=int(diff[0]) if len(diff) else n if len(x)!=len(y) else None)
        keys=('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index')
        aa,bb=([ {k:r[k] for k in keys} for r in load(path/'native/in_memory_camera_observations.json')['frames']] for path in (old,new))
        requests_a,requests_b=(load(path/'requests.json') for path in (old,new))
        sensor=first_difference(aa,bb);request=first_difference(requests_a,requests_b)
        command=first_difference([r['applied_command'] for r in requests_a],[r['applied_command'] for r in requests_b])
        results.append(dict(maze=maze,compared=True,native_arrays=arrays,first_sensor_difference_frame=sensor,
            first_request_difference_index=request,first_applied_command_difference_index=command,
            entire_recording_identical=all(r['exact'] for r in arrays.values()) and sensor is None and request is None,
            source_hashes={str(path/name):owner.sha(path/name) for path in (old,new) for name in ('config.json','requests.json','native/physics_trace.npz','native/in_memory_camera_observations.json')}))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('calibration','contract'));p.add_argument('--maze',type=int);p.add_argument('--episode',type=int);args=p.parse_args()
    output.install(BASE)
    if args.stage:return worker(args.stage,args.maze,args.episode)
    plan=load(PLAN);assert owner.sha(__file__)==plan['script_sha256']
    ROOT.mkdir(exist_ok=False);owner.save(ROOT/'config.json',plan|dict(plan_sha256=owner.sha(PLAN)))
    process=plan['wait_for_screen_process']
    while psutil.pid_exists(process['pid']):
        try:
            if psutil.Process(process['pid']).create_time()!=process['created']:break
        except psutil.NoSuchProcess:break
        time.sleep(30)
    screen=BASE/'cohorts/v1_paired_floor_C1_screen/result.json'
    if not screen.exists():
        owner.save(ROOT/'stop.json',dict(reason='Screen owner ended without cohort closeout; report before contract physics'))
        return
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    started=time.monotonic();calibration=[execute('calibration',i,0) for i in NORMAL]
    maximum=max(abs(r['rows']['c3']['signed_error_m']) for r in calibration)
    tolerance=max(.002,math.ceil(2*maximum/.001)*.001)
    if tolerance>.005:
        owner.save(ROOT/'stop.json',dict(reason='Primary-path calibration exceeds predeclared tolerance ceiling',maximum_error_m=maximum,proposed_tolerance_m=tolerance))
        return
    owner.save(ROOT/'frozen_tolerance.json',dict(tolerance_m=tolerance,maximum_primary_path_error_m=maximum,
        rule=plan['tolerance_rule'],calibration=[dict(episode=r['episode_id'],error_m=r['rows']['c3']['signed_error_m']) for r in calibration],
        frozen_before_any_new_path_contract_run=True,new_path_results_used=False))
    rows=[]
    for maze in range(10):
        for episode in (0,1):
            if time.monotonic()-started>plan['execution_wall_cap_s']:
                owner.save(ROOT/'stop.json',dict(reason='Startup-contract wall cap',completed=len(rows)));return
            rows.append(execute('contract',maze,episode))
    comparisons=compare_recordings()
    failures=[r['episode_id'] for r in rows if not r['rows']['paired_floor_v1']['passed']]
    result=dict(status='FAIL_REPORT_BEFORE_NEXT_CHANGE' if failures else 'PASS',tolerance_m=tolerance,
        episodes=rows,failed_starts=failures,normal_recording_comparisons=comparisons,
        old_c3_passed=sum(r['rows']['c3']['passed'] for r in rows),new_passed=20-len(failures),
        change_confined_to_auxiliary_path=False,version_charge_stands=True,outcome_versions_consumed=2,
        screen_unchanged=True,execution_wall_s=time.monotonic()-started,simulated_s=37.5)
    owner.save(ROOT/'result.json',result)
    lines=['# Paired-floor startup-output contract','',f"Status: **{result['status']}**. Tolerance fixed before new-path tests: {tolerance*1000:.1f} mm.",'',
        f"C3 passed {result['old_c3_passed']}/20; paired-floor V1 passed {result['new_passed']}/20.",'',
        '| Episode | C3 error (mm) | V1 error (mm) | V1 pass |','|---|---:|---:|---|']
    for r in rows:
        def error(k):return f"{r['rows'][k]['signed_error_m']*1000:.3f}" if r['rows'][k]['initialised'] else 'unresolved'
        lines.append(f"| {r['episode_id']} | {error('c3')} | {error('paired_floor_v1')} | {r['rows']['paired_floor_v1']['passed']} |")
    lines+=['','Normal-start recording comparison:', '']
    lines += [f"- {r['maze']:02d}/0: "+(f"exact={r['entire_recording_identical']}; first sensor difference frame {r['first_sensor_difference_frame']}; first applied-command difference index {r['first_applied_command_difference_index']}" if r['compared'] else r['reason']) for r in comparisons]
    lines+=['','The version charge stands: V1 changes normal-start floor initialisation too. Two of six versions remain charged. No running-screen code, assignments or criteria were changed.',
        '', 'Any failed or unresolved start requires reporting before a further change. No tracker estimation change was made.', '',f'Full evidence: `{ROOT}/result.json`.']
    with (ROOT/'report.md').open('x') as stream:stream.write('\n'.join(lines)+'\n')
    print(output.dumps(dict(status=result['status'],new_passed=result['new_passed'],tolerance_m=tolerance,report=str(ROOT/'report.md'))),flush=True)


if __name__=='__main__':main()

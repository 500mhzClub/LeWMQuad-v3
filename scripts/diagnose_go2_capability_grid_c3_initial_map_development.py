"""Bounded first-planning-frame regeneration; unchanged map, logged accepted poses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import cv2
import torch
from lewm import decision_headroom_json_v42_development as output
from lewm.navigation_capability_startup_recovery_development import RecoverableStartupMap
from scripts import run_go2_navigation_capability_correctness_c3_development as owner

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
ROOT=BASE/'grid_c3_initial_map_diagnosis_attempt001'
EPISODES=(0,1,2,6,8,9)


def run(index):
    config=json.loads((ROOT/'config.json').read_text())
    source=BASE/f'runs/v0_grid_c3_screen_C1_dev{index:02d}_ep0_attempt001'
    dest=ROOT/f'dev{index:02d}';dest.mkdir()
    protocol=json.loads(owner.PROTOCOL.read_text());budget=owner.Budget(BASE,protocol)
    budget.admit_persist(config['retained_cap_bytes'])
    for p in (BASE/'depth_retention.json',source/'depth_retention.json',source/'native/depth_retention.json'):
        if p.exists():raise RuntimeError('Depth retention manifest requires review before regeneration: '+str(p))
    spec=json.loads((source/'specification.json').read_text())
    requests=json.loads((source/'requests.json').read_text())[:60]
    poses={p['frame']:p['registered_pose'] for p in json.loads((source/'poses.json').read_text())}
    frames=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames'][:13]
    plans={p['frame']:p for p in json.loads((source/'planning.json').read_text()) if 'selection' in p}
    def recorded_pose(evidence,*,identity,now_ns):
        pose=poses[evidence['frame']]
        assert pose['measured_ns']==now_ns and identity==(0,0,0)
        return np.asarray(pose['position_initial_body_m']),np.asarray(pose['rotation_initial_body_from_current_body']),pose
    mapper=RecoverableStartupMap();mapper._read_pose=recorded_pose
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    session=None;rows=[];started=time.monotonic()
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as archive:
        trace={k:archive[k][:1350].copy() for k in archive.files}
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        native=dest/'native';native.mkdir()
        session=owner.make_session(spec,native,full_frames=True);session.install_contact_identity()
        owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(__file__))
        for tick in range(61):
            budget.check()
            if time.time()-config['started_unix_s']>config['wall_cap_s']:raise RuntimeError('Initial-map diagnosis wall cap')
            if tick%5==0:
                frame=tick//5;policy,depth,fast,aux,auxrgb,now=session.sensor_packets()
                camera=session.captured_pairs[-1];actual=camera['consumed_hash_record']
                for key in ('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index'):
                    assert actual[key]==frames[frame][key],(index,frame,key)
                camera['images']=[]
                if frame%4==0:
                    snapshot=mapper.update(policy,depth,dict(frame=frame),auxiliary_depth=aux,measured_ns=now)
                    if frame in plans:
                        expected=plans[frame]['selection']['routing_memory_scope']
                        assert len(snapshot.floor)==expected['retained_floor_cells']
                        assert len(snapshot.fine_occupied)==expected['retained_fine_obstacle_cells']
                    plane=mapper.last_floor_plane
                    p,R,_=recorded_pose(dict(frame=frame),identity=(0,0,0),now_ns=now)
                    Q,q=mapper.B@R,mapper.B@p
                    normal=Q@np.asarray(plane['normal_body']) if plane.get('normal_body') is not None else None
                    offset=float(plane['offset_body_m']-normal@q) if normal is not None else None
                    rows.append(dict(frame=frame,floor_height_m=mapper.floor_height,
                        initial_floor_source=mapper.initial_floor_source,primary_covered=snapshot.primary_current_floor_cells,
                        auxiliary_covered=snapshot.auxiliary_current_floor_cells,retained_floor=len(snapshot.floor),
                        retained_obstacles=len(snapshot.fine_occupied),paired_plane_available=plane['available'],
                        paired_plane_reason=plane.get('reason'),paired_normal_map=normal,paired_offset_map_m=offset,
                        paired_height_at_body_xy_m=float(-(normal[:2]@q[:2]+offset)/normal[2]) if normal is not None else None,
                        plane_height_difference_from_fixed_map_m=float(-(normal[:2]@q[:2]+offset)/normal[2]-mapper.floor_height) if normal is not None else None))
            if tick==60:break
            request=requests[tick];assert int(session.ctx.runner._sim_time_ns)==request['simulator_ns']
            session.phase=2
            np.testing.assert_array_equal(session.command_policy_step(request['requested_command']),request['applied_command'])
        for key,values in trace.items():np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]),values)
        owner.save(dest/'result.json',dict(status='PASS',maze=index,rows=rows,consumed_pairs_bitwise=13,
            native_arrays_exact=True,logged_planning_map_counts_exact=True,wall_s=time.monotonic()-started,
            source_config_sha256=owner.sha(source/'config.json'),map_source_unchanged=True,
            pose_input='Original accepted registered poses; estimator not recomputed or changed',frames_retained=False))
    except BaseException as exc:
        owner.save(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),stop_required=True));raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--maze',type=int,choices=EPISODES);args=parser.parse_args()
    output.install(BASE)
    if args.maze is not None:return run(args.maze)
    pose_result=json.loads((BASE/'grid_c3_pose_loss_diagnosis_attempt001/result.json').read_text())
    assert pose_result['status']=='PASS'
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    ROOT.mkdir(exist_ok=False)
    owner.save(ROOT/'config.json',dict(schema='navigation_capability_initial_map_diagnosis.v1',
        authority='Diagnose shared cause of startup-fixed failures before selecting one tuning change',
        started_unix_s=time.time(),episodes=EPISODES,control_episode=0,script_sha256=owner.sha(__file__),
        maximum_simulated_s=16.2,wall_cap_s=600,retained_cap_bytes=16*1024**2,
        mission_physics='Original 1.5-s settling plus 1.2-s zero-command context per episode; no new navigation',
        purpose='Check fixed mapping floor height against already-computed paired sensor plane, without fitting a new estimator',
        no_retries=True,frames_retained=False,programme_wall_cap_h=160,
        filesystem_reserves_and_environment='Unchanged C3 owner budget and pinned environment'))
    for index in EPISODES:
        with (ROOT/f'dev{index:02d}.log').open('x') as stream:
            process=subprocess.run([sys.executable,__file__,'--maze',str(index)],stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            owner.save(ROOT/'stop.json',dict(maze=index,returncode=process.returncode,automatic_retry=False));raise SystemExit(process.returncode)
    owner.save(ROOT/'result.json',dict(status='PASS',episodes=EPISODES,maximum_simulated_s=16.2))


if __name__=='__main__':main()

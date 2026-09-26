"""Bounded command regeneration and unchanged-tracker replay of three failures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import json
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from lewm import decision_headroom_json_v42_development as output
from lewm import process_mapped_runtime_development as process
from scripts import run_go2_navigation_capability_correctness_c3_development as owner

PLAN=owner.REPO/'docs/go2_navigation_capability_grid_c3_pose_diagnosis_plan_2026-09-26.json'


def initialize_pose(directory,base):
    output.install(Path(base))
    owner.source.initialize_pose(directory)


def tracking(packet):
    raw=process.pose_update(packet)
    if raw.get('current_pose') is None or raw.get('failure') is not None:
        model=process._motion.model
        return raw,{k:getattr(model,k,None) for k in ('last_selection','last_revisit_attempt','last_continuity')}
    return raw,None


def run(maze,root,plan):
    base=Path(plan['base']);source=base/f'runs/v0_grid_c3_screen_C1_dev{maze:02d}_ep0_attempt001'
    expected=next(r for r in plan['sources'] if r['maze']==maze)
    assert owner.sha(source/'config.json')==expected['source_config_sha256']
    dest=root/f'dev{maze:02d}';dest.mkdir()
    spec=json.loads((source/'specification.json').read_text())
    requests=json.loads((source/'requests.json').read_text())
    frames=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames']
    poses={x['frame']:x['raw_pose'] for x in json.loads((source/'poses.json').read_text())}
    retention={str(p):json.loads(p.read_text()) if p.exists() else 'absent; approved hash-only source'
        for p in (base/'depth_retention.json',source/'depth_retention.json',source/'native/depth_retention.json')}
    owner.save(dest/'config.json',dict(source=str(source),source_config_sha256=owner.sha(source/'config.json'),
        source_harness_sha256=owner.sha(owner.FREEZE),plan_sha256=owner.sha(PLAN),depth_retention=retention,
        maximum_simulated_s=1.5+len(requests)*.02,frame_retention=False,tracker_unchanged=True))
    protocol=json.loads(owner.PROTOCOL.read_text());budget=owner.Budget(base,protocol)
    budget.admit_persist(plan['retained_cap_bytes'])
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as f:trace={k:f[k].copy() for k in f.files}
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    started=time.monotonic();session=None;matched=0;verified=0;terminal=None;images=[];summaries=[]
    selected={max(0,len(frames)-11),len(frames)-2,len(frames)-1}
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
                initializer=partial(initialize_pose,str(dest),str(base))) as worker:
            assert worker.submit(owner.source.pose_ready).result()
            owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
            native=dest/'native';native.mkdir()
            session=owner.make_session(spec,native,full_frames=True);session.install_contact_identity()
            owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
            session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(__file__))
            for k,v in trace.items():np.testing.assert_array_equal(np.stack([r[k] for r in session.samples]),v[:len(session.samples)])
            for tick in range(len(requests)+1):
                budget.check()
                if time.time()-plan['started_unix_s']>plan['wall_cap_s']:raise RuntimeError('Diagnosis wall cap reached')
                if tick%5==0 and tick//5<len(frames):
                    frame=tick//5;policy,depth,fast,aux,auxrgb,now=session.sensor_packets()
                    camera=session.captured_pairs[-1];actual=camera['consumed_hash_record']
                    for k in ('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index'):
                        assert actual[k]==frames[frame][k],(frame,k)
                    verified+=1
                    if frame in selected:
                        images.append((frame,[Image.fromarray(rgb.copy()) for rgb,_ in camera['images']]))
                        summaries.append(dict(frame=frame,mission_s=frame*.1,
                            primary_valid_depth_pixels=int(depth['valid'].sum()),auxiliary_valid_depth_pixels=int(aux['valid'].sum()),
                            last_applied_command=requests[tick-1]['applied_command'] if tick else None))
                    camera['images']=[]
                    packet=owner.source.AcquiredFrame(frame,now,policy,depth,fast,auxrgb,aux,())
                    raw,diagnostic=worker.submit(tracking,packet).result()
                    pose=raw.get('current_pose')
                    if frame in poses:
                        assert pose is not None and raw.get('failure') is None,(frame,raw.get('failure'))
                        for k in ('position_initial_body_m','rotation_initial_body_from_current_body'):
                            np.testing.assert_array_equal(pose[k],poses[frame][k])
                        for k in ('mode','reference_frame'):assert pose[k]==poses[frame][k],(frame,k)
                        matched+=1
                    else:
                        assert frame==len(frames)-1 and (pose is None or raw.get('failure') is not None)
                        terminal=dict(frame=frame,raw=raw,diagnostic=diagnostic)
                    if frame%200==0:print('REPLAY',maze,frame,round(time.monotonic()-started,1),flush=True)
                if tick==len(requests):break
                req=requests[tick];assert int(session.ctx.runner._sim_time_ns)==req['simulator_ns']
                session.phase=2
                np.testing.assert_array_equal(session.command_policy_step(req['requested_command']),req['applied_command'])
                end=req['post_sample_index']
                for k,v in trace.items():np.testing.assert_array_equal(np.stack([r[k] for r in session.samples[-10:]]),v[end-9:end+1])
        assert matched==len(poses) and verified==len(frames) and terminal is not None
        # RAM-only inspection sheet, not retained frames or a publication video.
        canvas=Image.new('RGB',(1280,500*len(images)))
        draw=ImageDraw.Draw(canvas)
        for row,(frame,pair) in enumerate(images):
            for col,img in enumerate(pair):canvas.paste(img,(col*640,row*500+20))
            draw.text((5,row*500),f'dev{maze:02d} frame{frame} t={frame*.1:.1f}s | primary left; existing auxiliary right',fill='white')
        temporary=Path(plan['temporary_image_directory'])/f'dev{maze:02d}.png'
        canvas.save(temporary)
        owner.save(dest/'result.json',dict(status='PASS',source=str(source),consumed_frame_pairs_bitwise=verified,
            raw_poses_exact=matched,all_native_states_exact=True,terminal=terminal,view_summaries=summaries,
            temporary_inspection_sheet=str(temporary),retained_sensor_frames=False,
            wall_s=time.monotonic()-started,navigation_reexecuted=False,tracker_changed=False))
    except BaseException as exc:
        owner.save(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),verified_frames=verified,
            matched_poses=matched,stop_required=True,automatic_retry=False));raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--maze',type=int,choices=(1,2,7));args=parser.parse_args()
    frozen=json.loads(PLAN.read_text());base=Path(frozen['base']);root=base/frozen['root_name'];output.install(base)
    if args.maze is not None:
        plan=json.loads((root/'config.json').read_text());run(args.maze,root,plan);return
    screen=json.loads((base/'cohorts/v0_grid_c3_C1_screen/result.json').read_text())
    assert screen['complete'] and screen['refined_containment_passed']
    assert owner.sha(__file__)==frozen['script_sha256']
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    root.mkdir();Path(frozen['temporary_image_directory']).mkdir()
    plan=frozen|dict(started_unix_s=time.time());owner.save(root/'config.json',plan)
    rows=[]
    for i in frozen['episodes']:
        with (root/f'dev{i:02d}.log').open('x') as log:
            result=subprocess.run([sys.executable,__file__,'--maze',str(i)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            owner.save(root/'stop.json',dict(maze=i,returncode=result.returncode,completed=rows,automatic_retry=False))
            raise SystemExit(result.returncode)
        rows.append(dict(maze=i,result=str(root/f'dev{i:02d}/result.json')))
        paths=[root/'config.json',root/'result.json',root/'stop.json']
        for episode in frozen['episodes']:
            paths.append(root/f'dev{episode:02d}.log')
            paths.extend(root/f'dev{episode:02d}'/name for name in
                ('config.json','result.json','failure.json','pose_worker_identity.json'))
        size=sum(p.stat().st_size for p in paths if p.exists())
        if size>frozen['retained_cap_bytes']:raise RuntimeError('Diagnosis retained-output cap reached')
    owner.save(root/'result.json',dict(status='PASS',rows=rows,wall_s=time.time()-plan['started_unix_s'],
        no_new_navigation=True,no_controller_or_estimator_changes=True))


if __name__=='__main__':main()

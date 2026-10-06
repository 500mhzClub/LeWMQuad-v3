"""Bounded command-prefix replay to inspect unchanged tracker recovery witnesses."""
import argparse,json,subprocess,sys,time,traceback,os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import cv2,numpy as np,torch
from lewm import decision_headroom_json_v42_development as output
from lewm import process_mapped_runtime_development as process
from lewm.sparse_corner_completion_runtime_development import strong_corner_support
from scripts import run_go2_navigation_capability_live_turn_v3c1_development as owner
PLAN=owner.REPO/'docs/go2_navigation_capability_v3_support_replay_plan_2026-09-27.json'
def initialize_pose(directory,base):
    output.install(Path(base));owner.source.initialize_pose(directory)
def tracking(packet):
    raw=process.pose_update(packet)
    return raw, strong_corner_support(raw)
def run(index,root,plan):
    assignment=next(x for x in plan['assignments']if x['maze']==index)
    base=Path(plan['base']);source=base/assignment['source'];dest=root/f'dev{index:02d}';dest.mkdir()
    assert owner.sha(source/'config.json')==assignment['source_config_sha256']
    limit=assignment['last_frame'];spec=json.loads((source/'specification.json').read_text())
    requests=json.loads((source/'requests.json').read_text())[:limit*5]
    frames=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames'][:limit+1]
    poses={x['frame']:x['raw_pose']for x in json.loads((source/'poses.json').read_text())if x['frame']<=limit}
    plans={r['frame']:r for r in json.loads((source/'planning.json').read_text())}
    retention={str(p):json.loads(p.read_text())if p.exists()else 'absent; approved hash-only source'for p in [base/'depth_retention.json',source/'depth_retention.json',source/'native/depth_retention.json']}
    owner.save(dest/'config.json',dict(source=str(source),plan_sha256=owner.sha(PLAN),last_frame=limit,depth_retention=retention,frames_retained=False))
    with np.load(source/'native/physics_trace.npz')as f:trace={k:f[k].copy()for k in f.files}
    protocol=json.loads(owner.PROTOCOL.read_text());budget=owner.Budget(base,protocol);budget.admit_persist(plan['retained_cap_bytes'])
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    started=time.monotonic();session=None;verified=0;matched=0;summaries=[]
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=partial(initialize_pose,str(dest),str(base)))as worker:
            assert worker.submit(owner.source.pose_ready).result()
            owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning');native=dest/'native';native.mkdir()
            session=owner.make_session(spec,native,full_frames=True);session.install_contact_identity()
            owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
            session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(__file__))
            for k,v in trace.items():np.testing.assert_array_equal(np.stack([r[k]for r in session.samples]),v[:len(session.samples)])
            for tick in range(len(requests)+1):
                budget.check()
                if time.time()-plan['started_unix_s']>plan['wall_cap_s']:raise RuntimeError('Support replay wall cap')
                if budget.peak_rss>plan['ram_cap_bytes']:raise RuntimeError('Support replay RAM cap')
                if tick%5==0:
                    frame=tick//5;policy,depth,fast,aux,auxrgb,now=session.sensor_packets();camera=session.captured_pairs[-1];actual=camera['consumed_hash_record']
                    for k in ['pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index']:assert actual[k]==frames[frame][k],(frame,k)
                    verified+=1;camera['images']=[]
                    packet=owner.source.AcquiredFrame(frame,now,policy,depth,fast,auxrgb,aux,())
                    raw,receipt=worker.submit(tracking,packet).result();pose=raw.get('current_pose');assert pose is not None and raw.get('failure')is None
                    for k in ['position_initial_body_m','rotation_initial_body_from_current_body']:np.testing.assert_array_equal(pose[k],poses[frame][k])
                    for k in ['mode','reference_frame']:assert pose[k]==poses[frame][k]
                    matched+=1
                    if frame>=assignment['summary_first_frame']:
                        row=plans.get(frame,{})
                        summaries.append(dict(frame=frame,mission_s=frame*.1,receipt=receipt,primary_valid_depth_pixels=int(depth['valid'].sum()),auxiliary_valid_depth_pixels=int(aux['valid'].sum()),source_route=row.get('route_status'),source_action=row.get('action')))
                    if frame%500==0:print('REPLAY',index,frame,round(time.monotonic()-started,1),flush=True)
                if tick==len(requests):break
                req=requests[tick];assert int(session.ctx.runner._sim_time_ns)==req['simulator_ns'];session.phase=2
                np.testing.assert_array_equal(session.command_policy_step(req['requested_command']),req['applied_command']);end=req['post_sample_index']
                for k,v in trace.items():np.testing.assert_array_equal(np.stack([r[k]for r in session.samples[-10:]]),v[end-9:end+1])
        assert matched==len(poses)==verified==len(frames)
        owner.save(dest/'result.json',dict(status='PASS',consumed_frame_pairs_bitwise=verified,raw_poses_exact=matched,all_native_states_exact=True,window_summaries=summaries,frames_retained=False,tracker_changed=False,wall_s=time.monotonic()-started))
    except BaseException as exc:
        owner.save(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),verified=verified,matched=matched,automatic_retry=False));raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()
def main():
    a=argparse.ArgumentParser();a.add_argument('--maze',type=int,choices=[1,9]);args=a.parse_args();plan=json.loads(PLAN.read_text());base=Path(plan['base']);root=base/plan['root_name'];output.install(base)
    if args.maze is not None:run(args.maze,root,json.loads((root/'config.json').read_text()));return
    assert owner.sha(__file__)==plan['script_sha256'];assert json.loads((base/'cohorts/v3c1_live_turn_C1_screen/result.json').read_text())['complete']
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    root.mkdir();plan=plan|dict(started_unix_s=time.time());owner.save(root/'config.json',plan);rows=[]
    for assignment in plan['assignments']:
        maze=assignment['maze']
        with (root/f'dev{maze:02d}.log').open('x')as log:r=subprocess.run([sys.executable,__file__,'--maze',str(maze)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:owner.save(root/'stop.json',dict(maze=maze,returncode=r.returncode,automatic_retry=False));raise SystemExit(r.returncode)
        rows.append(dict(maze=maze,result=str(root/f'dev{maze:02d}/result.json')))
    size=0
    for directory,folders,files in os.walk(root):
        folders[:]=[n for n in folders if n!='sealed' and not n.startswith('sealed_')]
        size+=sum((Path(directory)/n).stat().st_size for n in files if n!='sealed_test.json')
    assert size<=plan['retained_cap_bytes']
    owner.save(root/'result.json',dict(status='PASS',rows=rows,retained_bytes=size,wall_s=time.time()-plan['started_unix_s'],no_new_navigation=True,no_estimator_change=True))
if __name__=='__main__':main()

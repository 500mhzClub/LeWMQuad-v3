"""Pilot-only replay video: verified source replay, then separate chase pass.

The decision replay recomputes the unchanged controller from regenerated sensor
packets and the recorded prediction-slot outputs. It checks candidate tapes,
selected actions and final dispatch commands. No model is fitted or repaired.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
from functools import partial
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import pickle
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_snapshot_development import restore
from lewm.dense_native_observation_development import dense_native_context
from lewm.physical_execution_development import rotation_xyzw
from scripts import run_go2_dense_horizon_navigation_development as source
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, make_session, save, sha


class RecordedPrediction(torch.nn.Module):
    def __init__(self, receipts, controller):
        super().__init__();self.calls={r['observed_ns']:r for r in receipts};self.current=None
        self.readout_identity=dict(arm=controller,training_horizons_ms=list(range(100,801,100)))
        self.used=[];self.eval()

    def set_native_context(self, packets, *, observed_ns):
        dense_native_context(packets,observed_ns=observed_ns)
        self.current=observed_ns

    def forward(self, *, observation_history, known_action_blocks, known_action_valid):
        row=self.calls[self.current];assert known_action_valid.all()
        requested=(known_action_blocks[:,:,0].cpu()*torch.tensor([.3,1.,.5])).numpy()
        np.testing.assert_array_equal(requested,np.asarray(row['requested_commands'],np.float32))
        motion=torch.tensor(row['motion_xy_yaw'],dtype=torch.float32)
        outcomes=torch.cat((motion[:,:,:2],motion[:,:,2:3].sin(),motion[:,:,2:3].cos(),torch.full((6,8,1),-1000.)),dim=-1)
        self.used.append(self.current)
        return dict(rollout_outcomes=outcomes,target_offsets_ns=torch.arange(1,9).mul(100_000_000).expand(6,8),
                    prediction_valid=torch.ones(6,8,dtype=torch.bool),contact_prediction_available=False)


def compare_poses(actual, expected):
    position=float(np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1).max())
    yaw=[]
    for a,b in zip(actual,expected,strict=True):
        A,B=rotation_xyzw(a[3:]),rotation_xyzw(b[3:])
        angle=math.atan2(A[1,0],A[0,0])-math.atan2(B[1,0],B[0,0])
        yaw.append(abs(math.atan2(math.sin(angle),math.cos(angle))))
    degrees=math.degrees(max(yaw))
    if position>.001 or degrees>.1:
        raise ValueError(f'replay pose tolerance failed: {position} m, {degrees} degrees')
    return position,degrees


def initialized_session(spec, directory, source_root):
    session=make_session(spec,directory)
    session.install_contact_identity()
    source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
    session.settle_recorded()
    binding=json.loads((source_root/'snapshot_binding.json').read_text())
    if sha(source_root/binding['path'])!=binding['sha256']:raise ValueError('source snapshot binding mismatch')
    with (source_root/binding['path']).open('rb') as stream:snapshot=pickle.load(stream)
    restore(session,snapshot)
    return session


def verify(source_root, root, budget, spec, episode, expected_trace, requests, frames):
    directory=root/'verification_native';directory.mkdir()
    config=json.loads((source_root/'config.json').read_text());arm=config['controller']
    receipts=json.loads((source_root/'model_calls.json').read_text())
    model=RecordedPrediction(receipts,arm);clock=source.UntimedSimulationClock();controller=session=None
    image_count=decision_count=0;max_position=max_yaw=0.
    expected_plans={r['frame']:r for r in json.loads((source_root/'planning.json').read_text()) if 'selection' in r}
    try:
        with contextlib.ExitStack() as stack:
            def pool(initializer):
                return stack.enter_context(ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initializer))
            registration=pool(source.previous.study.previous.reference.previous.initialize_registration)
            mapping=pool(source.previous.initialize_mapping)
            pose=pool(partial(source.initialize_pose,str(root)))
            obstacles=pool(source.previous.study.previous.reference.previous.initialize_obstacles)
            assert registration.submit(source.previous.native.baseline.registration_ready).result()
            assert mapping.submit(source.mapping_ready).result();assert pose.submit(source.pose_ready).result()
            assert obstacles.submit(source.previous.study.cohort.stable.obstacles_ready).result()
            session=initialized_session(spec,directory,source_root)
            runtime=source.DenseReactiveNavigationRuntime if arm=='C2' else source.DenseNavigationRuntime
            controller=runtime(model,goal_initial_xy=episode['mission']['goal_initial_body_xy_m'],condition='jepa',variant='full',
                clock_ns=clock,evidence_sink=None,planning_delay_ticks=3,maximum_initial_dispatch_lateness_ns=0,
                prediction_source='command_history' if arm=='C1' else 'neural',registration_executor=registration,
                navigation_ticks=4800,arrival_radius_m=.02,mapping_executor=mapping,pose_executor=pose,obstacle_executor=obstacles)
            session.physics_clock_callback=clock.advance;history=deque(maxlen=4)
            for tick,expected_request in enumerate(requests):
                budget.check();now=int(session.ctx.runner._sim_time_ns);clock.advance(now)
                assert now==expected_request['simulator_ns']
                if tick%5==0:
                    p,d,fast,ad,ar,measured=session.sensor_packets();history.append(p)
                    camera=session.captured_pairs[-1];expected_frame=frames[tick//5]
                    for label,pair in zip(('primary','auxiliary'),camera['images'],strict=True):
                        if hashlib.sha256(pair[0].tobytes()).hexdigest()!=expected_frame['pixel_sha256'][label]['rgb_sha256']:
                            raise ValueError('source RGB differs during replay: '+str(tick//5)+'/'+label)
                        image_count+=1
                    if camera['live_depth_noise']!=expected_frame['live_depth_noise']:
                        raise ValueError('replayed controller depth input differs from source')
                    acquired=source.AcquiredFrame(tick//5,measured,p,d,fast,ar,ad,tuple(history))
                    controller.submit(acquired);source.drain(controller)
                    actual_plan=next((r for r in reversed(controller.planning) if r['frame']==acquired.frame and 'selection' in r),None)
                    expected_plan=expected_plans.get(acquired.frame)
                    if (actual_plan is None)!=(expected_plan is None):raise ValueError('replay decision availability differs')
                    if expected_plan is not None:
                        if actual_plan['selection']['action']!=expected_plan['selection']['action']:
                            raise ValueError('replay selected action differs')
                        decision_count+=1
                    # Release only replay recording references after verification.
                    # This pass is not an evaluation recording and is not published.
                    camera['images']=[]
                actual=controller.request(now_ns=clock())
                if actual['requested_command']!=expected_request['requested_command'] or actual['reason']!=expected_request['reason']:
                    raise ValueError('replay final dispatch command/reason differs')
                session.phase=2;applied=session.command_policy_step(actual['requested_command'])
                np.testing.assert_array_equal(applied,expected_request['applied_command'])
                a=np.stack([r['base_pose_world'] for r in session.samples[-10:]])
                end=expected_request['post_sample_index'];b=expected_trace['base_pose_world'][end-9:end+1]
                position,yaw=compare_poses(a,b);max_position=max(max_position,position);max_yaw=max(max_yaw,yaw)
            controller.finish()
            return dict(passed=True,bitwise_source_rgb_matches=image_count,identical_selected_actions=decision_count,
                identical_final_command_steps=len(requests),maximum_position_error_m=max_position,maximum_yaw_error_degrees=max_yaw,
                prediction_slot='Recorded outputs with exact candidate-tape check; shared controller rerun from regenerated source inputs',
                current_source_inputs_bitwise_verified=True)
    finally:
        if controller is not None:
            controller.stopped.set()
            for thread in controller.threads:thread.join(timeout=2.)
        clock.close()
        if session is not None:session.ctx.build.scene.destroy()


class Canvas:
    def __init__(self, source_root, spec, episode, requests, mission):
        self.source_root=source_root;self.spec=spec;self.episode=episode;self.requests=requests
        self.mission={r['frame']:r for r in mission};self.ego_frame=-1;self.ego=None;self.path=[]

    def draw(self, chase, pose, elapsed):
        frame=min(int(elapsed*10+1e-8),max(self.mission))
        if frame!=self.ego_frame:
            self.ego=np.asarray(Image.open(self.source_root/f'native/rgb_{frame:04d}.png').convert('RGB'))
            self.ego_frame=frame
        canvas=np.full((1080,1920,3),20,np.uint8)
        canvas[:720,:960]=cv2.resize(self.ego,(960,720));canvas[:720,960:]=cv2.resize(chase,(960,720))
        map_image=np.full((360,640,3),235,np.uint8)
        def point(xy):return (int(320+(xy[0]-.65)*58),int(180-(xy[1]-.65)*58))
        for wall in self.spec['geometry']['wall_boxes']:
            c=np.array(wall['centre_xyz'][:2]);h=np.array(wall['size_xyz'][:2])/2
            cv2.rectangle(map_image,point(c-h),point(c+h),(65,65,65),-1)
        state=self.mission[frame];colour=(80,160,250) if state['phase']=='OUTBOUND' else (230,70,150)
        self.path.append((point(pose[:2]),colour))
        for (a,_),(b,c) in zip(self.path,self.path[1:]):cv2.line(map_image,a,b,c,2)
        cv2.circle(map_image,point(self.episode['home_se2_world'][:2]),6,(50,170,70),-1)
        cv2.circle(map_image,point(self.episode['beacon_xy_world']),6,(255,170,30),-1)
        cv2.circle(map_image,point(pose[:2]),5,(220,30,30),-1)
        Q=rotation_xyzw(pose[3:]);cv2.arrowedLine(map_image,point(pose[:2]),point(pose[:2]+Q[:2,0]*.3),(220,30,30),2)
        canvas[720:,:640]=map_image
        index=min(int(elapsed/.02),len(self.requests)-1);request=self.requests[index]
        lines=['PIPELINE TEST - C0 oracle - harness v0',f'Maze {self.episode["maze_id"]:02d}, episode {self.episode["episode_index"]} | {elapsed:.2f} simulated seconds',
            f'Phase: {state["phase"]} | command: {request["requested_command"]}',
            f'Hold: {not any(request["requested_command"])} | arrivals: {len(state["arrivals"])}',
            'Development pilot; validation success rate not yet available',
            'Left: actual received RGB at 10 Hz | right: separate replay chase pass']
        for row,line in enumerate(lines):cv2.putText(canvas,line,(675,770+row*48),cv2.FONT_HERSHEY_SIMPLEX,.74,(240,240,240),1,cv2.LINE_AA)
        return canvas


def chase_pass(source_root,root,budget,spec,episode,trace,requests):
    directory=root/'chase_native';directory.mkdir();session=initialized_session(spec,directory,source_root)
    destination=root/'pipeline_test_provisional.mp4'
    log=(root/'ffmpeg.log').open('x')
    process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30',
        '-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',str(destination)],stdin=subprocess.PIPE,stderr=log)
    canvas=Canvas(source_root,spec,episode,requests,json.loads((source_root/'mission.json').read_text()))
    count=0;position=None;keyframes=[];maximum_position=maximum_yaw=0.
    epoch=requests[0]['simulator_ns'];duration=len(requests)*.02
    def render_frame(pose):
        nonlocal count,position
        Q=rotation_xyzw(pose[3:]);desired=pose[:3]-1.8*Q[:,0]+np.array([0.,0.,1.5])
        position=desired if position is None else .2*desired+.8*position
        camera=session.ctx.build.camera;camera.set_pose(pos=position,lookat=pose[:3]+Q[:,0]*.5,up=[0.,0.,1.])
        rendered=camera.render(rgb=True,depth=False,segmentation=False,normal=False)
        rgb=np.asarray(session.ctx.runner._extract_rgb(rendered)).reshape(480,640,3)
        frame=canvas.draw(rgb,pose,count/30)
        process.stdin.write(frame.tobytes())
        if count%max(1,int(duration*30/12))==0:keyframes.append(cv2.resize(frame,(480,270)))
        count+=1
    original=session._sample
    def sampled(requested,applied,stamp):
        value=original(requested,applied,stamp)
        if count/30<duration and stamp-epoch/1e9+1e-9>=count/30:
            render_frame(session.samples[-1]['base_pose_world'])
        return value
    session._sample=sampled
    try:
        render_frame(session.samples[-1]['base_pose_world'])
        for request in requests:
            budget.check();session.phase=2
            applied=session.command_policy_step(request['requested_command'])
            np.testing.assert_array_equal(applied,request['applied_command'])
            actual=np.stack([r['base_pose_world'] for r in session.samples[-10:]])
            end=request['post_sample_index'];position_error,yaw_error=compare_poses(actual,trace['base_pose_world'][end-9:end+1])
            maximum_position=max(maximum_position,position_error);maximum_yaw=max(maximum_yaw,yaw_error)
        process.stdin.close();returncode=process.wait()
        if returncode:raise RuntimeError('ffmpeg failed; see retained log')
        if count!=math.ceil(duration*30-1e-8):raise ValueError('incorrect output frame count')
        sheet=np.zeros((1080,1440,3),np.uint8)
        for i,frame in enumerate(keyframes[:12]):sheet[i//3*270:(i//3+1)*270,i%3*480:(i%3+1)*480]=frame
        Image.fromarray(sheet).save(root/'pipeline_test_contact_sheet.png')
        return dict(frames=count,maximum_position_error_m=maximum_position,maximum_yaw_error_degrees=maximum_yaw,
            native_render_time_quantization_ms=2,primary_camera_exposed_to_chase=False)
    finally:
        if process.poll() is None:
            process.stdin.close();process.wait(timeout=30)
        log.close();session.ctx.build.scene.destroy()


def main(source_root):
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root'])
    assert source_root.resolve().is_relative_to((base/'runs').resolve())
    assert json.loads((source_root/'config.json').read_text())['controller']=='C0','initial pipeline test is C0 only'
    root=base/'videos/pipeline_test_attempt001';root.mkdir(parents=True,exist_ok=False);output.install(base)
    budget=Budget(base,protocol);budget.admit_persist(512*1024**2)
    spec=json.loads((source_root/'specification.json').read_text());episode=json.loads((source_root/'episode.json').read_text())
    requests=json.loads((source_root/'requests.json').read_text());assert requests
    frames=json.loads((source_root/'native/in_memory_camera_observations.json').read_text())['frames']
    with np.load(source_root/'native/physics_trace.npz',allow_pickle=False) as a:trace={'base_pose_world':a['base_pose_world'].copy()}
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    source.previous.warmup();source.previous.study.cohort.stable.floor.configure()
    started=time.monotonic()
    try:
        source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        fidelity=verify(source_root,root,budget,spec,episode,trace,requests,frames)
        save(root/'replay_verification.json',fidelity)
        source.shutdown_genesis();source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        chase=chase_pass(source_root,root,budget,spec,episode,trace,requests)
        (root/'pipeline_test_provisional.mp4').rename(root/'pipeline_test.mp4')
        if len(requests)*.02>120:
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(root/'pipeline_test.mp4'),'-vf',
                "setpts=PTS/4,drawtext=text='4x simulated time':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black",
                '-r','30','-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',str(root/'pipeline_test_4x.mp4')],check=True)
        save(root/'metadata.json',dict(status='PIPELINE_TEST_VERIFIED',episode=episode['episode_id'],source_root=str(source_root),
            source_config_sha256=sha(source_root/'config.json'),source_prediction_outputs_sha256=sha(source_root/'model_calls.json'),
            source_snapshot_sha256=sha(source_root/'initial_snapshot.pkl'),replay=fidelity,chase=chase,
            wall_s=time.monotonic()-started,video_sha256=sha(root/'pipeline_test.mp4'),
            capability_video=False,validation_success_rate_available=False))
    except BaseException as exc:
        save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),publish=False,automatic_retry=False))
        raise
    finally:source.shutdown_genesis()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    main(parser.parse_args().source_root)

"""Find first real legacy-domain exposure on five original command replays.

No controller selection, fitting or new episode. Images are regenerated,
verified against consumed hashes, used once, and discarded. Mapping uses the
source's recorded registered poses, never native pose. Native state is read
only for bitwise replay fidelity, not for boundary classification.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import cv2
import torch
from lewm import decision_headroom_json_v42_development as output
from lewm.navigation_capability_startup_recovery_development import RecoverableStartupMap
from scripts import run_go2_navigation_capability_correctness_c1_development as owner

EPISODES=(0,3,4,5,7)


def inspect_frame(mapper,packet,registered,plan,mission):
    pose=registered['registered_pose'];p=np.asarray(pose['position_initial_body_m']);R=np.asarray(pose['rotation_initial_body_from_current_body'])
    mapper._read_pose=lambda *a,**k:(p,R,pose)
    policy,depth,fast,auxiliary,auxrgb,measured=packet
    snap=mapper.update(policy,depth,registered,auxiliary_depth=auxiliary,measured_ns=measured)
    events=[]
    # Actual old storage clips outside [-100,100), not at the 4.9 point guard.
    for kind,cells,limit,resolution in [('floor_observation',snap.current_floor,100,.05),
        ('obstacle_observation',snap.current_occupied,100,.05),('fine_obstacle_observation',snap.current_fine_occupied,500,.01)]:
        excluded=sorted(c for c in cells if any(v < -limit or v>=limit for v in c))
        if excluded:events.append(dict(kind=kind,old_storage_half_width_m=5.,count=len(excluded),first_cell=excluded[0],cell_m=resolution))
    B=np.asarray(snap.map_from_initial);q=B@p;Q=B@R
    if plan:
        goal=B@np.r_[mission['active_goal_initial_body_xy_m'],0.]
        values=[q[:2],goal[:2]]
        for k in ('target_map_xy_m','original_target_map_xy_m'):
            if k in (plan.get('lookahead') or {}):values.append(plan['lookahead'][k])
        if np.max(np.abs(values))>4.9:events.append(dict(kind='route_query',old_point_bound_m=4.9,maximum_absolute_coordinate_m=float(np.max(np.abs(values)))))
        motion=plan.get('motion_correction',{}).get('corrected_forecast_xy_m')
        if motion is not None:
            paths=q[:2]+np.asarray(motion)@Q[:2,:2].T
            maximum=float(np.max(np.abs(paths)))
            if maximum>4.9:events.append(dict(kind='candidate_endpoint',old_point_bound_m=4.9,maximum_absolute_coordinate_m=maximum))
        coverage=plan.get('selection',{}).get('translation_footprint_coverage')
        if coverage and any(c['leaves_map_bounds'] for c in coverage['candidates']):
            events.append(dict(kind='candidate_footprint',old_storage_half_width_m=5.,source_logged_leaves_map_bounds=True))
    return events


def run(maze):
    assert maze in EPISODES
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    root=base/'grid_c3_bound_exposure'/f'dev{maze:02d}';root.mkdir(parents=True,exist_ok=False)
    source=base/f'runs/v0_task_c1_screen_C1_dev{maze:02d}_ep0_attempt001'
    retention={}
    for p in (base/'depth_retention.json',source/'depth_retention.json',source/'native/depth_retention.json'):
        retention[str(p)]=json.loads(p.read_text()) if p.exists() else 'absent; source uses approved hash-only retention'
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    budget=owner.Budget(base,protocol);budget.admit_persist(64*1024**2)
    spec=json.loads((source/'specification.json').read_text())
    requests=json.loads((source/'requests.json').read_text())
    frames=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames']
    poses={r['frame']:r for r in json.loads((source/'poses.json').read_text())}
    plans={r['frame']:r for r in json.loads((source/'planning.json').read_text())}
    missions={r['frame']:r for r in json.loads((source/'mission.json').read_text())}
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as a:trace={k:a[k].copy() for k in a.files}
    owner.save(root/'config.json',dict(source=str(source),source_config_sha256=owner.sha(source/'config.json'),
        script_sha256=owner.sha(__file__),depth_retention=retention,maximum_simulated_s=1.5+len(requests)*.02,
        programme_wall_cap_hours=160,retained_output_cap_bytes=64*1024**2,frame_retention=False,
        stop_on_first_exposure=True,source_commands_only=True,models_recomputed=False,
        source_pose_packets_only_for_mapping=True,all_frame_hashes_required=True,
        guard_semantics='4.9-m point guard and 5-m actual stored cell clipping both checked; no nominal envelope used as an exposure witness'))
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    session=None;started=time.monotonic();verified=0;exposure=None;mapper=RecoverableStartupMap()
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        native=root/'native';native.mkdir()
        session=owner.make_session(spec,native,full_frames=True);session.install_contact_identity()
        owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(owner.__file__))
        for key,values in trace.items():np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]),values[:len(session.samples)])
        with (root/'progress.jsonl').open('x') as progress:
            for tick in range(len(requests)+1):
                budget.check()
                if tick%5==0 and tick//5<len(frames):
                    frame=tick//5;packet=session.sensor_packets()
                    actual=session.captured_pairs[-1]['consumed_hash_record'];expected=frames[frame]
                    for key in ('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index'):
                        assert actual[key]==expected[key],(frame,key)
                    session.captured_pairs[-1]['images']=[];verified+=1
                    if frame%4==0 and frame in poses:
                        events=inspect_frame(mapper,packet,poses[frame],plans.get(frame),missions.get(frame))
                        if events:
                            exposure=dict(frame=frame,measured_ns=packet[-1],mission_s=frame*.1,events=events)
                            break
                    if frame%100==0:progress.write(json.dumps(dict(frame=frame,wall_s=time.monotonic()-started))+'\n')
                if tick==len(requests):break
                request=requests[tick];assert int(session.ctx.runner._sim_time_ns)==request['simulator_ns']
                session.phase=2;applied=session.command_policy_step(request['requested_command'])
                np.testing.assert_array_equal(applied,request['applied_command'])
                end=request['post_sample_index']
                for key,values in trace.items():np.testing.assert_array_equal(np.stack([r[key] for r in session.samples[-10:]]),values[end-9:end+1])
        owner.save(root/'result.json',dict(status='PASS',source=str(source),first_exposure=exposure,
            verified_frame_pairs=verified,total_source_frame_pairs=len(frames),native_arrays_exact_through_checked_prefix=True,
            consumed_sensor_hashes_bitwise=True,full_source_checked=verified==len(frames),wall_s=time.monotonic()-started,
            no_exposure_means='No legacy-bound truncation in the completed source trajectory' if exposure is None else None,
            no_new_trial=True,frame_retention=False))
    except BaseException as exc:
        owner.save(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),verified_frames=verified,automatic_retry=False));raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--maze',type=int,choices=EPISODES,required=True);run(p.parse_args().maze)

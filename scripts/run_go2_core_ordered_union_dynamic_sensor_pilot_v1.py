"""Fixed eight-run mounted ordered-union sensor diagnostic; no training."""
import shutil
import time
import json
import cv2
import torch
from lewm.independent_layout_collection_development import schedule,decision
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm.visual_led_motion_development import VisualLedMotion
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.core_ordered_dynamic_session_development import CoreOrderedDynamicSession
from scripts.ordered_dynamic_pilot_development import RUNS,RESERVE,BUDGET,ALLOWANCE,definitions,commit_episode
from scripts.ordered_dynamic_audit_development import audit_dynamic_condition
from scripts.independent_layout_batch_development import load_inventory
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts

OUTPUT=BASE/'go2_core_ordered_union_dynamic_sensor_pilot_v1_attempt_001'
PROTOCOL='docs/go2_core_ordered_union_dynamic_sensor_pilot_v1_2026-09-06.md'
FAILED=BASE/'go2_ordered_union_dynamic_sensor_pilot_v1_attempt_001'
FAILED_IDS={'launch.json':'ac2542dfab74b5c4df0f7ad4797999e400c0a6d58b2e11d52db87e777e16d626',
    'failure.json':'ace19931d193ad78054090b0f580e029a4005bb9481fb64ed6bca3961c9dce96',
    'dynamic_audit.json':'404a10fdcb7dfdb16553c6f5a4f4e4efa812af3edf706334a014e5457041fd94'}
CORE=BASE/'go2_core_raster_precision_native_probe_v1_attempt_001'
CORE_IDS={'launch.json':'8f30e3f9e4e9db5fb3025b1ce8753c5438454b895578f0f52739c5e49f93a813',
    'result.json':'fc2d51f3011294573247cfb1782f9c0631dca8a1daa1a8ae390af2e70db60819'}
BENCH=BASE/'go2_ordered_union_rgb_repeatability_probe_v1_attempt_001'
BENCH_IDS={'launch.json':'faf4511c45f9781f85413e9b41e3155eb2a51e1b1560d0826f8b901e68d5dbdd',
    'result.json':'6f8a966e2bcd7268bf2b0b274a565820e8c25d2102ddcd2318e3603c084fc0bf'}


def validate_launch(launch,inv):
    assert launch['output_root']==str(OUTPUT) and launch['planned_runs']==[r for r,_,_ in RUNS]
    assert launch['definitions']==definitions(inv)
    assert launch['maximum_artifact_bytes']==BUDGET and launch['minimum_free_bytes']==RESERVE
    assert launch['episode_allowance_bytes']==ALLOWANCE and launch['model_training'] is False


def preflight():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive dynamic pilot; no retry/resume')
    verify_artifacts(BENCH,BENCH_IDS);old=read_json(BENCH,'launch.json');verify_ordered_launch(old)
    result=read_json(BENCH,'result.json');verify_artifacts(BENCH,result['artifact_sha256'])
    assert result['passes_fixed_bench'] is True
    verify_artifacts(FAILED,FAILED_IDS);verify_artifacts(CORE,CORE_IDS)
    old=read_json(CORE,'launch.json');verify_ordered_launch(old)
    core_result=read_json(CORE,'result.json')
    assert core_result['expected_mechanism_observed'] is True and core_result['core_profile'] is True
    inv=load_inventory()
    launch={k:old[k] for k in ('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
        'opencv_binary_sha256','opencv_version','rules')}
    sources=discover_sources((PROTOCOL,'scripts/run_go2_core_ordered_union_dynamic_sensor_pilot_v1.py',
        'scripts/audit_go2_core_ordered_union_dynamic_sensor_pilot_v1.py',
        'lewm/tests/test_core_ordered_dynamic_pilot_development.py'),old['source_sha256'])
    launch.update(source_sha256=sources,output_root=str(OUTPUT),planned_runs=[r for r,_,_ in RUNS],
        definitions=definitions(inv),maximum_artifact_bytes=BUDGET,minimum_free_bytes=RESERVE,
        episode_allowance_bytes=ALLOWANCE,bench_sha256=BENCH_IDS|result['artifact_sha256'],
        failed_pilot_sha256=FAILED_IDS,core_query_probe_sha256=CORE_IDS,
        maximum_episode_physics_samples=2400,maximum_episode_rgbd_frames=34,maximum_command_ticks=33,
        model_training=False,strict_visibility_preserved=True,boundary_pixels_certified=False,
        data_role='train',hidden_robot_ideal_camera=True,real_time_qualified=False,navigation_qualified=False,goal_achieved=False)
    validate_launch(launch,inv);verify_ordered_launch(launch)
    if len((json.dumps(launch,indent=2,allow_nan=False)+'\n').encode())>BUDGET//16:raise ValueError('serialized metadata budget')
    if shutil.disk_usage(BASE.parent).free<RESERVE+BUDGET:raise ValueError('pilot storage reserve')
    return inv,launch


def collect(inventory,output,run,trial,definition):
    directory=output/run;directory.mkdir();spec=inventory.specification(trial);write_json(directory/'specification.json',spec)
    session=None;rows=[];tape=[];friction=[];stop=None;acquisition_stop=None;terminal=None;admitted=False
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=CoreOrderedDynamicSession(inventory,spec,directory);session.install_contact_identity();build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(directory/'actuator_identity.json',gains)
        write_json(directory/'floor_roles.json',dict(physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_only_link_ids=[int(l.idx) for e in build.visual_surfaces for l in e.links],
            visual_collision_geom_count=sum(len(e.geoms) for e in build.visual_surfaces)))
        friction.append(dict(stage='before_settle',**native_friction(build,spec['friction_mu'])))
        tracker=VisualLedMotion('gyro',identity=(0,0,0))
        try:
            session.settle_recorded();session.capture_current();admit_context_setup(session,definition);admitted=True
            for tick in range(len(schedule(spec['action_index'],spec['history_kind']))+1):
                free=shutil.disk_usage(BASE.parent).free
                if free<RESERVE:acquisition_stop='STORAGE_RESERVE_STOP';break
                friction.append(dict(stage='before_decision',tick=tick,**native_friction(build,spec['friction_mu'])))
                start=time.perf_counter_ns()
                try:
                    p,d,f,now=session.sensor_packets();selected=decision(spec['action_index'],spec['history_kind'],tick,p)
                except (ValueError,TypeError,KeyError) as error:
                    acquisition_stop='PACKET_CONTRACT_STOP: '+str(error);break
                shadow=tracker.observe(p,d,f,now_ns=now)
                rows.append(dict(tick=tick,observation_index=len(session.model_manifest)-1,
                    pre_sample_index=len(session.samples)-1,decision=selected,shadow=shadow,
                    observation_and_control_wall_ms=(time.perf_counter_ns()-start)/1e6,resource_free_bytes=free))
                if selected['terminal']:terminal='FIXED_CONTEXT_PULSE_COMPLETE';break
                session.phase=selected['phase']
                item=dict(tick=tick,requested_command=selected['requested_command'],phase=selected['phase'],role=selected['role'],
                    pre_sample_index=len(session.samples)-1,post_sample_index=None,completed=False)
                tape.append(item)
                try:session.command_tick(item['requested_command']);item['completed']=True
                finally:item['post_sample_index']=len(session.samples)-1
        except PhysicalStop as error:stop=str(error)
        friction.append(dict(stage='terminal',**native_friction(build,spec['friction_mu'])))
        terminal_gains=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        if terminal_gains!=gains['effective']:raise ValueError('frozen gait gains changed')
        write_json(directory/'terminal_actuator_gains.json',terminal_gains)
        write_json(directory/'terminal_native_robot_geometry.json',capture_native_robot_geometry(build.robot))
        write_json(directory/'terminal_environment_identity.json',appearance_environment_identity(session))
        result=dict(status='CONTEXT_TERMINAL_AUDIT_REQUIRED',trial=trial,physical_stop=stop,acquisition_stop=acquisition_stop,
            schedule_terminal=terminal,setup_admitted=admitted,setup_checked=(directory/'setup_checks.json').is_file(),
            departure_present=any(r['tick']==8 for r in rows),command_ticks=len(tape),completed_ticks=sum(t['completed'] for t in tape),
            physics_samples=len(session.samples),rgbd_frames=len(session.model_manifest),decisions=len(rows),
            tracker_required_for_commands=False,native_state_used_for_commands=False,navigation_qualified=False)
        write_json(directory/'result.json',result)
        print('DYNAMIC_EPISODE',run,result,flush=True);return result
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                write_json(directory/'context_decisions.json',rows);write_json(directory/'command_tape.json',tape)
                write_json(directory/'native_guard_rows.json',session.guard_rows);write_json(directory/'friction_checks.json',friction)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    inv,launch=preflight();create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    results={};commits={};prechecks={};used=(OUTPUT/'launch.json').stat().st_size;active=None
    try:
        for i,(run,trial,repeat) in enumerate(RUNS):
            verify_ordered_launch(launch)
            if used+ALLOWANCE>BUDGET or shutil.disk_usage(BASE.parent).free<RESERVE+ALLOWANCE:raise ValueError('episode storage allowance')
            active=run;r=collect(inv,OUTPUT,run,trial,launch['source_sha256'][PROTOCOL])
            commit=commit_episode(OUTPUT,run,inv.specification(trial),r)
            leaf=f'episode_{i:03d}_commit.json';write_json(OUTPUT/leaf,commit)
            commits[leaf]=digest(OUTPUT/leaf);results[run]=r;active=None
            used+=commit['artifact_bytes']+(OUTPUT/leaf).stat().st_size
            verify_artifacts(OUTPUT,commit['artifact_sha256'])
            if commit['absent_expected_artifacts']:raise ValueError('incomplete episode artifacts')
            if commit['artifact_bytes']>ALLOWANCE:raise ValueError('episode storage exceeded')
            precheck=audit_dynamic_condition(OUTPUT/run,inv.specification(trial),r,launch['source_sha256'][PROTOCOL])
            leaf=f'episode_{i:03d}_raw_precheck.json';write_json(OUTPUT/leaf,precheck)
            prechecks[leaf]=digest(OUTPUT/leaf);used+=(OUTPUT/leaf).stat().st_size
            if used>BUDGET:raise ValueError('total metadata/artifact budget exceeded')
            if commit['artifact_bytes']+(OUTPUT/f'episode_{i:03d}_commit.json').stat().st_size+(OUTPUT/leaf).stat().st_size>ALLOWANCE:
                raise ValueError('episode allowance including metadata exceeded')
            print('DYNAMIC_PRECHECK',i+1,run,precheck['report']['physical_visibility_pass'],
                precheck['report']['physical_stop'],precheck['report']['target_contact_positive'],flush=True)
            if r['acquisition_stop'] is not None:raise ValueError('acquisition failure; preserve case and stop pilot')
            # Strict depth failures are retained diagnostic outcomes, not a
            # changed qualification gate. All eight prescribed cases stay fixed.
        verify_ordered_launch(launch);validate_launch(launch,inv)
        write_json(OUTPUT/'result.json',dict(status='ORDERED_DYNAMIC_SENSOR_COLLECTION_COMPLETE',
            planned_runs=launch['planned_runs'],conditions=results,commits=commits,prechecks=prechecks,
            committed_bytes=used,uncommitted_run=None,training_eligibility_granted=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ORDERED_DYNAMIC_COLLECTION_FAILURE',reason=repr(error),
            planned_runs=launch['planned_runs'],conditions=results,commits=commits,prechecks=prechecks,
            committed_bytes=used,uncommitted_run=active,training_eligibility_granted=False,navigation_qualified=False,goal_achieved=False))
        raise


if __name__=='__main__':main()


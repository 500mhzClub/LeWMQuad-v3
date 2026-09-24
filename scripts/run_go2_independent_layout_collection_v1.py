"""One exact120-episode inventory layout batch; preserve every physical outcome."""
import shutil
import time
import cv2
import torch
import argparse
from lewm.independent_layout_collection_development import schedule,decision
from lewm.actuator_gain_development import configure_gains,read_gains
from lewm.support_friction_challenge_development import native_friction
from lewm.visual_led_motion_development import VisualLedMotion
from lewm_genesis.scene_builder import initialize_genesis,shutdown_genesis
from scripts.independent_layout_session_development import InventorySession
from scripts.pulse_context_setup_development import admit_context_setup
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.independent_layout_batch_development import (
    BATCHES,PROTOCOL,RESERVE,BATCH_BUDGET,EPISODE_ALLOWANCE,INVENTORY_ROOT,INVENTORY_IDS,
    inventory_bindings,load_inventory,output_root,validate_launch,commit_episode)
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.independent_layout_collection_audit_development import audit_condition

PILOT=BASE/'go2_independent_pulse_context_pilot_v1_attempt_001'
NEAR_FIELD_BENCH=BASE/'go2_near_field_visibility_probe_v1_attempt_001'
BENCH_IDS={
    'launch.json':'429c73911834c7c0243d9b68aa9297848ee248afa7f19ca5965b7c35e88c90e6',
    'result.json':'6eba751f612df0ad40d56b26fb4a62923528adfbfb3dd1fa897414641371504b',
}


def preflight(batch):
    output=output_root(batch)
    if output.exists() or output.is_symlink():raise ValueError('exclusive layout collection batch; no retry/resume')
    inventory=load_inventory()
    verify_artifacts(PILOT,{'launch.json':'bb06bb68d6cc8702c224d1a5b78d5b4285b2c3dcf6636e9c029f425d0519319d'})
    old=read_json(PILOT,'launch.json');verify(old)
    verify_artifacts(NEAR_FIELD_BENCH,BENCH_IDS)
    bench_launch=read_json(NEAR_FIELD_BENCH,'launch.json');verify(bench_launch)
    bench=read_json(NEAR_FIELD_BENCH,'result.json');verify_artifacts(NEAR_FIELD_BENCH,bench['artifact_sha256'])
    if bench['status']!='NATIVE_NEAR_FIELD_BENCH_COMPLETE' or bench['all_expected_outcomes_observed'] is not True:
        raise ValueError('completed fixed native visibility bench required before new collection')
    inherited=read_json(INVENTORY_ROOT,'launch.json')['source_sha256'];verify_bindings(inherited)
    sources=discover_sources((PROTOCOL,'scripts/run_go2_independent_layout_collection_v1.py',
        'scripts/audit_go2_independent_layout_collection_v1.py',
        'lewm/tests/test_independent_layout_batch_development.py',
        'lewm/tests/test_independent_layout_collection_adapter_development.py',
        'lewm/tests/test_near_field_capture_integration_development.py',
        'lewm/tests/test_physical_first_surface_depth_development.py'),inherited)
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    ids=inventory.episode_ids(batch)
    launch.update(source_sha256=sources,input_sha256=old['input_sha256']|inventory_bindings(),batch=batch,
        output_root=str(output),planned_trials=list(ids),conditions={c:inventory.specification(c) for c in ids},
        inventory_sha256=INVENTORY_IDS['inventory.json'],maximum_batch_bytes=BATCH_BUDGET,
        episode_storage_allowance_bytes=EPISODE_ALLOWANCE,minimum_free_bytes=RESERVE,
        maximum_command_ticks=33,maximum_episode_physics_samples=2400,maximum_episode_rgbd_frames=34,
        role=inventory.specification(ids[0])['data_role'],tracker_required_for_commands=False,
        near_field_bench_root=str(NEAR_FIELD_BENCH),near_field_bench_sha256=BENCH_IDS,
        model_training=False,real_time_qualified=False,hidden_robot_ideal_camera=True,navigation_qualified=False,goal_achieved=False)
    validate_launch(launch,inventory,batch);verify(launch)
    if shutil.disk_usage(BASE.parent).free<RESERVE+BATCH_BUDGET:raise ValueError('8GiB batch allowance plus40GiB reserve required')
    return inventory,launch


def collect(inventory,output,trial,definition):
    directory=output/trial;directory.mkdir();spec=inventory.specification(trial);write_json(directory/'specification.json',spec)
    session=None;rows=[];tape=[];friction=[];stop=None;acquisition_stop=None;terminal=None;admitted=False
    initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
    try:
        session=InventorySession(inventory,spec,directory);session.install_contact_identity();build=session.ctx.build
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
        print('LAYOUT_EPISODE',trial,result,flush=True);return result
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                write_json(directory/'context_decisions.json',rows);write_json(directory/'command_tape.json',tape)
                write_json(directory/'native_guard_rows.json',session.guard_rows);write_json(directory/'friction_checks.json',friction)
            finally:session.ctx.build.scene.destroy()
        shutdown_genesis()


def run_batch(batch):
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    inventory,launch=preflight(batch);output=output_root(batch)
    create_output(output);write_json(output/'launch.json',launch)
    results={};commits={};prechecks={};active=None;used=(output/'launch.json').stat().st_size
    try:
        for i,c in enumerate(launch['planned_trials']):
            verify(launch)
            if used+EPISODE_ALLOWANCE>BATCH_BUDGET:raise ValueError('fixed batch storage allowance exhausted')
            if shutil.disk_usage(BASE.parent).free<RESERVE+EPISODE_ALLOWANCE:raise ValueError('next episode storage reserve')
            active=c;r=collect(inventory,output,c,launch['source_sha256'][PROTOCOL])
            commit=commit_episode(output,inventory.specification(c),r)
            verify_artifacts(output,commit['artifact_sha256'])
            leaf=f'episode_{i:03d}_commit.json';write_json(output/leaf,commit)
            commits[leaf]=digest(output/leaf);results[c]=r
            used+=commit['artifact_bytes']+(output/leaf).stat().st_size;active=None
            print('BATCH_COMMITTED',batch,i+1,len(launch['planned_trials']),c,used,flush=True)
            if commit['absent_expected_artifacts']:raise ValueError('episode artifact coverage incomplete; preserve partial commit')
            if commit['artifact_bytes']>EPISODE_ALLOWANCE:raise ValueError('episode exceeded prospective storage allowance')
            report,prefix,window,targets=audit_condition(output/c,inventory.specification(c),r,launch['source_sha256'][PROTOCOL])
            leaf=f'episode_{i:03d}_raw_precheck.json'
            write_json(output/leaf,dict(report=report,prefix=prefix,window=window,targets=targets))
            prechecks[leaf]=digest(output/leaf);used+=(output/leaf).stat().st_size
            print('BATCH_RAW_PRECHECK',batch,i+1,c,report['departure_present'],report['target_contact_positive'],flush=True)
            episode_bytes=commit['artifact_bytes']+(output/f'episode_{i:03d}_commit.json').stat().st_size+(output/leaf).stat().st_size
            if episode_bytes>EPISODE_ALLOWANCE or used>BATCH_BUDGET:raise ValueError('episode/batch allowance including audit metadata exceeded')
            if report['physical_visibility_pass'] is False:
                raise ValueError('physical visibility failure; preserve raw precheck and do not collect next episode')
        verify(launch);validate_launch(launch,inventory,batch)
        write_json(output/'result.json',dict(status='LAYOUT_COLLECTION_COMPLETE',batch=batch,planned_trials=launch['planned_trials'],
            conditions=results,commits=commits,prechecks=prechecks,committed_bytes=used,uncommitted_trial=None,
            role=launch['role'],model_trained=False,navigation_qualified=False,goal_achieved=False))
    except Exception as error:
        write_json(output/'failure.json',dict(status='TERMINAL_LAYOUT_COLLECTION_FAILURE',reason=repr(error),batch=batch,
            planned_trials=launch['planned_trials'],conditions=results,commits=commits,prechecks=prechecks,committed_bytes=used,
            uncommitted_trial=active,model_trained=False,navigation_qualified=False,goal_achieved=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',choices=BATCHES,required=True)
    run_batch(parser.parse_args().batch)


if __name__=='__main__':main()

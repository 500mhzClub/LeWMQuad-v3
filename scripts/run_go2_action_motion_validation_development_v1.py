"""One B validation acquisition against the frozen A-only fit; no retry."""
import json
import time

from lewm.action_motion_identification_development import MotionState
from lewm.action_response_validation_development import MotionValidationController
from lewm.action_response_model_development import model_identity
from scripts.fit_go2_action_response_development_v1 import OUTPUT as FIT
from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.causal_sensor_state import SensorContractError
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.action_motion_identification_session_development import MotionSession, admit_motion_setup
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_startup_observation_turn_development_v1 import (
    artifact_names as inherited_artifacts, specification as inherited_specification, verify_extensions)
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_observation_turn_session_development import latest_policy
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_action_motion_validation_development_v1_attempt_001'
PROTOCOL='docs/go2_action_motion_validation_development_v1_2026-09-06.md'
FIT_IDENTITIES={
    'launch.json':'9cb3cfc1e0da625c940e00f3c37ecb91a6f3cbe309af96a29b0c40cbcd0f656f',
    'result.json':'5e61b02d27258da30bcc7439bf6f8d1b496d5b8f200c7756a864aab39ba6cf3e',
    'model.json':'090c0227e50969db07b6083f23ea76e11a20276c03a54a972e8813f0f81f0f3f',
    'training_rows.json':'819244cb39f280afbe54f2bb972506d9a79a7172a6d2816f2afa8a6a55ea8639'}
MODEL_IDENTITY='119b3612887ec19e293d402b0991c51459483d9cbe369ae7cc5815cdd31258c1'
SEEDS=('scripts/run_go2_action_motion_validation_development_v1.py',
       'scripts/audit_go2_action_motion_validation_development_v1.py',
       'lewm/tests/test_action_response_validation_development.py',PROTOCOL)


def specification():
    spec=inherited_specification()
    return spec | dict(scene_id='go2-action-motion-validation-development-v1',
        family='BOUNDED_ACTION_MOTION_VALIDATION',procedural_seed=2026090604)


def frozen_model():
    verify_bindings({str((FIT/n).relative_to(ROOT)):h for n,h in FIT_IDENTITIES.items()})
    model=json.loads((FIT/'model.json').read_text())
    if model_identity(model)!=MODEL_IDENTITY: raise ValueError('frozen model content mismatch')
    return model


def collect(definition,model):
    initialize_genesis(backend='cpu',seed=2026090604,logging_level='warning')
    session=None; owner=None; controller=None; decisions=[]; tape=[]; tail_states=[]; timings=[]
    physical_stop=None; sensor_stop=None; tail=0; outer_timings=[]
    try:
        session=MotionSession(specification(),OUTPUT); session.install_contact_identity()
        build=session.ctx.build
        gains=configure_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        write_json(OUTPUT/'actuator_identity.json',gains); write_json(OUTPUT/'floor_domain.json',build.floor_domain)
        write_json(OUTPUT/'floor_roles.json',dict(
            physical_ground_link_ids=[int(l.idx) for l in build.collision_floor.links],
            visual_only_link_ids=[int(l.idx) for l in build.visual_floor.links],
            physical_ground_geom_ids=[int(g.idx) for g in build.collision_floor.geoms],
            visual_collision_geom_count=len(build.visual_floor.geoms)))
        def step(command,phase,index):
            session.phase=phase
            entry=dict(tick=len(tape),pre_sample_index=len(session.samples)-1,
                requested_command=list(command),phase=phase,decision_index=index)
            tape.append(entry); start=time.perf_counter()
            try: session.command_tick(command)
            finally:
                entry['post_sample_index']=len(session.samples)-1
                entry['execution_wall_ms']=1000*(time.perf_counter()-start)
        try:
            session.settle_recorded(); session.capture_current()
            geometry,velocity,region,admission=admit_motion_setup(session,definition)
            owner=MotionState(geometry,velocity_prior=velocity,region_prior=region,admission=admission)
            controller=MotionValidationController(owner,model,MODEL_IDENTITY)
            for index in range(61):
                outer_start=time.perf_counter_ns()
                timing=dict(decision_index=index,captures_before=len(session.packet_rows),
                    start_perf_counter_ns=outer_start,observation_index=None,decision_recorded=False,
                    command_tick_attempted=False,completed_without_exception=False)
                try:
                    session.capture_current(); policy=latest_policy(session); now=int(policy['sensor_state']['decision_ns'])
                    timing['observation_index']=len(session.packet_rows)-1
                    timing['fresh_capture_inside_loop']=len(session.packet_rows)>timing['captures_before']
                    start=time.perf_counter()
                    decision=controller.observe(policy,session.latest_depth,session.fast_buffer.packet(now_ns=now),now_ns=now)
                    timings.append(dict(observation_index=len(session.packet_rows)-1,controller_wall_ms=1000*(time.perf_counter()-start)))
                    decisions.append(dict(observation_index=len(session.packet_rows)-1,decision=decision))
                    timing['decision_recorded']=True
                    print(json.dumps(dict(index=index,time_ns=now,status=decision['status'],
                        owner_status=decision['state']['status'],motion_index=decision['motion_index'],
                        command=decision['requested_command'])),flush=True)
                    if not decision['terminal']:
                        if index==60: raise SensorContractError('bounded validation decision limit')
                        timing['command_tick_attempted']=True
                        step(decision['requested_command'],decision['phase'],index)
                    timing['completed_without_exception']=True
                    if decision['terminal']: break
                finally:
                    timing['end_perf_counter_ns']=time.perf_counter_ns()
                    timing['outer_wall_ms']=(timing['end_perf_counter_ns']-outer_start)/1e6
                    outer_timings.append(timing)
            for _ in range(3):
                step((0.,0.,0.),4,None); tail+=1; session.capture_current()
                if owner is not None and not owner.status.startswith('FAILED_'):
                    policy=latest_policy(session); now=int(policy['sensor_state']['decision_ns'])
                    state=owner.observe(policy,session.latest_depth,session.fast_buffer.packet(now_ns=now),now_ns=now)
                    tail_states.append(dict(observation_index=len(session.packet_rows)-1,state=state))
        except PhysicalStop as error:
            physical_stop=str(error)  # No further physics after a native stop.
        except SensorContractError as error:
            sensor_stop=str(error)  # Unexpected contract error: retain partial record, no implicit restart.
        if session.samples:
            native=capture_native_robot_geometry(build.robot)
            write_json(OUTPUT/'terminal_native_robot_geometry.json',native)
            if session.motion_setup is not None:
                geometry=session.motion_setup[0]; raw=session.samples[-1]
                write_json(OUTPUT/'terminal_foot_identity.json',match_native_foot_geometries(native,geometry,
                    raw['joint_position'],raw['base_pose_world']))
        terminal=read_gains(build.robot,session.ctx.runner._leg_dof_idx.tolist())
        write_json(OUTPUT/'terminal_actuator_gains.json',terminal)
        if terminal!=gains['effective']: raise ValueError('actuator gain drift')
        return dict(physics_samples=len(session.samples),rgbd_frames=len(session.packet_rows),
            relative_observer_frames=len(session.relative_observations),controller_decisions=len(decisions),
            requested_ticks=len(tape),stopping_tail_ticks=tail,physical_stop_reason=physical_stop,
            sensor_stop_reason=sensor_stop,controller_status=controller.status if controller else None,
            owner_status=owner.status if owner else None,motion_commands_requested=controller.index if controller else 0,
            navigation_qualified=False,real_time_qualified=False,validation_run_launched=True,model_canonical_sha256=MODEL_IDENTITY)
    finally:
        try:
            if session is not None:
                try:
                    session.persist(OUTPUT); session.persist_observations(OUTPUT)
                    write_json(OUTPUT/'motion_decisions.json',decisions)
                    write_json(OUTPUT/'motion_command_tape.json',tape)
                    write_json(OUTPUT/'motion_tail_states.json',tail_states)
                    write_json(OUTPUT/'motion_timings.json',dict(captures=session.capture_timings,controller=timings,outer=outer_timings))
                    write_json(OUTPUT/'startup_guard_rows.json',session.startup_guard_rows)
                    write_json(OUTPUT/'motion_region_rows.json',session.motion_region_rows)
                finally: session.ctx.build.scene.destroy()
        finally: shutdown_genesis()


def artifact_names(frames):
    excluded={'startup_decisions.json','startup_command_tape.json','startup_timings.json'}
    return [p for p in inherited_artifacts(frames) if p not in excluded]+[
        'motion_setup_checks.json','motion_admission.json','motion_decisions.json','motion_command_tape.json',
        'motion_tail_states.json','motion_timings.json','motion_region_rows.json']


def preflight():
    bindings={str((FIT/n).relative_to(ROOT)):h for n,h in FIT_IDENTITIES.items()}
    verify_bindings(bindings); frozen_model()
    old=json.loads((FIT/'launch.json').read_text()); result=json.loads((FIT/'result.json').read_text())
    if result['status']!='A_ONLY_SENSOR_RESPONSE_MODEL_FROZEN' or result['independent_validation_complete']:
        raise ValueError('A-only frozen fit required')
    inputs=old['input_sha256'] | bindings
    sources=discover_sources(SEEDS,old['source_sha256'])
    verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256']); verify_extensions(old['native_geometry_sha256'])
    return dict(source_sha256=sources,input_sha256=inputs,native_sha256=old['native_sha256'],
        native_geometry_sha256=old['native_geometry_sha256'],specification=specification(),
        model_canonical_sha256=MODEL_IDENTITY,
        scope='one new fixed B command validation; no refit, retry, hardware or navigation qualification')


def main():
    if OUTPUT.exists(): raise ValueError('fresh fixed validation output required; no retry or resume')
    launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        start=time.perf_counter(); row=collect(launch['source_sha256'][PROTOCOL],frozen_model())
        row['total_acquisition_wall_ms']=1000*(time.perf_counter()-start)
        verify_bindings(launch['source_sha256'] | launch['input_sha256'])
        verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
        names=artifact_names(row['rgbd_frames'])
        write_json(OUTPUT/'result.json',dict(status='ACQUISITION_COMPLETE_AUDIT_REQUIRED',**row,
            artifact_sha256={p:digest(OUTPUT/p) for p in names if (OUTPUT/p).is_file()},
            absent_expected_artifacts=[p for p in names if not (OUTPUT/p).is_file()]))
        print('ACTION_MOTION_VALIDATION_ACQUISITION_COMPLETE',flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',dict(status='TERMINAL_FAILURE',error=repr(error),navigation_qualified=False))
        raise


if __name__=='__main__': main()

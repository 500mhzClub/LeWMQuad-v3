"""Fresh bounded identification acquisition. No validation-run or retry entry point."""
import json
import time

from lewm.action_motion_identification_development import MotionState, MotionIdentificationController
from lewm.actuator_gain_development import configure_gains, read_gains
from lewm.causal_sensor_state import SensorContractError
from lewm.native_foot_geometry_evaluation_development import match_native_foot_geometries
from lewm_genesis.scene_builder import initialize_genesis, shutdown_genesis
from scripts.action_motion_identification_session_development import MotionSession, admit_motion_setup
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.capture_native_robot_geometry_development import capture_native_robot_geometry
from scripts.probe_go2_continuous_startup_handoff_development import IDENTITIES
from scripts.run_go2_aligned_floor_interface_development_v1 import verify_native_bindings
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.run_go2_startup_observation_turn_development_v1 import (
    OUTPUT as PREVIOUS, artifact_names as inherited_artifacts, specification as inherited_specification, verify_extensions)
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_observation_turn_session_development import latest_policy
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_action_motion_identification_development_v1_attempt_001'
PROTOCOL='docs/go2_action_motion_identification_development_v1_2026-09-06.md'
PREDECESSOR='docs/go2_factored_trajectory_recorded_diagnostic_final_result_2026-09-06.json'
PREDECESSOR_SHA='7710a2398f0d120ba3ae3dbbbab6f0481aa371021e45397ade73dd1c36c6e749'
SEEDS=('scripts/run_go2_action_motion_identification_development_v1.py',
       'scripts/audit_go2_action_motion_identification_development_v1.py',
       'lewm/tests/test_action_motion_identification_development.py',PROTOCOL)


def specification():
    spec=inherited_specification()
    return spec | dict(scene_id='go2-action-motion-identification-development-v1',
        family='BOUNDED_ACTION_MOTION_IDENTIFICATION',procedural_seed=2026090603)


def collect(definition):
    initialize_genesis(backend='cpu',seed=2026090603,logging_level='warning')
    session=None; owner=None; controller=None; decisions=[]; tape=[]; tail_states=[]; timings=[]
    physical_stop=None; sensor_stop=None; tail=0
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
            controller=MotionIdentificationController(owner)
            for index in range(61):
                session.capture_current(); policy=latest_policy(session); now=int(policy['sensor_state']['decision_ns'])
                start=time.perf_counter()
                decision=controller.observe(policy,session.latest_depth,session.fast_buffer.packet(now_ns=now),now_ns=now)
                timings.append(dict(observation_index=len(session.packet_rows)-1,controller_wall_ms=1000*(time.perf_counter()-start)))
                decisions.append(dict(observation_index=len(session.packet_rows)-1,decision=decision))
                print(json.dumps(dict(index=index,time_ns=now,status=decision['status'],
                    owner_status=decision['state']['status'],motion_index=decision['motion_index'],
                    command=decision['requested_command'])),flush=True)
                if decision['terminal']: break
                if index==60: raise SensorContractError('bounded identification decision limit')
                step(decision['requested_command'],decision['phase'],index)
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
            navigation_qualified=False,real_time_qualified=False,validation_run_launched=False)
    finally:
        try:
            if session is not None:
                try:
                    session.persist(OUTPUT); session.persist_observations(OUTPUT)
                    write_json(OUTPUT/'motion_decisions.json',decisions)
                    write_json(OUTPUT/'motion_command_tape.json',tape)
                    write_json(OUTPUT/'motion_tail_states.json',tail_states)
                    write_json(OUTPUT/'motion_timings.json',dict(captures=session.capture_timings,controller=timings))
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
    bindings={str((PREVIOUS/p).relative_to(ROOT)):h for p,h in IDENTITIES.items()} | {PREDECESSOR:PREDECESSOR_SHA}
    verify_bindings(bindings)
    old=json.loads((PREVIOUS/'launch.json').read_text()); result=json.loads((PREVIOUS/'result.json').read_text())
    witness=json.loads((ROOT/PREDECESSOR).read_text())
    inherited=old['source_sha256'] | witness['source_sha256']
    inputs=old['input_sha256'] | bindings | {str((PREVIOUS/p).relative_to(ROOT)):h for p,h in result['artifact_sha256'].items()}
    sources=discover_sources(SEEDS,inherited)
    verify_bindings(sources | inputs); verify_native_bindings(old['native_sha256']); verify_extensions(old['native_geometry_sha256'])
    return dict(source_sha256=sources,input_sha256=inputs,native_sha256=old['native_sha256'],
        native_geometry_sha256=old['native_geometry_sha256'],specification=specification(),
        scope='one new bounded identification acquisition; no validation, retry, hardware or navigation qualification')


def main():
    if OUTPUT.exists(): raise ValueError('fresh fixed identification output required; no retry or resume')
    launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        start=time.perf_counter(); row=collect(launch['source_sha256'][PROTOCOL])
        row['total_acquisition_wall_ms']=1000*(time.perf_counter()-start)
        verify_bindings(launch['source_sha256'] | launch['input_sha256'])
        verify_native_bindings(launch['native_sha256']); verify_extensions(launch['native_geometry_sha256'])
        names=artifact_names(row['rgbd_frames'])
        write_json(OUTPUT/'result.json',dict(status='ACQUISITION_COMPLETE_AUDIT_REQUIRED',**row,
            artifact_sha256={p:digest(OUTPUT/p) for p in names if (OUTPUT/p).is_file()},
            absent_expected_artifacts=[p for p in names if not (OUTPUT/p).is_file()]))
        print('ACTION_MOTION_IDENTIFICATION_ACQUISITION_COMPLETE',flush=True)
    except Exception as error:
        write_json(OUTPUT/'result.json',dict(status='TERMINAL_FAILURE',error=repr(error),navigation_qualified=False))
        raise


if __name__=='__main__': main()

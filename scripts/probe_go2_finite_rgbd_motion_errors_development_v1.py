"""Finite member replay first; native scoring after predictions are persisted."""
from dataclasses import asdict
import time

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface
from lewm.finite_rgbd_error_members_development import FiniteRGBDMember, fixed_members
from lewm.paired_rgbd_physical_plane_development import minimum_gaps
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_shadow_motion_development import priors
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_bounded_depth_surface_development_v1 import OUTPUT as PREVIOUS, HYPOTHESES
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_rgbd_shadow_motion_development_v1 import OUTPUT as INPUT, PROTOCOL as INPUT_PROTOCOL, ARMS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_finite_rgbd_motion_errors_development_v1_attempt_001'
PROTOCOL = 'docs/go2_finite_rgbd_motion_errors_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/probe_go2_finite_rgbd_motion_errors_development_v1.py',
         'lewm/tests/test_finite_rgbd_error_members_development.py')
PREVIOUS_IDS = {'launch.json': 'b9f3d5f1db71e53b92036c3c687597c65919eae630970d2b9c87bc90f2ed1cd5',
    'result.json': 'f4d239b524e62f4c69fb7194fcc2d84353b866a65ed39a94e171d12ff35f980f',
    'endpoint_audit.json': '488853ba08c9e315a63ec0530f320f54d45899156317e0bceceb0f0f7381283c'}
INPUT_IDS = {'launch.json': '628168df54a4e8d96575f939609057f610de35d60fd579cd58d76ca58dd964dc',
    'result.json': '4a5b97bb50f28ca631bdb1bb1e3ee953658105613742333c638429b0c482bfd1',
    'raw_artifact_audit.json': 'dffa2ff61d64d32d54de92b9d445b6cc21e501649d368b657987a0459bfad42f'}


def preflight():
    bound = {str((p/n).relative_to(ROOT)): h for p, identities in ((PREVIOUS, PREVIOUS_IDS), (INPUT, INPUT_IDS))
             for n, h in identities.items()}
    verify_bindings(bound)
    old = read_json(PREVIOUS, 'launch.json'); verify(old)
    for path in (PREVIOUS, INPUT):
        result = read_json(path, 'result.json')
        bound |= {str((path/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    audit = read_json(PREVIOUS, 'endpoint_audit.json'); verify_bindings(audit['source_sha256']|audit['input_sha256'])
    raw_audit = read_json(INPUT, 'raw_artifact_audit.json')
    bound |= audit['source_sha256'] | raw_audit['auditor_source_sha256']
    sources = discover_sources(SEEDS, old['source_sha256'])
    launch = old | dict(source_sha256=sources, input_sha256=old['input_sha256']|bound,
        diagnostic_protocol=PROTOCOL, finite_members=[asdict(c) for c in fixed_members()],
        motion_arms=list(ARMS), surface_query_frames=[0, 27, 53],
        scope='finite independent-development replay; no covariance, calibration, physics or navigation permission')
    verify(launch); return launch


def replay(arm, launch):
    directory = INPUT/arm; saved = read_json(directory, 'shadow_observations.json')
    prior, _ = priors(launch['source_sha256'][INPUT_PROTOCOL])
    models = {c.name: FiniteRGBDMember(c, prior) for c in fixed_members()}
    surfaces = {}; surface_records = {}; rows = []; geometry = ArticulatedCollisionGeometry(URDF)
    shape_ids = [s['shape_id'] for s in geometry.supports(np.repeat([0., .8, -1.5], 4), np.eye(3))['shapes']]
    start = time.perf_counter()
    assert len(saved) == 54
    for index, expected in enumerate(saved):
        policy, depth = load_rgbd_observation(directory, index); fast = load_fast_packet(directory, index)
        results = {}
        for name, member in models.items():
            full = member.observe(policy, depth, fast)
            if name == 'nominal': exact(plain(full), expected['shadow'])
            row = {k: v for k, v in full.items() if k != 'state'}
            state = full['state']; row['state'] = None
            if state is not None:
                rotation = state['depth_state']['relative_orientation']['rotation_initial_body_from_current_body']
                row['state'] = dict(fusion=state['fusion'], rotation_initial_body_from_current_body=rotation,
                    depth_motion=state['depth_state']['motion'], point_motion=state['point_state']['motion'])
                if index == 0:
                    _, d, _ = member.last_packet; gravity = member.observer.model.state.integrator.gravity
                    surface = BoundedDepthSurface(d['depth_m'], d['valid'], gravity/9.81, **HYPOTHESES)
                    surfaces[name] = surface
                    surface_records[name] = dict(status=surface.status, seed_cell_rc=surface.seed,
                        diagnostics=surface.diagnostics, raw_depth_sha256=surface.frame.depth_sha256,
                        anchor=surface.anchor, normal=surface.normal)
                if index in (0, 27, 53):
                    surface = surfaces[name]
                    row['surface_status'] = surface.status
                    if surface.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE':
                        q = policy['sensor_state']['sensed']['joints']['values'][-1, :12]
                        row['surface_query'] = surface.query(geometry, q,
                            rotation_observation_from_body=rotation,
                            translation_observation_from_body=state['fusion']['position_initial_body_m'],
                            point_error_by_shape=dict.fromkeys(shape_ids, 0.))
            results[name] = plain(row)
        rows.append(dict(observation_index=index, measured_ns=int(policy['sensor_state']['decision_ns']), members=results))
        if index % 10 == 0:
            print(arm, index+1, 'admitted',sum(r['state'] is not None for r in results.values()), flush=True)
    return plain(dict(arm=arm, definitions=[asdict(c) for c in fixed_members()], rows=rows,
        initial_surfaces=surface_records, shape_ids=shape_ids, exact_nominal_frames=54,
        elapsed_wall_s=time.perf_counter()-start, commands_selected=False, error_population_calibrated=False))


def score(arm, predictions):
    # Caller persists predictions before entering this function. Native states
    # are never passed into a member, initial surface, or predicted query.
    directory = INPUT/arm; cameras = read_json(directory, 'camera_audit.json')
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as raw:
        poses = raw['base_pose_world'].copy(); ns = np.rint(raw['timestamp_s']*1e9).astype(np.int64)
        joints = raw['joint_position'].copy()
    assert len(cameras) == len(predictions['rows']) == 54
    assert [c['physical_sample_index'] for c in cameras] == [749+50*i for i in range(54)]
    initial = poses[749]; R0 = rotation_xyzw(initial[3:]); geometry = ArticulatedCollisionGeometry(URDF)
    summaries = {}; details = []
    for definition in predictions['definitions']:
        name = definition['name']; records = []; statuses = []; ranks = {}; complemented = fallback = rejected = 0
        for frame, row in enumerate(predictions['rows']):
            sample = cameras[frame]['physical_sample_index']; assert row['measured_ns'] == ns[sample]
            member = row['members'][name]; statuses.append(member['status'])
            if member['state'] is None: continue
            state = member['state']; fusion = state['fusion']
            ptrue = R0.T@(poses[sample, :3]-initial[:3]); Rtrue = R0.T@rotation_xyzw(poses[sample, 3:])
            Rest = np.asarray(state['rotation_initial_body_from_current_body'])
            entry = dict(frame=frame, position_error_m=float(np.linalg.norm(np.asarray(fusion['position_initial_body_m'])-ptrue)),
                orientation_error_rad=float(np.arccos(np.clip((np.trace(Rest.T@Rtrue)-1)/2, -1, 1))))
            if frame:
                rank = str(fusion['depth_rank']); ranks[rank] = ranks.get(rank, 0)+1
                complemented += int(fusion['constraints']['point_used_for_weak_directions'])
                fallback += int(fusion['kind'] == 'INERTIALLY_PREDICTED_WEAK_COMPONENT')
                rejected += int(state['point_motion']['status'] != 'CONDITIONAL_RGBD_POINT_TRANSLATION')
            if 'surface_query' in member:
                surface = predictions['initial_surfaces'][name]; query = member['surface_query']
                reference = minimum_gaps(geometry, joints[sample], surface['anchor'], surface['normal'], Rtrue, ptrue)
                estimate = np.array([g['nominal_minimum_gap_m'] for g in query['gap_bounds']['primitives']])
                entry['pose_only_reference_gap_error_m'] = np.abs(estimate-reference).tolist()
                entry['covered_physical_shapes'] = sum(query['floor_coverage'].values())
                entry['unobserved_reference_gap_is_not_floor_clearance'] = True
            records.append(entry)
        failures = [r['members'][name]['failure'] for r in predictions['rows'] if r['members'][name]['status']=='TERMINAL_SHADOW_FAILURE']
        assert len(failures) <= 1
        summaries[name] = dict(admitted=len(records), terminal_failures=len(failures),
            not_reinvoked=statuses.count('NOT_REINVOKED_AFTER_SHADOW_FAILURE'), failure=failures[0] if failures else None,
            depth_rank_counts=ranks, point_complemented=complemented, inertial_fallback=fallback, point_rejected=rejected,
            maximum_admitted_position_error_m=max((r['position_error_m'] for r in records), default=None),
            maximum_admitted_orientation_error_rad=max((r['orientation_error_rad'] for r in records), default=None))
        details.append(dict(member=name, admitted_rows=records))
    return dict(arm=arm, members=summaries, details=details, native_pose_scoring_only=True,
        finite_member_extrema_are_not_error_bounds=True, navigation_qualified=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh exclusive finite-error diagnostic only')
    cv2.setNumThreads(1); launch = preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    completed = []; summary = {}
    try:
        for arm in ARMS:
            predictions = replay(arm, launch); name = arm+'_predictions.json'
            write_json(OUTPUT/name, predictions); completed.append(name)
            evaluated = score(arm, predictions); name = arm+'_evaluation.json'
            write_json(OUTPUT/name, evaluated); completed.append(name)
            summary[arm] = evaluated['members']; verify(launch)
            print('ARM_COMPLETE', arm, flush=True)
        result = dict(status='FINITE_RGBD_MOTION_ERROR_DIAGNOSTIC_COMPLETE', arms=summary,
            artifact_sha256={n: digest(OUTPUT/n) for n in completed}, independent_new_trials=0,
            physics_executed=False, uncertainty_calibrated=False, navigation_qualified=False, goal_achieved=False)
        write_json(OUTPUT/'result.json', result); print(result['status'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_FINITE_RGBD_DIAGNOSTIC_FAILURE', reason=str(error),
            completed_artifacts={n: digest(OUTPUT/n) for n in completed}))
        raise


if __name__ == '__main__': main()

"""Frozen saved-sensor configuration diagnostic; no new physics or permission."""
from copy import deepcopy
import json
import math
import time

import cv2
import numpy as np

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.depth_proposal_navigation_development import DepthProposalNavigation
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_physical_configuration_evidence_development import RGBDPhysicalConfigurationEvidence
from lewm.rgbd_shadow_motion_development import POINT_HYPOTHESES
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import IDENTITIES as PREVIOUS_IDENTITIES, exact
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.fresh_maze_session_development import priors
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_fresh_fused_maze_development_v1 import OUTPUT as INPUT, PROTOCOL as INPUT_PROTOCOL
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_rgbd_physical_configuration_evidence_development_v1_attempt_001'
PROTOCOL = 'docs/go2_rgbd_physical_configuration_evidence_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/probe_go2_rgbd_physical_configuration_evidence_development_v1.py',
         'lewm/tests/test_rgbd_physical_configuration_evidence_development.py')
ERRORS = dict(normal_error=.002, up_error=.001, plane_offset_error=.001, range_error_m=.001)
POINT_ERROR_M = .005
YAW_DEGREES = (0, 30, -30, 60, -60, 90, -90)


def preflight():
    identities = {n+'.json': h for n, h in PREVIOUS_IDENTITIES.items()} | {
        'raw_artifact_audit.json': '8275b9edb0cc5a201f07f3f10b86db51f3738390e1fdfcd9ac35686f95ecb064'}
    bindings = {str((INPUT/n).relative_to(ROOT)): h for n, h in identities.items()}
    verify_bindings(bindings)
    old = read_json(INPUT, 'launch.json'); verify(old)
    result = read_json(INPUT, 'result.json'); audit = read_json(INPUT, 'raw_artifact_audit.json')
    verify_bindings(audit['auditor_source_sha256'])
    bindings |= {str((INPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    sources = discover_sources(SEEDS, old['source_sha256'])
    launch = old | dict(source_sha256=sources, input_sha256=old['input_sha256'] | bindings,
        configuration_yaws_degrees=list(YAW_DEGREES), configuration_point_error_m=POINT_ERROR_M,
        configuration_sensor_error_hypotheses=ERRORS, diagnostic_protocol=PROTOCOL,
        scope='one recorded terminal configuration/yaw diagnostic; no new motion, calibration or mission qualification')
    verify(launch)
    return launch


def replay_owner(launch):
    directory = INPUT/'mission'; saved = read_json(directory, 'task_decisions.json')
    velocity, _ = priors(launch['source_sha256'][INPUT_PROTOCOL])
    geometry = ArticulatedCollisionGeometry(URDF)
    controller = DepthProposalNavigation(geometry, memory_arm='episodic', prior=velocity, hypotheses=POINT_HYPOTHESES)
    for i, row in enumerate(saved):
        policy, depth = load_rgbd_observation(directory, i); now = policy['sensor_state']['decision_ns']
        exact(controller.observe_rgbd(policy, load_fast_packet(directory, i), depth, now_ns=now), row['controller'])
        if i % 40 == 0: print('EXACT_OWNER_REPLAY '+str(i+1), flush=True)
    assert controller.status == 'FAILED_LOCAL_FAILED_UNOBSERVED_TURN_VOLUME'
    return controller, geometry, policy, now


def summary(row):
    primitives = row['primitives']
    return dict(primitives=len(primitives),
        nonfloor_clear=sum(r['conditional_nonfloor_clearance'] for r in primitives),
        nonfloor_conflicted=sum(bool(r['nonfloor_conflict_sources']) for r in primitives),
        ground_separated=sum(r['ground']['physical_floor_separation_observed'] for r in primitives),
        covered_penetrated=sum(bool(r['ground']['observed_penetration_sources']) for r in primitives),
        incompatible_ground=sum(bool(r['ground']['incompatible_covered_plane_pairs']) for r in primitives),
        ground_unknown=sum(r['ground']['status'] == 'UNKNOWN_FLOOR_COVERAGE' for r in primitives),
        contact_candidates_with_nonfloor_clearance=sum(r['contact_candidate_with_nonfloor_clearance'] for r in primitives),
        fully_separated=sum(r['conditional_observed_separation'] for r in primitives),
        navigation_action_permitted=row['navigation_action_permitted'])


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('fresh fixed diagnostic only; no overwrite')
    cv2.setNumThreads(1); launch = preflight()
    OUTPUT.mkdir(); write_json(OUTPUT/'launch.json', launch)
    try:
        controller, geometry, policy, now = replay_owner(launch)
        owner = controller.sensor_memory
        before = deepcopy(owner.rays.fusion), owner.state.depth.orientation.samples_integrated
        evidence = RGBDPhysicalConfigurationEvidence(owner, geometry, **ERRORS)
        start = time.perf_counter_ns(); preparation = evidence.refresh(now_ns=now)
        preparation_ms = (time.perf_counter_ns()-start)/1e6
        q = policy['sensor_state']['sensed']['joints']['values'][-1, :12]
        records = []; names = []
        for i, degrees in enumerate(YAW_DEGREES):
            R = rotation_increment(owner.rays.rotation.T@owner.rays.up_initial*math.radians(degrees))
            start = time.perf_counter_ns()
            row = evidence.query([0., 0., 0.], R, q, POINT_ERROR_M, now_ns=now)
            elapsed = (time.perf_counter_ns()-start)/1e6
            # Full independent beam and floor-footprint reference for the
            # exact failed current configuration; other yaws remain diagnostics.
            if i == 0:
                reference = evidence.query([0., 0., 0.], R, q, POINT_ERROR_M, now_ns=now, backend='reference')
                exact(row, json.loads(json.dumps(reference, allow_nan=False)))
            name = f'configuration_{i:02d}.json'
            write_json(OUTPUT/name, dict(yaw_degrees=degrees, observation_ns=now, configuration=row,
                query_wall_ms=elapsed, full_reference_checked=i == 0))
            names.append(name); records.append(dict(yaw_degrees=degrees, query_wall_ms=elapsed, **summary(row)))
            print(json.dumps(records[-1]), flush=True)
        assert owner.rays.fusion == before[0] and owner.state.depth.orientation.samples_integrated == before[1]
        assert controller.status == 'FAILED_LOCAL_FAILED_UNOBSERVED_TURN_VOLUME'
        verify(launch)
        result = dict(status='RECORDED_RGBD_PHYSICAL_CONFIGURATION_DIAGNOSTIC_COMPLETE',
            exact_replayed_controller_frames=219, observation_ns=now, preparation=preparation,
            preparation_wall_ms=preparation_ms, configurations=records,
            artifact_sha256={n: digest(OUTPUT/n) for n in names},
            controller_or_sensor_state_changed=False, physics_executed=False, future_gait_qualified=False,
            calibrated_error_model=False, navigation_qualified=False, original_mission_success=False)
        write_json(OUTPUT/'result.json', result)
        print(result['status'], flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CONFIGURATION_DIAGNOSTIC_FAILURE', error=repr(error)))
        raise


if __name__ == '__main__': main()

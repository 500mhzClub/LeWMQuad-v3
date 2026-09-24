"""Independent reference replay of every saved configuration, no physics."""
from copy import deepcopy
import json
import math

import cv2
import numpy as np

from lewm.primitive_floor_relation_development import FOOT_SHAPES
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.rgbd_physical_configuration_evidence_development import RGBDPhysicalConfigurationEvidence
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.probe_go2_rgbd_physical_configuration_evidence_development_v1 import (
    OUTPUT, ERRORS, POINT_ERROR_M, YAW_DEGREES, replay_owner, preflight, summary)
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES = {'launch.json': '91df22609d12ed954ca66efa122bb92a48586c2c5d634fbee478d12c752c7c58',
              'result.json': 'd75e8a9f044fa22b5017bb17a5a796143cab30ace052230193cf9735718a85c3'}


def audit():
    cv2.setNumThreads(1)
    own = 'scripts/audit_go2_rgbd_physical_configuration_evidence_development_v1.py'
    sources = {own: digest(ROOT/own)}
    bound = {str((OUTPUT/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bound | sources)
    launch, saved_result = [read_json(OUTPUT, n) for n in ('launch.json', 'result.json')]
    verify(launch); exact(preflight(), launch)
    bound |= {str((OUTPUT/n).relative_to(ROOT)): h for n, h in saved_result['artifact_sha256'].items()}
    assert set(saved_result['artifact_sha256']) == {f'configuration_{i:02d}.json' for i in range(7)}
    verify_bindings(bound)
    controller, geometry, policy, now = replay_owner(launch)
    owner = controller.sensor_memory; before = deepcopy(owner.rays.fusion)
    gyro_count = owner.state.depth.orientation.samples_integrated
    evidence = RGBDPhysicalConfigurationEvidence(owner, geometry, **ERRORS)
    exact(evidence.refresh(now_ns=now), saved_result['preparation'])
    q = policy['sensor_state']['sensed']['joints']['values'][-1, :12]
    rows = []
    for i, degrees in enumerate(YAW_DEGREES):
        saved = read_json(OUTPUT, f'configuration_{i:02d}.json')
        assert saved['yaw_degrees'] == degrees and saved['observation_ns'] == now
        assert saved['full_reference_checked'] == (i == 0)
        R = rotation_increment(owner.rays.rotation.T@owner.rays.up_initial*math.radians(degrees))
        actual = evidence.query([0., 0., 0.], R, q, POINT_ERROR_M, now_ns=now, backend='reference')
        exact(actual, saved['configuration'])
        exact(dict(yaw_degrees=degrees, query_wall_ms=saved['query_wall_ms'], **summary(actual)), saved_result['configurations'][i])
        unseparated_nonfeet = []
        for primitive in actual['primitives']:
            ground = primitive['ground']
            assert not primitive['contact_permitted'] and not ground['contact_permitted']
            if primitive['contact_candidate_with_nonfloor_clearance']:
                assert primitive['shape_id'] in FOOT_SHAPES
            if primitive['shape_id'] not in FOOT_SHAPES and not primitive['conditional_observed_separation']:
                covered = [w for w in primitive['ground_witnesses'] if w['floor_coverage']]
                best = max(covered, key=lambda w: w['gap']['minimum_gap_lower_m']) if covered else None
                unseparated_nonfeet.append(dict(shape_id=primitive['shape_id'], status=ground['status'],
                    best_covered_gap_witness=None if best is None else dict(measured_ns=best['measured_ns'], gap=best['gap'])))
        rows.append(dict(yaw_degrees=degrees, reference_exact=True, summary=summary(actual),
                         unseparated_nonfeet=unseparated_nonfeet))
        print('ALL_PRIMITIVE_REFERENCE_PASS '+str(degrees), flush=True)
    assert owner.rays.fusion == before and owner.state.depth.orientation.samples_integrated == gyro_count
    assert controller.status == 'FAILED_LOCAL_FAILED_UNOBSERVED_TURN_VOLUME'
    verify(launch); exact(preflight(), launch); verify_bindings(bound | sources)
    return dict(status='ALL_SEVEN_CONFIGURATION_REFERENCE_REPLAY_PASS', auditor_source_sha256=sources,
        identity_sha256=IDENTITIES, configurations=rows, sensor_replay_exact_frames=219,
        original_controller_failure_preserved=True, original_global_pose_scales_preserved=True,
        physics_executed=False, navigation_action_permitted=False, future_gait_qualified=False)


if __name__ == '__main__':
    target = OUTPUT/'reference_audit.json'
    if target.exists() or target.is_symlink(): raise ValueError('fresh reference audit only')
    result = audit(); write_json(target, result); print(result['status'], flush=True)

"""Separate exact saved-sensor replay of palette and depth-proposal consumers."""
import json

import cv2

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_proposal_navigation_development import DepthProposalNavigation
from lewm.rgbd_fused_navigation_development import RGBDFusedNavigation
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgb_floor_evidence_development import observe_floor
from lewm.rgbd_shadow_motion_development import priors, POINT_HYPOTHESES
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_rgbd_shadow_motion_development_v1 import OUTPUT as INPUT, PROTOCOL, ARMS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

CASES = (
    ('go2_rgbd_fused_navigation_interface_development_v2_attempt_001', RGBDFusedNavigation,
     '12b77128bf89af2c8190ea0185ae9688f479c765d3122b16f24584b01f88f19e',
     '9abf0a59b7ab184a808680adda2d62fbd139e2aaed31959fdc93ca1d61a69eec'),
    ('go2_depth_proposal_navigation_interface_development_v1_attempt_001', DepthProposalNavigation,
     'd39e01184f1bc3526b0a32859ce2b6e9f2fa7f9b650643b1c47952ed8c324ea3',
     '434700298aab6afc7001d9247d3e708e0a3ff90628c51a3ef26ab0a75adb0e03'))


def exact(actual, saved):
    assert json.loads(json.dumps(actual, allow_nan=False)) == saved


def audit():
    cv2.setNumThreads(1)
    own = 'scripts/audit_go2_fused_navigation_interfaces_development_v1.py'
    source = {own: digest(ROOT/own)}
    report = {}
    for name, cls, launch_hash, result_hash in CASES:
        directory = ROOT/'.generated'/name
        bindings = {str((directory/n).relative_to(ROOT)): h
                    for n, h in [('launch.json', launch_hash), ('result.json', result_hash)]}
        verify_bindings(bindings | source)
        launch = read_json(directory, 'launch.json')
        verify(launch)
        result = read_json(directory, 'result.json')
        bindings |= {str((directory/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
        verify_bindings(bindings)
        counts = {}
        for arm in ARMS:
            prior, _ = priors(launch['source_sha256'][PROTOCOL])
            controller = cls(ArticulatedCollisionGeometry(URDF), memory_arm='local_only',
                             prior=prior, hypotheses=POINT_HYPOTHESES)
            rows = read_json(directory, arm+'.json')
            shadow = read_json(INPUT/arm, 'shadow_observations.json')
            for index, saved in enumerate(rows):
                assert controller.status == 'RUNNING'
                p, d = load_rgbd_observation(INPUT/arm, index)
                f = load_fast_packet(INPUT/arm, index)
                now = p['sensor_state']['decision_ns']
                assert saved['observation_index'] == index and saved['measured_ns'] == now
                floor = observe_floor(p, now_ns=now)
                assert saved['positive_floor_pixels'] == int(floor['floor_evidence_mask'].sum()) == 0
                assert saved['floor_supported_columns'] == int(floor['valid_columns'].sum()) == 0
                decision = failure = None
                try:
                    decision = controller.observe_rgbd(p, f, d, now_ns=now)
                except SensorContractError as error:
                    failure = []
                    while error is not None:
                        failure.append(str(error))
                        error = error.__cause__
                exact(decision, saved['decision'])
                exact(failure, saved['failure'])
                assert saved['commands_executed'] is False
                if decision is not None:
                    expected = shadow[index]['shadow']['state']
                    exact(decision['sensor_fusion'], expected['fusion'])
                    exact(decision['raw_depth_motion'], expected['depth_state']['motion'])
                    assert saved['fusion_exactly_matches_frozen_shadow'] is True
                else:
                    assert shadow[index]['shadow']['status'] == 'TERMINAL_SHADOW_FAILURE'
                    assert failure[-1] == 'conditional pose-error budget exhausted; stop'
                assert controller._context is None and controller.regions.memory.pending is None
            stats = result['statistics'][arm]
            assert stats['frames_consumed'] == len(rows)
            assert stats['controller_status'] == controller.status
            assert stats['last_stage'] == controller.stage
            exact(stats['failure'], rows[-1]['failure'])
            assert stats['proposed_nonzero_commands'] == sum(
                row['decision'] is not None and any(row['decision']['requested_command']) for row in rows)
            assert not stats['commands_executed'] and not stats['closed_loop_navigation_evaluated']
            if cls is RGBDFusedNavigation:
                assert len(rows) == 4 and controller.status == 'FAILED_INITIAL_NO_EXIT'
            elif arm == 'neutral':
                assert len(rows) == 29 and controller.status == 'FAILED_SENSOR'
            else:
                assert len(rows) == 54 and controller.status == 'RUNNING' and controller.stage == 'TRAVERSE'
            counts[arm] = len(rows)
        verify(launch)
        verify_bindings(bindings | source)
        report[name] = dict(exact_replayed_frames=counts, launch_sha256=launch_hash,
                           result_sha256=result_hash, artifacts_verified=len(result['artifact_sha256']))
        print('EXACT_INTERFACE_REPLAY_PASS '+name, flush=True)
    return dict(status='SAVED_SENSOR_NAVIGATION_INTERFACES_EXACT_REPLAY_PASS', cases=report,
                auditor_source_sha256=source, commands_executed=False,
                closed_loop_navigation_evaluated=False, navigation_qualified=False)


if __name__ == '__main__':
    target = ROOT/'.generated'/CASES[-1][0]/'interface_audit.json'
    if target.exists():
        raise ValueError('fresh independent interface audit only')
    result = audit()
    write_json(target, result)
    print(json.dumps(result), flush=True)

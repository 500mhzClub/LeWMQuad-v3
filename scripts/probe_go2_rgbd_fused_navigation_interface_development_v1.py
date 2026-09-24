"""Frozen recorded-sensor consumer diagnostic; never execute proposed commands."""
import json
import cv2

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgb_floor_evidence_development import observe_floor
from lewm.rgbd_fused_navigation_development import RGBDFusedNavigation
from lewm.rgbd_shadow_motion_development import priors, POINT_HYPOTHESES
from scripts.analyze_go2_ground_plane_development_v1 import URDF, verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_rgbd_shadow_motion_development_v1 import OUTPUT as INPUT, PROTOCOL as INPUT_PROTOCOL, ARMS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = ROOT/'.generated/go2_rgbd_fused_navigation_interface_development_v1_attempt_001'
PROTOCOL = 'docs/go2_rgbd_fused_navigation_interface_development_v1_2026-09-06.md'
SEEDS = (PROTOCOL, 'scripts/probe_go2_rgbd_fused_navigation_interface_development_v1.py',
         'lewm/tests/test_rgbd_fused_navigation_development.py')
IDENTITIES = {
    'launch.json': '628168df54a4e8d96575f939609057f610de35d60fd579cd58d76ca58dd964dc',
    'result.json': '4a5b97bb50f28ca631bdb1bb1e3ee953658105613742333c638429b0c482bfd1',
    'raw_artifact_audit.json': 'dffa2ff61d64d32d54de92b9d445b6cc21e501649d368b657987a0459bfad42f'}


def preflight():
    bindings = {str((INPUT/n).relative_to(ROOT)): h for n, h in IDENTITIES.items()}
    verify_bindings(bindings)
    old = read_json(INPUT, 'launch.json')
    verify(old)
    result = read_json(INPUT, 'result.json')
    bindings |= {str((INPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    verify_bindings(bindings)
    sources = discover_sources(SEEDS, old['source_sha256'])
    launch = old | dict(source_sha256=sources, input_sha256=old['input_sha256'] | bindings,
        diagnostic_protocol=PROTOCOL, commands_executed=False,
        scope='already-seen sensor/controller interface diagnostic; no closed-loop outcome')
    verify(launch)
    return launch


def run_arm(arm, launch):
    directory = INPUT/arm
    prior, _ = priors(launch['source_sha256'][INPUT_PROTOCOL])
    controller = RGBDFusedNavigation(ArticulatedCollisionGeometry(URDF), memory_arm='local_only',
                                    prior=prior, hypotheses=POINT_HYPOTHESES)
    saved = read_json(directory, 'shadow_observations.json')
    rows = []
    for index, shadow in enumerate(saved):
        policy, depth = load_rgbd_observation(directory, index)
        fast = load_fast_packet(directory, index)
        now = policy['sensor_state']['decision_ns']
        floor = observe_floor(policy, now_ns=now)
        item = dict(observation_index=index, measured_ns=now,
                    positive_floor_pixels=int(floor['floor_evidence_mask'].sum()),
                    floor_supported_columns=int(floor['valid_columns'].sum()),
                    decision=None, failure=None, commands_executed=False)
        try:
            decision = controller.observe_rgbd(policy, fast, depth, now_ns=now)
            expected = shadow['shadow']['state']
            assert expected is not None
            assert decision['sensor_fusion'] == expected['fusion']
            assert decision['raw_depth_motion'] == expected['depth_state']['motion']
            item.update(decision=decision, fusion_exactly_matches_frozen_shadow=True)
        except SensorContractError as error:
            reasons = []
            while error is not None:
                reasons.append(str(error))
                error = error.__cause__
            item['failure'] = reasons
        rows.append(item)
        if item['failure'] is not None or item['decision']['terminal']:
            break
    write_json(OUTPUT/(arm+'.json'), rows)
    return dict(frames_consumed=len(rows), controller_status=controller.status,
                last_stage=controller.stage, failure=rows[-1]['failure'],
                terminal_decision=None if rows[-1]['decision'] is None else rows[-1]['decision']['status'],
                positive_floor_pixels_first=rows[0]['positive_floor_pixels'],
                positive_floor_pixels_last=rows[-1]['positive_floor_pixels'],
                proposed_nonzero_commands=sum(r['decision'] is not None and
                    any(v != 0 for v in r['decision']['requested_command']) for r in rows),
                commands_executed=False, closed_loop_navigation_evaluated=False)


def main():
    cv2.setNumThreads(1)
    launch = preflight()
    OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', launch)
    try:
        statistics = {arm: run_arm(arm, launch) for arm in ARMS}
        verify(launch)
        result = dict(status='RECORDED_FULL_TASK_CONSUMER_INTERFACE_DIAGNOSTIC_COMPLETE',
            statistics=statistics, commands_executed=False, navigation_qualified=False,
            artifact_sha256={arm+'.json': digest(OUTPUT/(arm+'.json')) for arm in ARMS})
        write_json(OUTPUT/'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()

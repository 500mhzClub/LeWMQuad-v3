"""Bounded actual-capture integration of the distinct auxiliary RGB contract."""
import argparse
import json
import time
import cv2
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from lewm.causal_auxiliary_rgb_observation_development import depth_digest, validate_rgb
from scripts.diagnose_go2_auxiliary_return_pairs_v2 import (
    OUTPUT as DIAGNOSIS, INPUT, CASE, FRAMES, verify_all as verify_diagnosis)
from scripts.navigation_artifact_root_development import (
    BASE, create_output, validate_root, verify_artifacts)
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

OUTPUT = BASE/'go2_auxiliary_rgb_packet_audit_v1_attempt_001'
PROTOCOL = 'docs/go2_auxiliary_rgb_packet_audit_v1_2026-09-09.md'
RESULT_SHA = '4980170b5c02e9ab8de55f331583f7b0070decdc53aa753dbe9466821cf855da'


def verify_all(launch):
    source_check(launch['source_sha256'])
    verify_artifacts(DIAGNOSIS, launch['diagnosis_artifact_sha256'])
    verify_diagnosis(read_json(DIAGNOSIS, 'launch.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive packet audit required')
    cv2.setNumThreads(1)
    verify_artifacts(DIAGNOSIS, {'result.json': RESULT_SHA})
    diagnosis = read_json(DIAGNOSIS, 'result.json')
    if diagnosis['status'] != 'AUXILIARY_RETURN_PAIR_DIAGNOSIS_V2_COMPLETE':
        raise ValueError('completed pair diagnosis required')
    sources = discover_sources((PROTOCOL, 'scripts/audit_go2_auxiliary_rgb_packet_v1.py',
        'lewm/tests/test_causal_auxiliary_rgb_observation_development.py'), diagnosis['source_sha256'])
    resources = hardware()
    launch = dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        diagnosis_artifact_sha256={'result.json': RESULT_SHA, **diagnosis['artifact_sha256']},
        hardware=resources, frames=list(FRAMES), cpu_processes=1, numerical_threads=1,
        minimum_available_ram_bytes=2*1024**3, output_allowance_bytes=64*1024**2,
        os_resource_limits_enforced=False, native_scene_workers=0,
        concurrency_reason='small packet audit beside existing native scene and independent replay',
        native_execution=False, model_training=False)
    verify_all(launch)
    memory_ok = resources['memory_available_bytes'] >= 2*1024**3
    storage_ok = resources['artifact_free_bytes'] >= 40*1024**3+64*1024**2
    if args.preflight_only:
        print('AUXILIARY_RGB_PACKET_PREFLIGHT', json.dumps(dict(source_count=len(sources),
            hardware=resources, memory_admission_pass=memory_ok, storage_admission_pass=storage_ok,
            output_created=False)), flush=True)
        return
    if not memory_ok or not storage_ok: raise ValueError('packet audit resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', launch)
    print('AUXILIARY_RGB_PACKET_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        directory = INPUT/CASE
        reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
        rows = []
        for i in FRAMES:
            policy, _, _, now = reader.packet(i)
            image, depth = packet(directory, i, policy, public_acquisition(acquisitions[i]), now_ns=now)
            validate_rgb(image, depth, policy, now_ns=now)
            expected = diagnosis['feature_witnesses'][str(i)]
            features = CornerSupportFeatureFrame(image['rgb'], depth).witness()
            assert features == expected['auxiliary']
            assert image['rgb_sha256'] == expected['auxiliary_rgb_sha256']
            assert depth_digest(depth) == expected['auxiliary_depth_sha256']
            assert image['measured_ns'] == expected['measured_ns']
            rows.append(dict(frame=i, measured_ns=now, rgb_sha256=image['rgb_sha256'],
                auxiliary_depth_sha256=depth_digest(depth), feature_witness=features))
        assert len(rows) == 18
        verify_all(launch)
        write_json(OUTPUT/'result.json', dict(status='AUXILIARY_RGB_PACKET_AUDIT_COMPLETE',
            source_sha256=sources, artifact_sha256={'launch.json': digest(OUTPUT/'launch.json')},
            rows=rows, frames=len(rows), all_pixels_depth_and_feature_witnesses_exact=True,
            public_packet_contract_validated=True, native_pose_or_segmentation_used=False,
            ideal_simulated_zero_latency=True, hardware_calibrated=False,
            controller_installed=False, continuous_pose_evaluated=False,
            native_execution=False, navigation_qualified=False, goal_achieved=False,
            wall_s=time.perf_counter()-started, hardware_after=hardware()))
        print('AUXILIARY_RGB_PACKET_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__':
    main()

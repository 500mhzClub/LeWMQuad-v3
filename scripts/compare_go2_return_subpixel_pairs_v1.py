"""One fixed measured-feature candidate on every diagnosed transition pair."""
import json
import cv2
import numpy as np
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.subpixel_corner_support_features_development import SubpixelCornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.return_transition_match_diagnosis_development import diagnose_pair
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register
from lewm.causal_sensor_state import SensorContractError
from scripts.navigation_artifact_root_development import BASE, create_output, artifact_path, verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.diagnose_go2_return_transition_matches_v1 import INPUT, CASE, FRAMES, OUTPUT as DIAGNOSIS

OUTPUT = BASE / 'go2_return_subpixel_pair_comparison_v1_attempt_001'
RESULT = '29565cd5f2d87ba6c9d657780fc9ca75f908ab7cd28d0713f4da28d2b758e081'


def main():
    if not __debug__:
        raise ValueError('assertions required')
    cv2.setNumThreads(1)
    verify_artifacts(DIAGNOSIS, {'result.json': RESULT})
    result = json.loads(artifact_path(DIAGNOSIS, 'result.json').read_text())
    ids = {'result.json': RESULT, **result['artifact_sha256']}
    verify_artifacts(DIAGNOSIS, ids)
    predecessor = json.loads(artifact_path(DIAGNOSIS, 'launch.json').read_text())
    sources = discover_sources(('scripts/compare_go2_return_subpixel_pairs_v1.py',
        'docs/go2_return_subpixel_pair_comparison_v1_2026-09-09.md',
        'lewm/tests/test_subpixel_corner_support_features_development.py'), result['source_sha256'])
    source_check(sources); verify_artifacts(INPUT, predecessor['input_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 2*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+64*1024**2:
        raise ValueError('bounded comparison resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT / 'launch.json', dict(diagnosis_artifact_sha256=ids, source_sha256=sources,
        input_sha256=predecessor['input_sha256'], hardware=resources, cpu_processes=1,
        numerical_threads=1, native_scene_workers=0, model_training=False,
        minimum_available_ram_bytes=2*1024**3, output_allowance_bytes=64*1024**2,
        os_resource_limits_enforced=False, candidate_count=1, parameter_search=False,
        concurrency_reason='bounded public pair comparison beside the existing full native audit'))
    print('RETURN_SUBPIXEL_PAIR_COMPARISON_LAUNCHED', digest(OUTPUT / 'launch.json'), flush=True)
    try:
        reader = IntentReturnRGBDReplay(INPUT / CASE); gyro = FastRelativeOrientation()
        original, candidate, rotations, witnesses = {}, {}, {}, {}
        for i in FRAMES:
            p, d, f, now = reader.packet(i)
            attitude = (gyro.begin(p, f, now_ns=now) if i == FRAMES[0] else gyro.step(p, f, now_ns=now))
            rotations[i] = np.asarray(attitude['rotation_initial_body_from_current_body'])
            original[i] = CornerSupportFeatureFrame(p['image']['rgb'], d)
            candidate[i] = SubpixelCornerSupportFeatureFrame(p['image']['rgb'], d)
            assert original[i].witness() == {k:result['feature_witnesses'][str(i)][k] for k in original[i].witness()}
            witnesses[i] = candidate[i].witness()
        pairs = []
        for saved in result['pairs']:
            a, b = saved['reference'], saved['current']; row = dict(reference=a, current=b, role=saved['role'])
            relative = rotations[a].T @ rotations[b]
            for variant, frames in (('original', original), ('subpixel', candidate)):
                counts = diagnose_pair(frames[a], frames[b])
                if variant == 'original':
                    assert counts == {k:saved[k] for k in counts}
                try:
                    R, t, mask, fit = register(*matched_points(frames[a], frames[b]),
                        gyro_rotation=relative, mode='joint', frame=b)
                    fit = dict(status='PAIR_RIGID_FIT_ACCEPTED', registration=fit,
                        rotation_reference_body_from_current_body=R.tolist(),
                        translation_reference_body_m=t.tolist())
                except SensorContractError as error:
                    fit = dict(status='PAIR_RIGID_FIT_REJECTED', reason=str(error))
                row[variant] = counts | fit
            pairs.append(row)
        verify_artifacts(DIAGNOSIS, ids); verify_artifacts(INPUT, predecessor['input_sha256']); source_check(sources)
        write_json(OUTPUT / 'result.json', dict(status='RETURN_SUBPIXEL_PAIR_COMPARISON_COMPLETE',
            source_sha256=sources, artifact_sha256={'launch.json': digest(OUTPUT / 'launch.json')},
            candidate_feature_witnesses=witnesses, pairs=pairs, original_stage_counts_exact=True,
            input_bytes_unchanged=True, downstream_acceptance_thresholds_unchanged=True,
            online_continuity_evaluated=False, candidate_installed=False, navigation_qualified=False))
        print('RETURN_SUBPIXEL_PAIR_COMPARISON_COMPLETE', digest(OUTPUT / 'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__':
    main()

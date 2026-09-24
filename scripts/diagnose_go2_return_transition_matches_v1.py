"""Bounded public-image diagnosis of the ninth pilot's return tracking failure."""
import json
import cv2
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.return_transition_match_diagnosis_development import diagnose_pair
from scripts.navigation_artifact_root_development import BASE, create_output, verify_artifacts, artifact_path
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

INPUT = BASE / 'go2_later_floor_resolution_maze_pilot_v1_attempt_001'
OUTPUT = BASE / 'go2_return_transition_match_diagnosis_v1_attempt_001'
CASE = 'full_jepa_novel_maze_00'
FRAMES = tuple(range(1853, 1871))
REFERENCES = (1866, 1861, 1860, 1859, 1857, 1856, 1854, 1853)
IDENTITIES = {'launch.json': 'c3d035abcc69b3b42ecb160021203e7d6d685a176e860e5c3044afa2035cefa4',
    CASE + '/result.json': '69532d7591532ba5400e05a6cf596766df567a660392d973f4ef22da52845a15'}


def main():
    if not __debug__:
        raise ValueError('assertions required')
    cv2.setNumThreads(1)
    verify_artifacts(INPUT, IDENTITIES)
    native = json.loads(artifact_path(INPUT, 'launch.json').read_text())
    sources = discover_sources(('scripts/diagnose_go2_return_transition_matches_v1.py',
        'docs/go2_return_transition_match_diagnosis_v1_2026-09-09.md'), native['source_sha256'])
    source_check(sources)
    names = ['policy_observations.json', 'policy_histories.npz', 'depth_observations.json', 'fast_gyro_histories.npz']
    names += [name for i in FRAMES for name in (f'rgb_{i:04d}.png', f'depth_{i:04d}.npz')]
    bindings = IDENTITIES | {CASE + '/' + n: digest(artifact_path(INPUT, CASE + '/' + n)) for n in names}
    resources = hardware()
    if resources['memory_available_bytes'] < 2 * 1024**3 or resources['artifact_free_bytes'] < 40 * 1024**3 + 64 * 1024**2:
        raise ValueError('diagnosis resource admission failed')
    create_output(OUTPUT)
    write_json(OUTPUT / 'launch.json', dict(input_root=str(INPUT), input_sha256=bindings,
        source_sha256=sources, hardware=resources, frames=list(FRAMES), retained_references=list(REFERENCES),
        cpu_processes=1, numerical_threads=1, native_scene_workers=0, model_training=False,
        concurrency_reason='bounded 18-frame public replay beside existing native audit',
        minimum_available_ram_bytes=2*1024**3, output_allowance_bytes=64*1024**2,
        os_resource_limits_enforced=False))
    print('RETURN_TRANSITION_MATCH_DIAGNOSIS_LAUNCHED', digest(OUTPUT / 'launch.json'), flush=True)
    try:
        reader = IntentReturnRGBDReplay(INPUT / CASE)
        frames, witnesses = {}, {}
        for i in FRAMES:
            p, d, _, now = reader.packet(i)
            frame = CornerSupportFeatureFrame(p['image']['rgb'], d)
            frames[i] = frame
            witnesses[i] = frame.witness() | dict(measured_ns=now, rgb_sha256=d['rgb_sha256'])
        assert witnesses[1869]['rgb_sha256'] == '921c4ddde6b503f8437b007cea4e05d7d8ee70723e46e47dc1b9bb589a73b057'
        assert witnesses[1869]['selected_features'] == 29
        pairs = [dict(reference=i-1, current=i, role='consecutive', **diagnose_pair(frames[i-1], frames[i]))
                 for i in FRAMES[1:]]
        pairs += [dict(reference=i, current=1870, role='retained', **diagnose_pair(frames[i], frames[1870]))
                  for i in REFERENCES]
        assert all(not p['minimum_match_count_pass'] for p in pairs if p['current'] == 1870)
        verify_artifacts(INPUT, bindings); source_check(sources)
        write_json(OUTPUT / 'result.json', dict(status='RETURN_TRANSITION_MATCH_DIAGNOSIS_COMPLETE',
            artifact_sha256={'launch.json': digest(OUTPUT / 'launch.json')}, source_sha256=sources,
            feature_witnesses=witnesses, pairs=pairs, input_bytes_unchanged=True,
            terminal_insufficient_matches_reproduced=True, thresholds_unchanged=True,
            native_audit_pending_at_launch=True, pose_acceptance_claim=False, navigation_qualified=False))
        print('RETURN_TRANSITION_MATCH_DIAGNOSIS_COMPLETE', digest(OUTPUT / 'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT / 'failure.json', dict(reason=repr(error)))
        raise


if __name__ == '__main__':
    main()

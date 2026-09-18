"""Bounded post-hoc association probe on all retained references at frame863.

The unchanged direct-flow matcher is evaluated outside its 100ms runtime scope.
Pair counts are not pose admission, a new controller, or a successful recovery.
"""
import hashlib
import json
import time
from pathlib import Path

import cv2

from scripts import diagnose_go2_direct_flow_bridge_exhaustion_v1 as diagnosis
from scripts.startup_source_inventory_development import discover_sources
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.causal_rgb_dataset_development import _leaf
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay

SOURCE = 'scripts/probe_go2_direct_flow_retained_anchor_pairs_v1.py'
OUTPUT = Path('docs/go2_direct_flow_retained_anchor_pair_probe_2026-09-11.json')
DIAGNOSIS_SHA = 'a3d654262ee2f54e9c8a371bbf03b0487545b872762421e834d3d83228bf54d1'
REFERENCES = (850, 849, 848, 847, 846, 845, 844, 840)
CURRENT = 863


def main():
    started = time.monotonic()
    require, digest = diagnosis.require, diagnosis.digest
    require(not OUTPUT.exists(), 'exclusive probe output required')
    require(digest(diagnosis.OUTPUT) == DIAGNOSIS_SHA, 'fixed completed diagnosis required')
    prior = json.loads(diagnosis.OUTPUT.read_text())
    sources = discover_sources((SOURCE,), prior['source_sha256'])
    diagnosis.verify_sources(sources)
    directory = diagnosis.ROOT/diagnosis.CASE
    # Only explicit public packet inputs. Capture immutable byte identities before
    # construction; the separately running native audit is not claimed complete.
    names = ['policy_observations.json', 'policy_histories.npz', 'depth_observations.json',
             'fast_gyro_histories.npz', 'auxiliary_camera_audit.json']
    for frame in (*REFERENCES, CURRENT):
        names.extend((f'rgb_{frame:04d}.png', f'depth_{frame:04d}.npz',
                      f'auxiliary_rgb_{frame:04d}.png', f'auxiliary_depth_{frame:04d}.npz'))
    artifacts = {name: digest(_leaf(directory, name)) for name in names}
    resources = hardware()
    require(resources['memory_available_bytes'] >= 8*1024**3, '8GiB available RAM required')
    require(resources['workspace_free_bytes'] >= 64*1024**2, '64MiB report headroom required')
    cv2.setNumThreads(1)
    require(not cv2.ocl.useOpenCL(), 'fixed CPU OpenCV required')
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = json.loads(_leaf(directory, 'auxiliary_camera_audit.json').read_text())
    require(len(reader.frames) == len(acquisitions) == 874, 'complete fixed collection population required')
    features = {}
    for frame in (*REFERENCES, CURRENT):
        policy, depth, _, now = reader.packet(frame)
        require(now == 1_500_000_000 + frame*100_000_000, 'original packet clock required')
        image, auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
            auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
    pairs = []
    for camera in ('primary', 'auxiliary'):
        expected = prior['camera_measurements'][camera]['anchor_attempts']
        require(tuple(a['reference_frame'] for a in expected) == REFERENCES, 'complete original reference population')
        for frame, old in zip(REFERENCES, expected, strict=True):
            reference, current = features[frame][camera], features[CURRENT][camera]
            original = matched_points(reference, current)
            require(old['reason'] == 'insufficient rigid-pose matches' and len(original[0]) < 12,
                    'original insufficient-match rejection must reconstruct')
            values, receipt = tracked_points(reference, current)
            repeated, check = tracked_points(reference, current)
            require(receipt == check, 'repeat association receipt must match')
            arrays = {}
            for name, left, right in zip(('reference_points', 'current_points', 'reference_pixels', 'current_pixels'),
                                         values, repeated, strict=True):
                require(left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes(),
                        'repeat association arrays must be byte-exact')
                arrays[name] = dict(shape=list(left.shape), dtype=str(left.dtype),
                    sha256=hashlib.sha256(left.tobytes()).hexdigest())
            pairs.append(dict(camera=camera, reference_frame=frame, current_frame=CURRENT,
                interval_ns=(CURRENT-frame)*100_000_000, original_valid_depth_pairs=len(original[0]),
                original_failure=old['reason'], direct_flow=receipt, arrays=arrays,
                repeated_association_arrays_byte_exact=True))
    require(len(pairs) == 16, 'all eight anchors in both cameras required')
    diagnosis.verify_sources(sources)
    for name, expected in artifacts.items():
        require(digest(_leaf(directory, name)) == expected, 'input changed during probe: '+name)
    require(digest(diagnosis.OUTPUT) == DIAGNOSIS_SHA, 'diagnosis changed during probe')
    report = dict(status='DIRECT_FLOW_RETAINED_ANCHOR_ASSOCIATION_PROBE_COMPLETE',
        diagnosis_sha256=DIAGNOSIS_SHA, source_sha256=sources, input_root=str(directory),
        input_sha256=artifacts, hardware=resources, maximum_workers=1, opencv_threads=1,
        feature_witnesses={str(f): {c: v.witness() for c, v in views.items()} for f, views in features.items()},
        pairs=pairs, elapsed_seconds=time.monotonic()-started,
        scope=dict(posthoc_pair_probe=True, longer_interval_association_only=True,
            current_runtime_fallback_supports_these_anchor_intervals=False,
            original_match_failures_reconstructed=True, full_observer_history_replayed=False,
            rigid_registration_evaluated=False, gyro_gate_evaluated=False, pose_admitted=False,
            controller_changed=False, model_loaded=False, native_execution=False,
            native_audit_verified=False, navigation_recovered=False, goal_achieved=False))
    with OUTPUT.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), sha256=digest(OUTPUT), elapsed_seconds=report['elapsed_seconds'],
        pairs=[dict(camera=p['camera'], reference=p['reference_frame'], original=p['original_valid_depth_pairs'],
                    direct=p['direct_flow']['counts']['valid_depth_pair']) for p in pairs])), flush=True)


if __name__ == '__main__':
    main()

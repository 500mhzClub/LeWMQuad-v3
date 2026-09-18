"""Fixed 16-pair image-chain/endpoint-fit probe; no full observer or native run."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from scripts import probe_go2_direct_flow_retained_anchor_pairs_v1 as prior_probe
from scripts.startup_source_inventory_development import discover_sources
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from lewm.causal_rgb_dataset_development import _leaf
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.chained_corner_flow_association_development import chained_points
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.joint_rgbd_rigid_pose_development import register
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference, pose_in_body

SOURCE = 'scripts/probe_go2_chained_retained_anchor_pairs_v1.py'
TEST = 'lewm/tests/test_chained_corner_flow_association_development.py'
OUTPUT = Path('docs/go2_chained_retained_anchor_pair_probe_2026-09-11.json')
PRIOR_SHA = '0f552d0a06450cddb772f7dfcb06a0eab90a78745751c914f9230cc5165b643d'


def main():
    started = time.monotonic()
    require, digest = prior_probe.diagnosis.require, prior_probe.diagnosis.digest
    require(not OUTPUT.exists(), 'exclusive chained-pair output required')
    require(digest(prior_probe.OUTPUT) == PRIOR_SHA, 'fixed prior all-anchor probe required')
    prior = json.loads(prior_probe.OUTPUT.read_text())
    sources = discover_sources((SOURCE, TEST), prior['source_sha256'])
    prior_probe.diagnosis.verify_sources(sources)
    directory = prior_probe.diagnosis.ROOT/prior_probe.diagnosis.CASE
    frames = tuple(range(min(prior_probe.REFERENCES), prior_probe.CURRENT+1))
    names = set(prior['input_sha256'])
    for frame in frames:
        names.update((f'rgb_{frame:04d}.png', f'depth_{frame:04d}.npz',
                      f'auxiliary_rgb_{frame:04d}.png', f'auxiliary_depth_{frame:04d}.npz'))
    artifacts = {name: digest(_leaf(directory, name)) for name in sorted(names)}
    require(all(artifacts[k] == v for k, v in prior['input_sha256'].items()), 'prior raw input identities required')
    resources = hardware()
    require(resources['memory_available_bytes'] >= 8*1024**3 and
            resources['workspace_free_bytes'] >= 64*1024**2, 'bounded CPU probe resources required')
    cv2.setNumThreads(1)
    require(not cv2.ocl.useOpenCL(), 'fixed CPU OpenCV required')
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = json.loads(_leaf(directory, 'auxiliary_camera_audit.json').read_text())
    require(len(reader.frames) == len(acquisitions) == 874, 'complete closed collection required')
    features, gyros = {}, {}
    orientation = FastRelativeOrientation()
    for frame in frames:
        policy, depth, fast, now = reader.packet(frame)
        require(now == 1_500_000_000+frame*100_000_000, 'fixed original measured clock required')
        state = (orientation.begin(policy, fast, now_ns=now) if frame == frames[0]
                 else orientation.step(policy, fast, now_ns=now))
        gyros[frame] = np.asarray(state['rotation_initial_body_from_current_body'])
        image, auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
            auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
        if str(frame) in prior['feature_witnesses']:
            require({c: f.witness() for c, f in features[frame].items()} == prior['feature_witnesses'][str(frame)],
                    'original retained feature populations must reconstruct')
    pairs = []
    for camera in ('primary', 'auxiliary'):
        for reference in prior_probe.REFERENCES:
            sequence = [(f, 1_500_000_000+f*100_000_000, features[f][camera]) for f in frames if f >= reference]
            values, receipt = chained_points(sequence)
            repeated, check = chained_points(sequence)
            require(receipt == check, 'repeat chain receipt required')
            arrays = {}
            for name, a, b in zip(('reference_points', 'current_points', 'reference_pixels', 'current_pixels'),
                                   values, repeated, strict=True):
                require(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                        'repeat endpoint arrays must be byte-exact')
                arrays[name] = dict(shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest())
            relative_gyro = gyros[reference].T@gyros[prior_probe.CURRENT]
            G = relative_gyro if camera == 'primary' else gyro_in_reference(relative_gyro)
            try:
                R, t, mask, evidence = register(*values, gyro_rotation=G, mode='joint', frame=prior_probe.CURRENT)
                body_R, body_t = (R, t) if camera == 'primary' else pose_in_body(R, t)
                registration = dict(qualified=True, failure=None, evidence=evidence,
                    reference_body_from_current_body=body_R.tolist(), translation_reference_body_m=body_t.tolist(),
                    inlier_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
            except SensorContractError as error:
                registration = dict(qualified=False, failure=str(error))
            pairs.append(dict(camera=camera, reference_frame=reference, current_frame=prior_probe.CURRENT,
                association=receipt, endpoint_arrays=arrays, registration=registration,
                repeated_association_byte_exact=True))
    require(len(pairs) == 16, 'all original retained pairs required')
    prior_probe.diagnosis.verify_sources(sources)
    for name, expected in artifacts.items():
        require(digest(_leaf(directory, name)) == expected, 'raw input changed: '+name)
    require(digest(prior_probe.OUTPUT) == PRIOR_SHA, 'prior probe changed')
    report = dict(status='CHAINED_RETAINED_ANCHOR_ENDPOINT_PROBE_COMPLETE', source_sha256=sources,
        predecessor_sha256=PRIOR_SHA, input_root=str(directory), input_sha256=artifacts,
        hardware=resources, maximum_workers=1, opencv_threads=1, pairs=pairs,
        gyro=dict(start_frame=frames[0], end_frame=frames[-1], public_packets_validated=len(frames),
            samples_integrated=orientation.samples_integrated, role='relative_rotation_consistency_monitor'),
        elapsed_seconds=time.monotonic()-started,
        scope=dict(posthoc_pair_probe=True, original_endpoint_rigid_and_gyro_gates_applied=True,
            original_anchor_coordinates_retained=True, pose_increments_composed=False,
            full_observer_history_replayed=False, prior_pose_envelopes_checked=False,
            anchor_increment_conflict_checked=False, pose_admitted=False, model_loaded=False,
            controller_changed=False, native_execution=False, native_audit_verified=False,
            bridge_allowance_changed=False, navigation_recovered=False, goal_achieved=False))
    with OUTPUT.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUTPUT), sha256=digest(OUTPUT), elapsed_seconds=report['elapsed_seconds'],
        pairs=[dict(camera=p['camera'], reference=p['reference_frame'],
                    endpoints=p['association']['endpoint_depth_pairs'], registration=p['registration']) for p in pairs])), flush=True)


if __name__ == '__main__':
    main()

"""Test the unchanged direct-flow association on the actual failed contact pair."""
import hashlib
import json

import cv2

from scripts import diagnose_go2_contact_anchored_worker_tracking_failure_v1 as diagnosis
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay

SOURCE = 'scripts/probe_go2_contact_anchored_direct_flow_pair_v1.py'
DIAGNOSIS_SHA = '9f28b43e90a9e4076b61ced5bac7daa43476a2b18d3e6acec8db8ec926244cea'
OUTPUT = ROOT/'docs/go2_contact_anchored_direct_flow_pair_probe_2026-09-11.json'


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive fixed-pair probe required')
    verify({str(diagnosis.OUTPUT.relative_to(ROOT)):DIAGNOSIS_SHA})
    old = json.loads(diagnosis.OUTPUT.read_text())
    if old['status'] != 'CONTACT_ANCHORED_WORKER_RAW_TRACKING_FAILURE_RECONSTRUCTED':
        raise ValueError('complete original match-stage diagnosis required')
    sources = discover_sources((SOURCE,), old['source_sha256']); verify(sources)
    root = diagnosis.native.OUTPUT; directory = root/diagnosis.native.CASE[0]
    verify_artifacts(root, old['artifact_sha256']); diagnosis.worker_ended()
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); features = {}
    cv2.setNumThreads(1)
    if cv2.ocl.useOpenCL(): raise ValueError('fixed CPU OpenCV required')
    for frame in (560, 561):
        policy, depth, _, now = reader.packet(frame)
        if now != 1_500_000_000+frame*100_000_000: raise ValueError('original current packet clock required')
        image, auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
            auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
        if {c:f.witness() for c,f in features[frame].items()} != old['feature_witnesses'][str(frame)]:
            raise ValueError('both original measured feature witnesses must reconstruct')
    pairs = []
    for camera in ('primary', 'auxiliary'):
        a, b = features[560][camera], features[561][camera]
        original = matched_points(a, b)
        prior = next(r for r in old['raw_match_pairs'] if r['camera'] == camera and r['reference_role'] == 'increment')
        if len(original[0]) != prior['counts']['valid_depth_pair']:
            raise ValueError('same failed original correspondence population required')
        values, receipt = tracked_points(a, b); repeated, check = tracked_points(a, b)
        if receipt != check: raise ValueError('repeat association receipt must be exact')
        arrays = {}
        for name, x, y in zip(('reference_points', 'current_points', 'reference_pixels', 'current_pixels'), values, repeated, strict=True):
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                raise ValueError('repeat association arrays must be byte-exact')
            arrays[name] = dict(shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(x.tobytes()).hexdigest())
        if len(values[0]) != receipt['counts']['valid_depth_pair']:
            raise ValueError('receipt must count every actual lifted pair')
        pairs.append(dict(camera=camera, original_valid_depth_pairs=len(original[0]),
            original_failure=prior['registration']['original_failure'], direct_flow=receipt,
            repeated_association_arrays_byte_exact=True, arrays=arrays))
    diagnosis.worker_ended(); verify(sources); verify_artifacts(root, old['artifact_sha256'])
    verify({str(diagnosis.OUTPUT.relative_to(ROOT)):DIAGNOSIS_SHA})
    write_json(OUTPUT, dict(status='CONTACT_ANCHORED_DIRECT_FLOW_PAIR_PROBE_COMPLETE',
        source_sha256=sources, source_count=len(sources), diagnosis_sha256=DIAGNOSIS_SHA,
        worker_terminal_sha256=diagnosis.WORKER_SHA, reference_frame=560, current_frame=561,
        pairs=pairs, original_failures_preserved=True, existing_fallback_source_unchanged=True,
        full_observer_history_replayed=False, rigid_registration_evaluated=False,
        gyro_gate_evaluated=False, pose_admitted=False, command_selected=False,
        new_model_inference=False, native_execution=False, policy_selected=False, goal_achieved=False))
    print('CONTACT_DIRECT_FLOW_PAIR_PROBE', digest(OUTPUT), len(sources), flush=True)
    for p in pairs: print(p['camera'], 'original', p['original_valid_depth_pairs'], 'flow', p['direct_flow']['counts'], flush=True)


if __name__ == '__main__': main()

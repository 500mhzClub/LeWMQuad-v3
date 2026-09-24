"""Reproduce a live mapping trial's sensor-estimation failure from saved packets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import torch

from lewm.eligible_floor_registration_development import bind
from lewm.local_feature_depth_consensus_development import (
    LocalFeatureDepthConsensusMotion, LocalFeatureDepthRegistration)
from lewm.local_inverse_depth_floor_development import measured_candidates
from lewm.partial_floor_height_development import read_pose
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE
from scripts.diagnose_go2_depth_noise_failures_development import save, registration_diagnostic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    i = parser.parse_args().layout_index
    root = BASE/f'go2_live_local_floor_mapping_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    expected = json.loads((root/'failure.json').read_text())
    recorded = json.loads((root/'poses.json').read_text())
    output = root/'sensor_failure_replay_v1'
    output.mkdir()
    save(output, 'launch.json', dict(layout_index=i, expected_live_failure=expected,
        expected_accepted_poses=len(recorded), thresholds_changed=False,
        noisy_packets_reconstructed_and_verified=True, native_physics_used=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__, 'lewm/local_feature_depth_consensus_development.py',
             'lewm/local_inverse_depth_floor_tracking_development.py',
             'lewm/partial_floor_height_development.py')}))
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = NoisyPublicReplay(root/'native')
    motion = LocalFeatureDepthConsensusMotion(); registration = LocalFeatureDepthRegistration()
    prior = None; matches = 0; failure = None; started = time.monotonic()
    keys = ('frame', 'measured_ns', 'position_initial_body_m',
        'rotation_initial_body_from_current_body', 'reference_frame', 'rgb_sha256', 'depth_sha256')
    for frame in range(expected['acquired_frames']):
        p,d,fast,rgb,auxiliary,now = reader.packet(frame)
        raw = motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=auxiliary,now_ns=now)
        stage = 'tracking' if raw['current_pose'] is None else 'registration'
        try:
            if stage == 'tracking':
                raise ValueError('tracking failed')
            evidence = registration.observe(p,d,auxiliary,raw,now_ns=now)
            read_pose(evidence,identity=(0,0,0),now_ns=now)
        except Exception as error:
            failure = dict(frame=frame, stage=stage, reason=repr(error))
            save(output, 'terminal_raw_snapshot.json', raw)
            save(output, 'previous_registered_evidence.json', prior)
            save(output, 'terminal_floor_state.json', dict(
                registration_anchor=registration.anchor, registration_reference=registration.reference,
                registration_frame=registration.frame, registration_failed=registration.failed,
                pending_plane=None if motion.model._pending_plane is None else motion.model._pending_plane[2]))
            if stage == 'registration':
                diagnostic = bind(registration_diagnostic, measured_candidates=measured_candidates)
                save(output, 'registration_diagnostic.json', diagnostic(registration, raw, d, auxiliary, now))
            break
        if frame >= len(recorded):
            raise ValueError('replay accepted beyond the recorded live pose prefix')
        for name, pose in (('raw_pose', raw['current_pose']), ('registered_pose', evidence['current_pose'])):
            if any(pose[k] != recorded[frame][name][k] for k in keys):
                raise ValueError(f'{name} diverged from recorded pose fields at frame {frame}')
        matches += 1; prior = evidence
        if frame % 250 == 0:
            print('LIVE_FAILURE_REPLAY_FRAME', i, frame, flush=True)
    report = dict(layout_index=i, exact_pose_fields=list(keys), matched_raw_and_registered_poses=matches,
        all_recorded_pose_fields_reproduced=matches==len(recorded), failure=failure,
        live_failure_reason_matches=failure is not None and any(
            fault['stage'] == failure['stage'] and fault['reason'] == failure['reason']
            for fault in json.loads((root/'pipeline_faults.json').read_text())),
        native_physics_used=False, thresholds_changed=False, wall_seconds=time.monotonic()-started)
    save(output, 'result.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

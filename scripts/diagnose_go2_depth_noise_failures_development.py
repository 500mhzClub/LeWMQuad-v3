"""Reproduce fixed noisy failures and retain estimator evidence without physics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from scripts.replay_go2_depth_noise_tracking_development import (
    BASE, SEED, PublicReplay, perturbed_packet, configure,
    CompiledFloorStableGyroReferenceMotion, PartialHeightRegistration, read_pose)
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.sampled_plane_candidates_development import measured_candidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.partial_floor_height_development import height_correction, MISSING_EXTENT
from lewm.extended_return_budget_transport_development import composition
from lewm.floor_transport_conflict_readout_development import candidate_residuals


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(output, name, value):
    with (output/name).open('x') as sink:
        json.dump(value, sink, indent=2, default=serial)


def registration_diagnostic(registration, raw, primary, auxiliary, now):
    R = np.asarray(raw['current_pose']['rotation_initial_body_from_current_body'])
    up = R.T @ np.asarray(registration.reference['initial_up_body'])
    pairs = [measured_candidates(d['depth_m'], d['valid'], E, up)
        for d, E in ((primary, np.asarray(BODY_FROM_OPTICAL)),
                     (auxiliary, body_from_optical()))]
    plane = fit_joint_plane(*(cloud for cloud, mask in pairs), up)
    compose = height_correction if (plane['reason'] == MISSING_EXTENT
        and plane['candidate_count'] >= 100) else composition
    correction = compose(registration.anchor, raw, plane,
        identity=(0, 0, 0), now_ns=now)
    normal = np.asarray(correction['transported_reference_normal_body'])
    offset = correction['transported_reference_offset_body_m']
    return dict(joint_plane=plane, correction=correction,
        camera_residuals=[candidate_residuals(cloud, mask, normal, offset, camera=camera)
            for (cloud, mask), camera in zip(pairs, ('primary', 'auxiliary'), strict=True)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0, 2, 3), required=True)
    i = parser.parse_args().layout_index
    root = BASE/f'go2_routing_memory_persistent_native_layout{i:02d}_4800_v1_attempt_001'
    study = root/'depth_noise_2mm_tracking_601_v1'
    expected = json.loads((study/'result.json').read_text())
    original = [json.loads(line) for line in (study/'frames.jsonl').read_text().splitlines()]
    output = root/'depth_noise_2mm_failure_diagnostic_v1'
    output.mkdir()
    save(output, 'launch.json', dict(layout_index=i, seed=SEED, sigma_mm=2,
        expected_failure=expected['failure'], thresholds_changed=False,
        native_physics_used=False, maximum_concurrent_replays=2,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path('scripts/replay_go2_depth_noise_tracking_development.py'))}))
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = PublicReplay(root/'native')
    motion = CompiledFloorStableGyroReferenceMotion(activation_frame=0)
    registration = PartialHeightRegistration()
    prior = None; matches = 0; failure = None; start = time.monotonic()
    for frame in range(expected['failure']['frame'] + 1):
        p, d, fast, rgb, auxiliary, now = perturbed_packet(reader.packet(frame),
            layout=i, frame=frame, sigma_m=.002)
        raw = motion.observe(p, d, fast, auxiliary_rgb=rgb, auxiliary_depth=auxiliary, now_ns=now)
        stage = 'tracking' if raw['current_pose'] is None else 'registration'
        try:
            if stage == 'tracking':
                raise ValueError('tracking failed; details saved in tracking_failure.json')
            evidence = registration.observe(p, d, auxiliary, raw, now_ns=now)
            read_pose(evidence, identity=(0, 0, 0), now_ns=now)
        except Exception as error:
            failure = dict(frame=frame, reason=repr(error))
            save(output, 'terminal_raw_snapshot.json', raw)
            save(output, 'previous_registered_evidence.json', prior)
            save(output, 'terminal_floor_state.json', dict(
                pending_plane=None if motion.model._pending_plane is None else motion.model._pending_plane[2],
                last_measured_plane=motion.model.last_measured_plane,
                registration_anchor=registration.anchor,
                registration_reference=registration.reference,
                registration_frame=registration.frame, registration_failed=registration.failed))
            if stage == 'registration':
                save(output, 'registration_diagnostic.json',
                    registration_diagnostic(registration, raw, d, auxiliary, now))
            break
        row = dict(frame=frame,
            registered_position_m=evidence['current_pose']['position_initial_body_m'],
            reference_frame=raw['current_pose']['reference_frame'],
            promotion_reason=raw['current_pose']['promotion_reason'])
        if frame >= len(original) or row != original[frame]:
            raise ValueError(f'diagnostic replay diverged at frame {frame}')
        matches += 1; prior = evidence
        if frame % 100 == 0:
            print('DIAGNOSTIC_FRAME', i, frame, flush=True)
    result = dict(layout_index=i, exact_accepted_rows=matches, failure=failure,
        expected_failure_reproduced=failure == expected['failure'],
        full_prefix_reproduced=matches == len(original), native_physics_used=False,
        elapsed_seconds=time.monotonic()-start)
    save(output, 'result.json', result)
    print(json.dumps(result), flush=True)
    if not result['expected_failure_reproduced'] or not result['full_prefix_reproduced']:
        raise ValueError('expected fixed noisy failure not reproduced')


if __name__ == '__main__':
    main()

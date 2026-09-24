"""Retrospective floor-support diagnosis on actual delivered noisy packets."""
import argparse
import json
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.sampled_plane_candidates_development import measured_candidates as original_candidates
from lewm.local_inverse_depth_floor_development import measured_candidates as local_candidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.partial_floor_height_development import fit_gyro_height
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.replay_go2_depth_noise_tracking_development import BASE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    root = BASE/f'go2_live_local_feature_depth_noise_2mm_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    output = root/'local_floor_obstacle_frame_probe_v1.json'
    if output.exists():
        raise ValueError('preserve prior diagnostic')
    receipts = json.loads((root/'independent_depth_receipts.json').read_text())
    first_missing = next(r['frame'] for r in receipts if not r['joint_plane']['available'])
    frames = (0, first_missing, len(receipts)-1)
    configure(); reader = NoisyPublicReplay(root/'native'); results = []
    for frame in frames:
        p, d, fast, rgb, auxiliary, now = reader.packet(frame)
        saved = receipts[frame]['joint_plane']
        up = np.asarray(saved['up_body'])
        variants = {}
        for name, candidates in (('original', original_candidates), ('local_depth', local_candidates)):
            points = [candidates(packet['depth_m'], packet['valid'], E, up)[0]
                for packet, E in ((d, np.asarray(BODY_FROM_OPTICAL)), (auxiliary, body_from_optical()))]
            fit = fit_joint_plane if frame == 0 else fit_gyro_height
            variants[name] = fit(*points, up)
        results.append(dict(frame=frame, up_source='saved public gyro obstacle receipt',
            original_plane_exactly_reproduced=variants['original'] == saved,
            original_valid_pixels=[int(d['valid'].sum()), int(auxiliary['valid'].sum())],
            planes=variants))
    report = dict(layout_index=args.layout_index,
        selected_frames='startup, first unavailable independent floor, final frame',
        actual_delivered_noisy_packet_digests_verified=True,
        thresholds_changed=False, native_physics_used=False,
        recorded_gyro_up_held_fixed=True, counterfactual_navigation_executed=False,
        results=results)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps([dict(frame=r['frame'], reproduced=r['original_plane_exactly_reproduced'],
        original_count=r['planes']['original']['candidate_count'],
        local_count=r['planes']['local_depth']['candidate_count'],
        original_available=r['planes']['original']['available'],
        local_available=r['planes']['local_depth']['available']) for r in results]), flush=True)


if __name__ == '__main__':
    main()

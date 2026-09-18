"""Measure raw point support in a gyro-normal height slab on a failed recording.

This deliberately tests a different candidate selection rule. It does not
change any estimator, certify a floor, or replay counterfactual navigation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from lewm.causal_depth_observation_development import body_points as primary_points
from lewm.auxiliary_downward45_depth_observation_development import body_points as auxiliary_points
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    i = parser.parse_args().layout_index
    root = BASE/f'go2_live_local_floor_mapping_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    output = root/'raw_gyro_height_support_probe_v1.json'
    if output.exists():
        raise ValueError('preserve completed height-support probe')
    receipts = json.loads((root/'independent_depth_receipts.json').read_text())
    first = next(r['frame'] for r in receipts if not r['joint_plane']['available'])
    frames = sorted({0, max(0,first-1), first, len(receipts)-2, len(receipts)-1})
    reader = NoisyPublicReplay(root/'native'); results = []
    for frame in frames:
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        saved = receipts[frame]['joint_plane']; up = np.asarray(saved['up_body'])
        clouds = [f(d,policy,now_ns=now,stride=4) for f,d in
            ((primary_points,depth),(auxiliary_points,auxiliary))]
        points = []; cameras = []
        for j,c in enumerate(clouds):
            xyz = c['points_body_m'][c['valid']]
            xyz = xyz[xyz@up < -.15]
            points.append(xyz); cameras.extend([j]*len(xyz))
        xyz = np.concatenate(points); camera = np.asarray(cameras)
        height = xyz@up
        order = np.argsort(height,kind='stable'); h = height[order]
        right = np.searchsorted(h,h+.006,side='right')
        left = int(np.argmax(right-np.arange(len(h))))
        indices = order[left:right[left]]
        offset = -float((h[left]+h[right[left]-1])*.5)
        selected = xyz[indices]; residual = selected@up+offset
        delta = selected-selected.mean(0)
        eigenvalues = np.linalg.eigvalsh(delta.T@delta/len(selected))
        results.append(dict(frame=frame, original_floor=saved,
            valid_below_body_points=len(xyz), densest_6mm_slab_points=len(selected),
            inlier_fraction_of_below_body_points=len(selected)/len(xyz),
            camera_inlier_counts=[int(np.count_nonzero(camera[indices]==j)) for j in range(2)],
            gyro_normal_body=up.tolist(), slab_offset_body_m=offset,
            maximum_inlier_residual_m=float(np.abs(residual).max()),
            covariance_eigenvalues_m2=eigenvalues.tolist(),
            at_least_100_points_and_2cm_second_extent=bool(len(selected)>=100 and eigenvalues[1]>=.02**2),
            inliers_selected_by_height=True, excluded_points=len(xyz)-len(selected),
            floor_identity_established=False, normal_measured_from_current_points=False))
    report = dict(layout_index=i, selected_frames='startup, before/at first unavailable floor, final two acquisitions',
        recorded_independent_public_gyro_normal_held_fixed=True, original_noisy_depth_used=True,
        actual_delivered_packet_digests_verified=True, native_physics_used=False,
        original_mesh_normal_candidate_rule_replaced_in_diagnostic=True,
        slab_width_m=.006, width_source='twice existing 3-mm floor residual threshold',
        estimator_or_controller_changed=False, counterfactual_navigation_executed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), results=results)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps([dict(frame=r['frame'], original_count=r['original_floor']['candidate_count'],
        slab_count=r['densest_6mm_slab_points'], fraction=r['inlier_fraction_of_below_body_points'],
        extent_pass=r['at_least_100_points_and_2cm_second_extent']) for r in results]),flush=True)


if __name__ == '__main__':
    main()

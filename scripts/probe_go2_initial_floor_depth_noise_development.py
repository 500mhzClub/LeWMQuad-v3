"""Fixed saved-frame sensitivity probe; not a calibrated sensor or navigation test."""
import hashlib
import json
from pathlib import Path
import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.jit_floor_candidates_development import measured_candidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.two_cm_floor_extent_development import configure
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE, read


def main():
    output = BASE/'go2_initial_floor_depth_noise_sensitivity_v1_attempt_001'
    output.mkdir()
    sigmas = (0., .00025, .0005, .001, .002, .005)
    seeds = tuple(range(2026091400, 2026091405))
    config = dict(layouts=list(range(4)), frame=0, optical_depth_noise_sigma_m=sigmas,
        seeds=seeds, noise='independent zero-mean Gaussian per valid pixel per camera',
        original_invalid_rays_remain_invalid=True, out_of_range_perturbations_become_invalid=True,
        up_source='saved public-sensor initial gravity direction',
        native_state_read=False, real_sensor_noise_calibrated=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__, 'lewm/jit_floor_candidates_development.py',
             'lewm/joint_measured_floor_plane_development.py')})
    with (output/'launch.json').open('x') as f: json.dump(config, f, indent=2)
    configure(); rows=[]
    for i in range(4):
        root = BASE/f'go2_routing_memory_persistent_native_layout{i:02d}_4800_v1_attempt_001'
        packet = PublicReplay(root/'native').packet(0)
        depths = (packet[1], packet[4])
        up = np.asarray(read(root, 'independent_depth_receipts.json')[0]['initial_up_body'])
        for sigma in sigmas:
            for seed in seeds:
                rng = np.random.default_rng(seed); clouds=[]
                for depth, transform in zip(depths, (np.asarray(BODY_FROM_OPTICAL), body_from_optical())):
                    values = depth['depth_m'].copy()
                    valid = depth['valid'].copy()
                    if sigma:
                        values[valid] += rng.normal(0., sigma, int(valid.sum())).astype(np.float32)
                    valid &= (values >= .2) & (values <= 5.)
                    values[~valid] = 0.
                    clouds.append(measured_candidates(values, valid, transform, up)[0])
                plane = fit_joint_plane(*clouds, up)
                rows.append(dict(layout_index=i, root_name=root.name, sigma_m=sigma, seed=seed,
                    available=plane['available'], reason=plane['reason'],
                    candidate_count=plane['candidate_count'], camera_residuals=plane['camera_residuals']))
    summary = [dict(sigma_m=sigma, trials=sum(r['sigma_m']==sigma for r in rows),
        available=sum(r['sigma_m']==sigma and r['available'] for r in rows)) for sigma in sigmas]
    result = dict(config=config, rows=rows, summary=summary,
        independent_recorded_frames=4, zero_noise_seeds_repeat_identical_inputs=True,
        full_pose_tracking_test=False, closed_loop_navigation_test=False,
        interpretation='Sensitivity of current startup floor extraction and acceptance on four recorded initial frames; no hardware-noise calibration or robustness claim.')
    with (output/'result.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

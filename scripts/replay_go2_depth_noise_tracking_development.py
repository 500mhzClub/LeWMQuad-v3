"""Fixed 601-frame public-sensor depth-noise replay, scored after estimation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.partial_floor_height_development import PartialHeightRegistration, read_pose
from lewm.physical_execution_development import rotation_xyzw
from lewm.stable_gyro_reference_development import CompiledFloorStableGyroReferenceMotion
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import BASE, read
from scripts.in_memory_public_replay_development import PublicReplay

SEED = 2026091414
FRAME_COUNT = 601


def perturbed_packet(packet, *, layout, frame, sigma_m):
    if sigma_m == 0:
        return packet
    p, d, fast, rgb, auxiliary, now = packet
    modified = []
    for camera, depth in enumerate((d, auxiliary)):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, layout, frame, camera]))
        values = depth['depth_m'].copy(); valid = depth['valid'].copy()
        values[valid] += rng.normal(0., sigma_m, int(valid.sum())).astype(np.float32)
        valid &= (values >= .2) & (values <= 5.)
        values[~valid] = 0.
        modified.append(depth | dict(depth_m=values, valid=valid))
    d, auxiliary = modified
    rgb = from_captured_rgb(rgb['rgb'], auxiliary, p,
        measured_ns=now, available_ns=now, now_ns=now)
    return p, d, fast, rgb, auxiliary, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--sigma-mm', type=int, choices=(0, 2), required=True)
    args = parser.parse_args(); i = args.layout_index
    root = BASE/f'go2_routing_memory_persistent_native_layout{i:02d}_4800_v1_attempt_001'
    output = root/f'depth_noise_{args.sigma_mm}mm_tracking_601_v1'
    output.mkdir()
    config = dict(root_name=root.name, frames=FRAME_COUNT, sigma_mm=args.sigma_mm,
        seed=SEED, tracker='CompiledFloorStableGyroReferenceMotion',
        reference_refresh_variant_used=False, depth_noise='independent Gaussian per valid pixel, camera and frame',
        original_invalid_pixels_remain_invalid=True, out_of_range_values_become_invalid=True,
        rgb_and_gyro_unchanged=True, timestamps_and_acquisition_synchronization_unchanged=True,
        perturbed_packets_are_synthetic_not_original_capture=True,
        real_sensor_noise_calibrated=False,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__, 'lewm/stable_gyro_reference_development.py',
             'lewm/jit_floor_candidates_development.py', 'lewm/joint_measured_floor_plane_development.py')})
    with (output/'launch.json').open('x') as f:json.dump(config,f,indent=2)
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = PublicReplay(root/'native')
    motion = CompiledFloorStableGyroReferenceMotion(activation_frame=0)
    registration = PartialHeightRegistration()
    original = read(root, 'poses.json'); rows=[]; failure=None; start=time.monotonic()
    raw_matches = 0
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(FRAME_COUNT):
            try:
                p,d,fast,rgb,auxiliary,now = perturbed_packet(reader.packet(frame),
                    layout=i, frame=frame, sigma_m=args.sigma_mm/1000)
                raw = motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=auxiliary,now_ns=now)
                if raw['current_pose'] is None:
                    with (output/'tracking_failure.json').open('x') as f:json.dump(raw['terminal_failure'],f,indent=2)
                    raise ValueError('tracking failed; details saved in tracking_failure.json')
                evidence = registration.observe(p,d,auxiliary,raw,now_ns=now)
                read_pose(evidence,identity=(0,0,0),now_ns=now)
                raw_matches += raw['current_pose'] == original[frame]['raw_pose']
                row=dict(frame=frame,registered_position_m=evidence['current_pose']['position_initial_body_m'],
                    reference_frame=raw['current_pose']['reference_frame'],
                    promotion_reason=raw['current_pose']['promotion_reason'])
                rows.append(row);sink.write(json.dumps(row)+'\n');sink.flush()
                if frame%100==0:print('DEPTH_NOISE_FRAME',frame,flush=True)
            except Exception as error:
                failure=dict(frame=frame,reason=repr(error));break
    # Load physical truth only after all estimation has finished.
    frames=read(root,'native/in_memory_camera_observations.json')['frames']
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as a:physics=a['base_pose_world']
    origin=physics[frames[0]['physical_sample_index']];R0=rotation_xyzw(origin[3:])
    errors=[float(np.linalg.norm((physics[frames[r['frame']]['physical_sample_index'],:3]-origin[:3])@R0-r['registered_position_m'])) for r in rows]
    report=dict(config=config,accepted_frames=len(rows),failure=failure,
        raw_poses_equal_original=raw_matches,
        zero_noise_exact_reproduction=(args.sigma_mm==0 and raw_matches==FRAME_COUNT),
        pose_error_m=None if not errors else dict(median=float(np.median(errors)),maximum=max(errors),final=errors[-1]),
        elapsed_seconds=time.monotonic()-start,native_state_loaded_only_after_estimation=True,
        closed_loop_navigation_test=False,hardware_noise_calibrated=False)
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='config'}),flush=True)


if __name__ == '__main__':
    main()

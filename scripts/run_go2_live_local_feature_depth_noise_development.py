"""Prospective clean/noisy navigation using local correspondence depth tracking."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.current_pair_routing_memory_development import initialize_mapping
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_post_repeatability_transfer_development as transfer
from scripts.run_go2_routing_memory_scope_development import RoutingMemoryRuntime
from scripts.live_depth_noise_session_development import LiveDepthNoiseMixin
from scripts.replay_go2_depth_noise_tracking_development import SEED


class NoisyTransferSession(LiveDepthNoiseMixin, transfer.TransferCameraSession):
    pass


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.local_feature_depth_consensus_development import LocalFeatureDepthConsensusMotion
    transfer.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = LocalFeatureDepthConsensusMotion(activation_frame=0)


def initialize_registration():
    from lewm import process_registered_round_trip_development as process
    from lewm.local_feature_depth_consensus_development import LocalFeatureDepthRegistration
    from lewm.jit_floor_candidates_development import warmup
    transfer.stable.floor.configure()
    process.initialize_registration()
    warmup()
    process._registration = LocalFeatureDepthRegistration()


def annotate_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='live_local_feature_depth_noise_development_v1',
            comparison_condition=f'depth_noise_{SIGMA_MM}mm',
            new_independent_development_layout=False,
            tracker='LocalFeatureDepthConsensusMotion',
            registration='LocalFeatureDepthRegistration',
            synthetic_depth_noise=dict(sigma_mm=SIGMA_MM, seed=SEED,
                distribution='independent Gaussian per originally valid pixel, frame and camera',
                invalid_rays_remain_unknown=True, out_of_range_becomes_invalid=True,
                calibrated_to_hardware=False, rgb_and_gyro_unchanged=True),
            noisy_depth_delivered_to_all_consumers=SIGMA_MM != 0,
            floor_and_correspondence_depth_source='local_inverse_depth_5x5',
            mapping_and_independent_obstacle_algorithms_changed=False,
            routing_memory_scope='persistent', current_pair_capture_enabled=True,
            timed_live_camera_packets_unchanged=SIGMA_MM == 0,
            stored_native_depth_requires_noise_recipe=True,
            sensor_replay_class='scripts.live_depth_noise_session_development.NoisyPublicReplay',
            acquisition_noise_and_hash_cost_charged_to_execution=True,
            absolute_measurement_thresholds_changed=False,
            full_closed_loop_navigation_test=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'scripts/live_depth_noise_session_development.py',
                    'scripts/replay_go2_depth_noise_tracking_development.py',
                    'lewm/local_feature_depth_consensus_development.py',
                    'lewm/gyro_initial_camera_consensus_development.py',
                    'lewm/consecutive_retained_camera_pair_development.py',
                    'lewm/local_inverse_depth_floor_tracking_development.py',
                    'lewm/local_inverse_depth_floor_development.py',
                    'scripts/run_go2_routing_memory_scope_development.py',
                    'lewm/current_pair_routing_memory_development.py')})
    RAW_WRITE(name, value)


def finish_write(name, value):
    annotated = bind(annotate_write, SIGMA_MM=SIGMA_MM, RAW_WRITE=RAW_WRITE)
    bind(transfer.finish_write, LAYOUT_INDEX=LAYOUT_INDEX, ARM='learned',
        RAW_WRITE=annotated)(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--sigma-mm', type=int, choices=(0, 2), required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest() != transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    stable = transfer.stable
    output = stable.source.BASE/f'go2_live_local_feature_depth_noise_{args.sigma_mm}mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial live-noise trial')
    writer = bind(transfer.make_writer,
        finish_write=bind(finish_write, SIGMA_MM=args.sigma_mm))(output, i, 'learned')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed learned model assignment required')
            return RoutingMemoryRuntime(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=transfer.layouts.specification, public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_jepa',
            PacedNativeSession=partial(NoisyTransferSession,
                noise_layout_index=i, noise_sigma_mm=args.sigma_mm),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready, initialize_mapping=initialize_mapping)()


if __name__ == '__main__':
    main()

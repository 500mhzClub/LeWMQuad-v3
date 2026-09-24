"""Prospective noisy navigation with current-gyro-seeded raw height floor points."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_live_local_floor_mapping_noise_development as previous

floor=previous.floor
original=floor.original


def initialize_pose():
    from lewm import process_mapped_runtime_development as process
    from lewm.gyro_seeded_height_floor_tracking_development import GyroSeededHeightFloorMotion
    original.initialize_pose()
    process._motion=GyroSeededHeightFloorMotion()


def initialize_registration():
    from lewm import process_registered_round_trip_development as process
    from lewm.robust_height_floor_tracking_development import RobustHeightFloorRegistration
    original.initialize_registration()
    process._registration=RobustHeightFloorRegistration()


def initialize_obstacles():
    from lewm import independent_depth_process_development as process
    from lewm.robust_height_floor_tracking_development import RobustHeightIndependentObstacles
    floor.initialize_obstacles()
    process._observer=RobustHeightIndependentObstacles()


def annotate_write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='live_gyro_seeded_raw_height_floor_noise_development_v1',
            comparison_condition='gyro_seeded_raw_height_floor_depth_noise_2mm',
            reference_root_name=f'go2_live_local_floor_mapping_noise_2mm_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            tracker='GyroSeededHeightFloorMotion',registration='RobustHeightFloorRegistration',
            independent_obstacle_observer='RobustHeightIndependentObstacles',
            independent_floor_candidate_depth_source='paired_raw_depth_dominant_height_cluster',
            independent_floor_candidates_are_raw_pixel_depth=True,
            floor_and_correspondence_depth_source=dict(floor='paired_raw_depth_dominant_height_cluster',
                correspondence='local_inverse_depth_5x5'),
            floor_candidate_selection_changed=True,raw_pool_outliers_explicitly_excluded=True,
            height_cluster_minimum_pool_fraction=.25,height_cluster_maximum_refinements=8,
            tracker_height_cluster_uses_current_public_gyro=True,
            downstream_plane_and_image_acceptance_thresholds_changed=False,
            mapping_algorithm_changed=False,independent_obstacle_floor_changed_from_reference=True,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/robust_height_floor_candidates_development.py',
                    'lewm/robust_height_floor_tracking_development.py',
                    'lewm/gyro_seeded_height_floor_tracking_development.py')})
    RAW_WRITE(name,value)


def finish_write(name,value):
    annotated=bind(annotate_write,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=RAW_WRITE)
    bind(previous.finish_write,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=annotated)(name,value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=range(4),required=True)
    i=parser.parse_args().layout_index;transfer=floor.transfer;stable=transfer.stable
    if sorted(os.sched_getaffinity(0))!=transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest()!=transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    output=stable.source.BASE/f'go2_live_gyro_height_floor_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve completed or partial height-floor trial')
    writer=bind(transfer.make_writer,finish_write=finish_write)(output,i,'learned')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!='jepa':raise ValueError('fixed learned model assignment required')
            return original.RoutingMemoryRuntime(*args,registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_jepa',
            PacedNativeSession=partial(original.NoisyTransferSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=initialize_obstacles,OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=previous.initialize_mapping)()


if __name__=='__main__':main()

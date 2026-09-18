"""Fixed remaining development layouts with the successful plane-coverage implementation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.current_plane_floor_coverage_development import initialize_mapping
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_live_gyro_height_floor_noise_development as previous
from scripts.run_go2_retained_depth_cache_noise_development import initialize_pose


def annotate_write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='current_paired_plane_floor_coverage_remaining_noisy_development_v1',
            comparison_condition='current_plane_raw_floor_coverage_with_retained_depth_cache',
            reference_root_name=f'go2_live_gyro_height_floor_noise_2mm_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            tracker='RetainedDepthCacheMotion',retained_local_depth_cache=True,
            retained_local_depth_cache_maximum_entries=32,
            mapper='CurrentPlaneFloorRoutingMap',mapping_algorithm_changed=True,
            mapping_floor_depth_source=dict(current_plane_available='original_raw_pixel_depth',
                current_plane_unavailable='original_local_inverse_depth_5x5'),
            mapping_floor_is_raw_pixel_depth=False,
            mapping_floor_is_raw_pixel_depth_when_current_plane_available=True,
            mapping_floor_acceptance_thresholds_changed=True,
            mapping_floor_classifier_changed=True,
            mapping_floor_height_tolerance_m=.01,
            mapping_pixel_mesh_orientation_replaced_by_current_paired_plane=True,
            mapping_complete_valid_pixel_rectangle_required=True,
            independent_obstacle_floor_changed_from_reference=False,
            comparison_also_changes_retained_depth_cache=True,
            isolated_live_mapping_causal_effect_established=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/current_plane_floor_coverage_development.py',
                    'lewm/retained_depth_cache_tracking_development.py',
                    'scripts/run_go2_retained_depth_cache_noise_development.py')})
    RAW_WRITE(name,value)


def finish_write(name,value):
    annotated=bind(annotate_write,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=RAW_WRITE)
    bind(previous.finish_write,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=annotated)(name,value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--layout-index",type=int,choices=(0,1,3),required=True)
    i=parser.parse_args().layout_index
    floor=previous.floor;transfer=floor.transfer;stable=transfer.stable;original=previous.original
    if sorted(os.sched_getaffinity(0))!=transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest()!=transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    output=stable.source.BASE/f'go2_current_plane_coverage_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve prior plane-coverage trial')
    writer=bind(transfer.make_writer,finish_write=finish_write)(output,i,'learned')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!='jepa':raise ValueError('fixed learned model required')
            return original.RoutingMemoryRuntime(*args,registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_jepa',
            PacedNativeSession=partial(original.NoisyTransferSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=previous.initialize_obstacles,OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=initialize_mapping)()


if __name__=='__main__':main()

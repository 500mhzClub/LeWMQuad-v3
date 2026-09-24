"""One prospective noisy layout-0 test of bounded retained depth reuse."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_live_gyro_height_floor_noise_development as previous


def initialize_pose():
    from lewm import process_mapped_runtime_development as process
    from lewm.retained_depth_cache_tracking_development import RetainedDepthCacheMotion
    previous.original.initialize_pose()
    process._motion=RetainedDepthCacheMotion()


def annotate_write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='retained_local_depth_cache_noisy_layout0_development_v1',
            comparison_condition='retained_depth_cache_32_entries',
            reference_root_name='go2_live_gyro_height_floor_noise_2mm_native_layout00_4800_v1_attempt_001',
            tracker='RetainedDepthCacheMotion',retained_local_depth_cache=True,
            retained_local_depth_cache_maximum_entries=32,
            local_depth_estimator_and_pose_acceptance_unchanged=True,
            independent_obstacle_floor_changed_from_reference=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/retained_depth_cache_tracking_development.py')})
    RAW_WRITE(name,value)


def finish_write(name,value):
    annotated=bind(annotate_write,RAW_WRITE=RAW_WRITE)
    bind(previous.finish_write,LAYOUT_INDEX=0,RAW_WRITE=annotated)(name,value)


def main():
    floor=previous.floor;transfer=floor.transfer;stable=transfer.stable;original=previous.original
    if sorted(os.sched_getaffinity(0))!=transfer.CPU_GROUPS[0]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest()!=transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    output=stable.source.BASE/'go2_retained_depth_cache_noise_2mm_native_layout00_4800_v1_attempt_001'
    stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve prior cached trial')
    writer=bind(transfer.make_writer,finish_write=finish_write)(output,0,'learned')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!='jepa':raise ValueError('fixed learned model required')
            return original.RoutingMemoryRuntime(*args,registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=0,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_jepa',
            PacedNativeSession=partial(original.NoisyTransferSession,noise_layout_index=0,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=previous.initialize_obstacles,OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=previous.previous.initialize_mapping)()


if __name__=='__main__':main()

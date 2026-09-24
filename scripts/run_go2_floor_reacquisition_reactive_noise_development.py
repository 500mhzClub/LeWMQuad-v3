"""Prospective reactive layout-3 test of holding through floor disagreement."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.floor_reacquisition_development import (
    FloorReacquisitionRuntimeMixin,initialize_registration)
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_current_plane_heading_reactive_noise_development as previous


class ReacquiringReactiveRuntime(FloorReacquisitionRuntimeMixin,previous.MatchedHeadingReactiveRuntime):
    pass


def annotate_write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='floor_reacquisition_reactive_noise_development_v1',
            comparison_condition='hold_and_reacquire_after_partial_floor_conflict',
            reference_root_name='go2_current_plane_heading_reactive_noise_2mm_native_layout03_4800_v1_attempt_001',
            registration='ReacquiringFloorRegistration',
            temporary_floor_conflict_is_missing_observation=True,
            consecutive_accepted_poses_before_resuming_planning=4,
            rejected_floor_pose_used_by_map_or_mission=False,
            floor_rejection_resets_arrival_dwell=True,
            floor_rejection_cancels_pending_commands=True,
            global_budget_continues_during_missingness=True,
            sensor_cache_mapping_and_noise_matched_to_learned=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/floor_reacquisition_development.py')})
    RAW_WRITE(name,value)


def finish_write(name,value):
    writer=bind(annotate_write,RAW_WRITE=RAW_WRITE)
    bind(previous.finish_write,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=writer)(name,value)


def main():
    i=3;transfer=previous.transfer;stable=transfer.stable
    if sorted(os.sched_getaffinity(0))!=transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest()!=transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    output=stable.source.BASE/'go2_floor_reacquisition_reactive_noise_2mm_native_layout03_4800_v1_attempt_001'
    stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve prior reacquisition trial')
    writer=bind(transfer.make_writer,finish_write=finish_write)(output,i,'reactive')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!='reactive':raise ValueError('model-free assignment required')
            return ReacquiringReactiveRuntime(*args,registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='reactive',
            PacedNativeSession=partial(previous.noisy.NoisyTransferSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=previous.learned.initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=previous.gyro.initialize_obstacles,OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=previous.learned.initialize_mapping)()


if __name__=='__main__':main()

"""Two declared floor-gap probes: reactive layout 3 and learned layout 0."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.declared_floor_gap_development import GAP_FRAMES,initialize_registration
from lewm.floor_reacquisition_development import FloorReacquisitionRuntimeMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_floor_reacquisition_reactive_noise_development as natural

previous=natural.previous


class ReacquiringLearnedRuntime(FloorReacquisitionRuntimeMixin,previous.noisy.RoutingMemoryRuntime):
    pass


def annotate_write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='declared_floor_gap_noise_development_v1',
            comparison_condition=ARM+'_declared_floor_pose_gap',
            reference_root_name=(
                'go2_floor_reacquisition_reactive_noise_2mm_native_layout03_4800_v1_attempt_001'
                if ARM=='reactive' else
                'go2_current_plane_coverage_noise_2mm_native_layout00_4800_v1_attempt_001'),
            registration='DeclaredFloorGapRegistration',
            declared_floor_pose_gap_frames=list(GAP_FRAMES),
            fault_injection=True,fault_is_calibrated_hardware_noise=False,
            rgb_gyro_and_independent_depth_continue_during_gap=True,
            withheld_floor_pose_does_not_update_anchor=True,
            mapping_floor_acceptance_thresholds_changed=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,natural.__file__,'lewm/declared_floor_gap_development.py')})
    RAW_WRITE(name,value)


def finish_write(name,value):
    writer=bind(annotate_write,ARM=ARM,RAW_WRITE=RAW_WRITE)
    writer=bind(natural.annotate_write,RAW_WRITE=writer)
    source=previous.finish_write if ARM=='reactive' else previous.learned.finish_write
    bind(source,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=writer)(name,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('learned','reactive'),required=True)
    arm=parser.parse_args().arm;i=3 if arm=='reactive' else 0
    transfer=previous.transfer;stable=transfer.stable
    if sorted(os.sched_getaffinity(0))!=transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest()!=transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    output=stable.source.BASE/f'go2_declared_floor_gap_{arm}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve prior declared-gap trial')
    writer=bind(transfer.make_writer,finish_write=bind(finish_write,ARM=arm))(output,i,arm)
    runtime_type=natural.ReacquiringReactiveRuntime if arm=='reactive' else ReacquiringLearnedRuntime
    condition='reactive' if arm=='reactive' else 'jepa'
    assignment='reactive' if arm=='reactive' else 'seed_2026091001_full_jepa'
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!=condition:raise ValueError('assigned model condition required')
            return runtime_type(*args,registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT=assignment,
            PacedNativeSession=partial(previous.noisy.NoisyTransferSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=previous.learned.initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=previous.gyro.initialize_obstacles,OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=previous.learned.initialize_mapping)()


if __name__=='__main__':main()

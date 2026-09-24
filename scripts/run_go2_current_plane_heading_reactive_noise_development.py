"""Matched-perception stronger reactive control for the noisy development cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.current_pair_routing_memory_development import RoutingMemoryScopeMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts.run_go2_heading_first_terminal_reactive_development import HeadingFirstReactiveRuntime
from scripts import run_go2_current_plane_coverage_remaining_noise_development as learned

gyro=learned.previous
floor=gyro.floor
noisy=gyro.original
mapping=gyro.previous
transfer=noisy.transfer


class MatchedHeadingReactiveRuntime(RoutingMemoryScopeMixin,HeadingFirstReactiveRuntime):
    pass


def annotate_write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='current_plane_heading_first_reactive_noisy_development_v1',
            comparison_condition='heading_first_instantaneous_reactive',
            reference_root_name=f'go2_current_plane_coverage_noise_2mm_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            heading_first_terminal_control=True,terminal_heading_tolerance_rad=.1,
            learned_model_used_for_action_selection=False,candidate_future_outcomes_evaluated=False,
            sensor_cache_mapping_and_noise_matched_to_learned=True,
            mapping_algorithm_changed=False,mapping_floor_classifier_changed=False,
            mapping_current_plane_floor_classifier=True,
            mapping_and_independent_obstacle_algorithms_changed=False,
            independent_obstacle_floor_algorithm_changed=False,
            independent_obstacle_floor_changed_from_reference=False,
            comparison_also_changes_retained_depth_cache=False,
            comparison_is_complete_predictive_vs_instantaneous_control=True,
            isolated_jepa_training_effect_established=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'scripts/run_go2_heading_first_terminal_reactive_development.py',
                    'lewm/heading_first_terminal_reactive_development.py',
                    'lewm/pulsed_fine_goal_reactive_development.py',
                    'lewm/continuous_reactive_runtime_development.py',
                    'lewm/continuous_reactive_selection_development.py')})
    RAW_WRITE(name,value)


def finish_write(name,value):
    # Reuse sensor/mapping annotations without their learned-arm finish wrappers.
    writer=bind(annotate_write,LAYOUT_INDEX=LAYOUT_INDEX,RAW_WRITE=RAW_WRITE)
    for annotation in reversed((noisy.annotate_write,floor.annotate_write,
            mapping.annotate_write,gyro.annotate_write,learned.annotate_write)):
        writer=bind(annotation,LAYOUT_INDEX=LAYOUT_INDEX,SIGMA_MM=2,RAW_WRITE=writer)
    bind(transfer.finish_write,LAYOUT_INDEX=LAYOUT_INDEX,ARM='reactive',RAW_WRITE=writer)(name,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-index',type=int,choices=range(4),required=True)
    i=parser.parse_args().layout_index;stable=transfer.stable
    if sorted(os.sched_getaffinity(0))!=transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(transfer.INVENTORY.read_bytes()).hexdigest()!=transfer.INVENTORY_SHA256:
        raise ValueError('fixed development layout inventory required')
    output=stable.source.BASE/f'go2_current_plane_heading_reactive_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve prior reactive trial')
    writer=bind(transfer.make_writer,finish_write=finish_write)(output,i,'reactive')
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!='reactive':raise ValueError('model-free reactive assignment required')
            return MatchedHeadingReactiveRuntime(*args,registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=transfer.layouts.specification,public_mission=transfer.layouts.public_mission,
            MODEL_ASSIGNMENT='reactive',
            PacedNativeSession=partial(noisy.NoisyTransferSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=learned.initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=gyro.initialize_obstacles,OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=learned.initialize_mapping)()


if __name__=='__main__':main()

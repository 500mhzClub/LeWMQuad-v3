"""Four frozen controllers on four new development mazes with shared perception."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm import post_training_comparison_layouts_development as layouts
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_current_plane_matched_training_noise_development as training
from scripts import run_go2_current_plane_heading_reactive_noise_development as reactive
from scripts.compact_depth_retention_session_development import CompactDepthRetentionMixin
from scripts.independent_round_trip_session_development import IndependentRoundTripPhysicalInit
from scripts.in_memory_paired_camera_session_development import LzmaRawDepthPairedCameraSession
from scripts.live_depth_noise_session_development import LiveDepthNoiseMixin

learned = training.learned
gyro = learned.previous
noisy = gyro.original
transfer = noisy.transfer
stable = transfer.stable
CONDITIONS = ('jepa', 'direct', 'supervised_rollout', 'reactive')
INVENTORY = Path('docs/go2_post_training_comparison_layout_inventory_2026-09-15.json')
INVENTORY_SHA256 = '7ee6653289d294189abb0f7641f91b8d13f26d5670c952eaae0f45c7acda731e'


class PostTrainingPhysicalInit(IndependentRoundTripPhysicalInit):
    __init__ = bind(IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class PostTrainingCameraSession(LiveDepthNoiseMixin, CompactDepthRetentionMixin,
        LzmaRawDepthPairedCameraSession, PostTrainingPhysicalInit):
    pass


def annotate(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='post_training_four_controller_transfer_noise_development_v1',
            comparison_condition=CONDITION,
            comparison='three_frozen_training_methods_and_heading_first_reactive',
            reference_root_name=None, fresh_layout_inventory=layouts.build_inventory(),
            frozen_layout_inventory_sha256=INVENTORY_SHA256,
            new_independent_development_layout=True,
            layout_novelty_scope='distinct_from_explicit_64_layout_source_registry',
            planned_conditions=list(CONDITIONS), planned_layout_count=4,
            sensor_cache_mapping_and_noise_matched_to_learned=True,
            shared_perception_across_all_conditions=True,
            shared_planner_and_recovery_across_training_conditions=True,
            reactive_recovery_rules_differ_from_predictive=True,
            isolated_predictive_scoring_effect_established=False,
            final_evaluation=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, training.__file__, reactive.__file__,
                    'lewm/post_training_comparison_layouts_development.py',
                    'scripts/run_go2_heading_first_terminal_reactive_development.py',
                    'lewm/heading_first_terminal_reactive_development.py',
                    'lewm/pulsed_fine_goal_reactive_development.py',
                    'lewm/continuous_reactive_runtime_development.py',
                    'lewm/continuous_reactive_selection_development.py')})
    RAW_WRITE(name, value)


def make_writer(output, index, condition):
    predictive = condition != 'reactive'
    assignment = f'seed_2026091001_full_{condition}' if predictive else 'reactive'
    final = bind(annotate, CONDITION=condition, RAW_WRITE=bind(stable.source.write, OUTPUT=output))
    coherent = bind(training.annotate_gyro_coherent, LAYOUT_INDEX=index, RAW_WRITE=final)
    if predictive:
        method = bind(training.annotate_write, CONDITION=condition, LAYOUT_INDEX=index, RAW_WRITE=coherent)
        finish = bind(learned.finish_write, LAYOUT_INDEX=index, RAW_WRITE=method)
    else:
        finish = bind(reactive.finish_write, LAYOUT_INDEX=index, RAW_WRITE=coherent)
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=not predictive,
        ARRIVAL_ENTRY_PRIORITY=predictive, source=SimpleNamespace(write=finish))
    return bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=assignment, USE_CLEARANCE_TURN_RECOVERY=predictive,
        USE_STEPWISE_TURN_RECOVERY=predictive, USE_TERMINAL_POSITION_PRIORITY=predictive,
        USE_PROGRESS_REJOINING=predictive, USE_PREDICTIVE_ARRIVAL_HOLD=predictive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != INVENTORY_SHA256:
        raise ValueError('fixed prospective development layout inventory required')
    output = stable.source.BASE/f'go2_post_training_transfer_{args.condition}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve every prospective assignment')
    predictive = args.condition != 'reactive'
    assignment = f'seed_2026091001_full_{args.condition}' if predictive else 'reactive'
    runtime_type = noisy.RoutingMemoryRuntime if predictive else reactive.MatchedHeadingReactiveRuntime
    writer = make_writer(output, i, args.condition)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != args.condition:
                raise ValueError('frozen controller assignment required')
            return runtime_type(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment,
            PacedNativeSession=partial(PostTrainingCameraSession, noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=training.initialize_gyro_coherent_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=gyro.initialize_obstacles, OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=learned.initialize_mapping)()


if __name__ == '__main__': main()

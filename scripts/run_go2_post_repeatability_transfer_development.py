"""Fixed learned/reactive comparison on four prospective development mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm import post_repeatability_transfer_layouts_development as layouts
from lewm.nearby_panorama_directed_view_development import NearbyPanoramaRuntimeMixin
from lewm.pulsed_fine_goal_reactive_development import PulsedFineGoalReactiveRuntime
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_hold_relative_heading_recovery_development as original
from scripts.compact_depth_retention_session_development import CompactDepthRetentionMixin
from scripts.independent_round_trip_session_development import IndependentRoundTripPhysicalInit
from scripts.in_memory_paired_camera_session_development import LzmaRawDepthPairedCameraSession


stable = original.prior.prior.fresh.stable
CPU_GROUPS = original.prior.CPU_GROUPS
INVENTORY = Path('docs/go2_post_repeatability_transfer_layout_inventory_2026-09-14.json')
INVENTORY_SHA256 = '021ed31825a0b59ab2cfa7e5cdf8ad02f0f2ff5816ba0c4330301555373487d7'


class TransferPhysicalInit(IndependentRoundTripPhysicalInit):
    __init__ = bind(IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class TransferCameraSession(CompactDepthRetentionMixin,
        LzmaRawDepthPairedCameraSession, TransferPhysicalInit):
    pass


class TransferReactiveRuntime(NearbyPanoramaRuntimeMixin, PulsedFineGoalReactiveRuntime):
    """Keep the learned arm's observed frontier-view strategy without forecasts."""


def finish_write(name, value):
    if name == 'launch.json':
        predictive = ARM == 'learned'
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != CPU_GROUPS[LAYOUT_INDEX % 2]:
            raise ValueError('assigned physical CPU group required')
        profile = dict(recording='native_pixels_with_captured_derived_packet_digests',
            derived_depth_arrays_retained=False, packet_hashing_during_capture=True,
            hashing_cost_charged_to_execution=True, cpu_affinity=affinity,
            paired_layout_index=LAYOUT_INDEX ^ 1, maximum_planned_native_owners=2,
            actual_concurrent_owners_not_inferred_from_profile=True,
            real_time_qualified=False)
        correction_root, correction_hash = original.prior.prior.FITS['jepa']
        value = value | dict(experiment='post_repeatability_transfer_development_v1',
            comparison_condition=ARM, training_condition='jepa' if predictive else None,
            closed_loop_motion_residual_fit_sha256=correction_hash if predictive else None,
            motion_residual_correction_root=correction_root if predictive else None,
            full_reserve_heading_release=predictive,
            terminal_approach_excluded_from_heading_release=True,
            survey_clearance_reposition=predictive,
            hold_relative_heading_recovery=predictive,
            nearby_panorama_directed_view=True, nearby_panorama_radius_m=.25,
            execution_profile=profile, fresh_layout_inventory=layouts.build_inventory(),
            frozen_layout_inventory_sha256=INVENTORY_SHA256,
            new_independent_development_layout=True, final_evaluation=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, original.__file__, original.prior.__file__,
                    original.prior.prior.__file__, original.prior.prior.matched.__file__,
                    original.prior.prior.fresh.__file__,
                    'lewm/post_repeatability_transfer_layouts_development.py',
                    'lewm/fresh_stable_reference_layouts_development.py',
                    'lewm/independent_round_trip_layouts_development.py',
                    'scripts/independent_round_trip_session_development.py',
                    'scripts/compact_depth_retention_session_development.py',
                    'lewm/hold_relative_heading_recovery_development.py',
                    'lewm/nearby_panorama_directed_view_development.py',
                    'lewm/survey_clearance_reposition_development.py',
                    'lewm/full_reserve_heading_release_development.py',
                    'lewm/matched_motion_residual_runtime_development.py')})
        RAW_WRITE('parallel_execution_profile.json', profile)
    RAW_WRITE(name, value)


def make_writer(output, layout_index, arm):
    predictive = arm == 'learned'
    assignment = 'seed_2026091001_full_jepa' if predictive else 'reactive'
    final_write = bind(finish_write, LAYOUT_INDEX=layout_index, ARM=arm,
        RAW_WRITE=bind(stable.source.write, OUTPUT=output))
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=not predictive,
        ARRIVAL_ENTRY_PRIORITY=predictive, source=SimpleNamespace(write=final_write))
    return bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=assignment, USE_CLEARANCE_TURN_RECOVERY=predictive,
        USE_STEPWISE_TURN_RECOVERY=predictive, USE_TERMINAL_POSITION_PRIORITY=predictive,
        USE_PROGRESS_REJOINING=predictive, USE_PREDICTIVE_ARRIVAL_HOLD=predictive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--arm', choices=('learned', 'reactive'), required=True)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != CPU_GROUPS[args.layout_index % 2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != INVENTORY_SHA256:
        raise ValueError('preserve the prospective layout roster')
    output = stable.source.BASE/f'go2_post_repeatability_transfer_{args.arm}_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve completed or partial transfer run')
    assignment = 'seed_2026091001_full_jepa' if args.arm == 'learned' else 'reactive'
    writer = make_writer(output, args.layout_index, args.arm)
    runtime_type = original.HeadingRecoveryRuntime if args.arm == 'learned' else TransferReactiveRuntime
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            expected = 'jepa' if args.arm == 'learned' else 'reactive'
            if kwargs.get('condition') != expected:
                raise ValueError('fixed model assignment required')
            return runtime_type(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=TransferCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

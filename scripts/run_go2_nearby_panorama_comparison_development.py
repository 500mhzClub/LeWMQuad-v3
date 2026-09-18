"""Compare full and directed frontier revisits with matched compact execution."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.nearby_panorama_directed_view_development import NearbyPanoramaRuntimeMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_survey_reposition_native_development as prior
from scripts.compact_depth_retention_session_development import CompactDepthRetentionMixin


CPU_GROUPS = {0: list(range(8)) + list(range(16, 24)),
              1: list(range(8, 16)) + list(range(24, 32))}


class CompactFreshCameraSession(CompactDepthRetentionMixin, prior.fresh.FreshCameraSession):
    pass


class DirectedRevisitRuntime(NearbyPanoramaRuntimeMixin, prior.MatchedSurveyRepositionRuntime):
    pass


def finish_write(name, value):
    if name == 'launch.json':
        affinity = sorted(os.sched_getaffinity(0))
        if affinity != CPU_GROUPS[LAYOUT_INDEX]:
            raise ValueError('assigned physical CPU group required')
        profile = dict(recording='native_pixels_with_captured_derived_packet_digests',
            derived_depth_arrays_retained=False, packet_hashing_during_capture=True,
            hashing_cost_charged_to_execution=True, cpu_affinity=affinity,
            paired_layout_index=1-LAYOUT_INDEX, maximum_planned_native_owners=2,
            actual_concurrent_owners_not_inferred_from_profile=True,
            real_time_qualified=False)
        value = value | dict(experiment='nearby_panorama_directed_view_comparison',
            comparison_condition=CONDITION, training_condition='jepa',
            closed_loop_motion_residual_fit_sha256=CORRECTION_HASH,
            motion_residual_correction_root=CORRECTION_ROOT,
            full_reserve_heading_release=True, terminal_approach_excluded_from_heading_release=True,
            survey_clearance_reposition=True, nearby_panorama_directed_view=CONDITION=='directed',
            nearby_panorama_radius_m=.25, execution_profile=profile,
            fresh_layout_inventory=prior.fresh.layouts.build_inventory(),
            extra_sources=value['extra_sources'] | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, prior.__file__, prior.matched.__file__, prior.fresh.__file__,
                    'lewm/survey_clearance_reposition_development.py',
                    'lewm/full_reserve_heading_release_development.py',
                    'lewm/matched_motion_residual_runtime_development.py',
                    'lewm/nearby_panorama_directed_view_development.py',
                    'lewm/fresh_stable_reference_layouts_development.py',
                    'lewm/independent_round_trip_layouts_development.py',
                    'scripts/independent_round_trip_session_development.py',
                    'scripts/compact_depth_retention_session_development.py')})
        RAW_WRITE('parallel_execution_profile.json', profile)
    RAW_WRITE(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--condition', choices=('baseline', 'directed'), required=True)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != CPU_GROUPS[args.layout_index]:
        raise ValueError('launch with taskset using the assigned CPU group')
    stable = prior.fresh.stable; layouts = prior.fresh.layouts
    assignment = 'seed_2026091001_full_jepa'
    output = stable.source.BASE/f'go2_nearby_panorama_{args.condition}_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists():
        raise ValueError('preserve previous nearby-panorama trial')
    correction_root, correction_hash = prior.FITS['jepa']
    final_write = bind(finish_write, CORRECTION_HASH=correction_hash,
        CORRECTION_ROOT=correction_root, LAYOUT_INDEX=args.layout_index,
        CONDITION=args.condition, RAW_WRITE=bind(stable.source.write, OUTPUT=output))
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=False,
        ARRIVAL_ENTRY_PRIORITY=True, source=SimpleNamespace(write=final_write))
    writer = bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=assignment, USE_CLEARANCE_TURN_RECOVERY=True,
        USE_STEPWISE_TURN_RECOVERY=True, USE_TERMINAL_POSITION_PRIORITY=True,
        USE_PROGRESS_REJOINING=True, USE_PREDICTIVE_ARRIVAL_HOLD=True)
    runtime_class = DirectedRevisitRuntime if args.condition=='directed' else prior.MatchedSurveyRepositionRuntime
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed JEPA model required')
            return runtime_class(*runtime_args, registration_executor=executor,
                navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=CompactFreshCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

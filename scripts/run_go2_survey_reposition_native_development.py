"""Prospective survey-repositioning trial with the prior serial JEPA setup."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.matched_motion_residual_runtime_development import FITS, MatchedMotionResidualRuntime
from lewm.process_registered_round_trip_development import registration_ready
from lewm.survey_clearance_reposition_development import SurveyClearanceRepositionRuntime
from scripts import run_go2_fresh_stable_reference_native_development as fresh
from scripts import run_go2_matched_training_navigation_development as matched


class MatchedSurveyRepositionRuntime(SurveyClearanceRepositionRuntime, MatchedMotionResidualRuntime):
    """Use the same fixed JEPA fit with the single survey recovery change."""


def finish_write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='survey_clearance_reposition_native_development',
            comparison='prospective_repositioning_against_recorded_serial_jepa',
            training_condition='jepa', closed_loop_motion_residual_fit_sha256=CORRECTION_HASH,
            motion_residual_correction_root=CORRECTION_ROOT,
            full_reserve_heading_release=True, terminal_approach_excluded_from_heading_release=True,
            survey_clearance_reposition=True, fresh_layout_inventory=fresh.layouts.build_inventory(),
            baseline_root_name=f'go2_matched_training_jepa_heading_release_native_layout{LAYOUT_INDEX:02d}_4800_v1_attempt_001',
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, matched.__file__, fresh.__file__,
                    'lewm/survey_clearance_reposition_development.py',
                    'lewm/full_reserve_heading_release_development.py',
                    'lewm/matched_motion_residual_runtime_development.py',
                    'lewm/fresh_stable_reference_layouts_development.py',
                    'lewm/independent_round_trip_layouts_development.py',
                    'scripts/independent_round_trip_session_development.py')})
    RAW_WRITE(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0, 1), required=True)
    args = parser.parse_args(); stable = fresh.stable; layouts = fresh.layouts
    assignment = 'seed_2026091001_full_jepa'
    output = stable.source.BASE/f'go2_survey_reposition_matched_jepa_native_layout{args.layout_index:02d}_4800_v1_attempt_001'
    stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve previous survey-reposition trial')
    correction_root, correction_hash = FITS['jepa']
    raw_write = bind(stable.source.write, OUTPUT=output)
    final_write = bind(finish_write, CORRECTION_HASH=correction_hash,
        CORRECTION_ROOT=correction_root, LAYOUT_INDEX=args.layout_index, RAW_WRITE=raw_write)
    stable_writer = bind(stable.write, OUTPUT=output, REACTIVE=False,
        ARRIVAL_ENTRY_PRIORITY=True, source=SimpleNamespace(write=final_write))
    writer = bind(stable.configuration.write, _write=stable_writer,
        MODEL_ASSIGNMENT=assignment, USE_CLEARANCE_TURN_RECOVERY=True,
        USE_STEPWISE_TURN_RECOVERY=True, USE_TERMINAL_POSITION_PRIORITY=True,
        USE_PROGRESS_REJOINING=True, USE_PREDICTIVE_ARRIVAL_HOLD=True)
    stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=stable.floor.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'jepa':
                raise ValueError('fixed JEPA model required')
            return MatchedSurveyRepositionRuntime(*runtime_args,
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)

        bind(stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=args.layout_index,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment, PacedNativeSession=fresh.FreshCameraSession,
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stable.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=stable.floor.initialize_obstacles,
            OBSTACLE_READY=stable.obstacles_ready,
            initialize_mapping=stable.floor.initialize_mapping)()


if __name__ == '__main__':
    main()

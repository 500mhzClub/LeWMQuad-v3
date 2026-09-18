"""Eight fixed learned/control assignments on four fresh development mazes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from lewm.camera_frontier_visits_development import CameraFrontierRuntimeMixin
from lewm.combined_motion_source_development import CombinedMotionSourceMixin, SOURCES
from lewm import combined_perception_motion_layouts_development as layouts
from scripts import run_go2_contact_score_ablation_development as contact
from scripts import run_go2_camera_frontier_viewpoint_development as camera
from scripts.run_go2_coherent_reference_refresh_development import initialize_pose

cohort = contact.cohort
INVENTORY = Path('docs/go2_combined_perception_motion_layout_inventory_2026-09-15.json')
INVENTORY_SHA256 = 'b1ab545f649d0efc8a9f406418a5cc454da70a18aa434c029ce99ae45d0cef4e'


class FreshPhysicalInit(cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(cohort.LiveDepthNoiseMixin, cohort.CompactDepthRetentionMixin,
                         cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class MotionRuntime(CombinedMotionSourceMixin, CameraFrontierRuntimeMixin, contact.ContactScoreRuntime):
    pass


def annotate_combined(name, value):
    if name == 'launch.json':
        learned = SOURCE == 'learned'
        value = value | dict(experiment='combined_perception_motion_comparison_v1',
            comparison='learned_corrected_motion_vs_pose_command_motion',
            comparison_condition=SOURCE, motion_prediction_source=SOURCE,
            forecast_xy_source=SOURCE, forecast_yaw_source='learned' if learned else 'command',
            learned_yaw_retained=learned, learned_yaw_and_contact_retained=False,
            neural_xy_used_for_scoring=learned, neural_outcomes_used_for_scoring=learned,
            both_yaw_alternatives_computed_in_both_arms=True,
            learned_xy_includes_frozen_residual_correction=True,
            pose_command_xy_remains_a_fitted_model=True,
            planned_conditions=list(SOURCES), planned_layout_indices=[0, 1, 2, 3],
            planned_layout_count=4, planned_native_assignments=8,
            fixed_first_source_by_layout=['learned', 'pose_command', 'pose_command', 'learned'],
            recent_reference_refresh_from_accepted_anchor=True,
            maximum_recent_reference_age_ns=400_000_000,
            bridge_measurements_promoted=False,
            bridge_budget_and_measurement_acceptance_rules_unchanged=True,
            tracker_identical_to_original_yaw_reference=False,
            reference_root_name=None, fresh_layout_inventory=layouts.build_inventory(),
            frozen_layout_inventory_sha256=INVENTORY_SHA256,
            new_independent_development_layout=True,
            layouts_previously_exposed_by_fixed_four_controller_cohort=False,
            layout_novelty_scope='distinct_from_explicit_68_layout_source_registry',
            xy_yaw_and_physical_guards_unchanged=False,
            contact_and_physical_guards_unchanged=True,
            predictive_planning_in_both_arms=True, fully_model_free_controller=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/combined_perception_motion_layouts_development.py',
                    'lewm/combined_motion_source_development.py',
                    'lewm/yaw_source_ablation_development.py',
                    'lewm/commanded_planar_motion_development.py',
                    'lewm/coherent_reference_refresh_development.py',
                    'lewm/recent_anchored_reference_refresh_development.py',
                    'scripts/run_go2_coherent_reference_refresh_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(annotate_combined, SOURCE=SOURCE, RAW_WRITE=RAW_WRITE)
    viewing = bind(camera.annotate_camera, RAW_WRITE=emit)
    scoring = bind(contact.annotate_contact, MODE='disabled', RAW_WRITE=viewing)
    bind(contact.xy.annotate, XY_SOURCE=SOURCE, RAW_WRITE=scoring)(name, value)


def make_writer(output, index, source):
    return bind(cohort.make_writer, annotate=bind(annotate, SOURCE=source))(
        output, index, 'supervised_rollout')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--motion-source', choices=SOURCES, required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned physical CPU group required')
    output = cohort.stable.source.BASE / (
        f'go2_combined_perception_motion_{args.motion_source}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve all fixed assignments and failures')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != INVENTORY_SHA256:
        raise ValueError('fixed fresh development inventory required')
    writer = make_writer(output, i, args.motion_source); cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args, **kwargs):
            if kwargs.get('condition') != 'supervised_rollout':
                raise ValueError('same frozen supervised model required')
            return MotionRuntime(*runtime_args, motion_prediction_source=args.motion_source,
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

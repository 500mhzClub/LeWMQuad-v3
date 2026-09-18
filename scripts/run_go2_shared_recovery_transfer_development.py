"""Ten fresh-maze assignments with shared perception and signed view recovery."""
import argparse
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.signed_veto_view_recovery_development import SignedVetoViewMixin
from lewm import shared_recovery_transfer_layouts_development as layouts
from scripts import run_go2_transport_conditioned_floor_recovery_development as previous
from scripts.run_go2_combined_tracking_floor_recovery_development import initialize_pose

study = previous.study
ARMS = ('reactive', 'seed_2026091001_full_jepa', 'seed_2026091001_full_direct',
    'seed_2026091001_full_supervised_rollout', 'pose_command')
ASSIGNMENTS = tuple((0, arm) for arm in ARMS) + tuple((1, arm) for arm in reversed(ARMS))
ROOT = 'go2_shared_recovery_transfer_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
INVENTORY = Path('docs/go2_shared_recovery_transfer_layout_inventory_2026-09-15.json')
INVENTORY_SHA256 = 'ff645dbfc2af51d7c0a8b805c6c6fa107a81ccafd11dfaf132ee9006c6d40ff8'


class FreshPhysicalInit(study.cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(study.cohort.LiveDepthNoiseMixin, study.cohort.CompactDepthRetentionMixin,
        study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class SignedLearnedRuntime(SignedVetoViewMixin, previous.previous.LearnedRecoveryRuntime):
    pass


class SignedReactiveRuntime(SignedVetoViewMixin, previous.previous.ReactiveRecoveryRuntime):
    pass


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='shared_recovery_transfer_v1',
            comparison='three_training_methods_fitted_motion_and_reactive_with_shared_recovery',
            reference_root_name=None, planned_conditions=list(ARMS),
            planned_layout_indices=[0,1], planned_layout_count=2, planned_native_assignments=10,
            fixed_dispatch_pairs=None, fixed_sequential_assignments=list(ASSIGNMENTS),
            fresh_layout_inventory=layouts.build_inventory(),
            frozen_layout_inventory_sha256=INVENTORY_SHA256,
            new_independent_development_layout=True, exposed_development_layout=False,
            repeated_exposed_development_maze=False,
            layout_novelty_scope='distinct_from_explicit_78_layout_development_registry',
            actual_runtime_class='SignedReactiveRuntime' if value['study_arm']=='reactive' else 'SignedLearnedRuntime',
            tracker='CachedMomentsDeferredCopyMotion', raw_tracker_changed=True,
            perception_and_recovery_shared_across_conditions=True,
            shared_translation_veto_view_direction=True, view_recovery_angle_degrees=45,
            view_recovery_completion_tolerance_rad=.1,
            straight_veto_and_primary_blind_view_direction_unchanged=True,
            command_guards_unchanged=True, training_seed_selection_used_runtime_outcomes=False,
            native_jobs_run_sequentially=True, hardware_validated=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/shared_recovery_transfer_layouts_development.py',
                    'lewm/signed_veto_view_recovery_development.py',
                    'lewm/veto_view_round_trip_development.py',
                    'lewm/cached_moments_deferred_copy_tracking_development.py',
                    'lewm/cached_floor_moments_development.py',
                    'lewm/deferred_registration_copy_development.py',
                    'scripts/run_go2_combined_tracking_floor_recovery_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(previous.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    base = study.cohort.stable.source.BASE
    position = ASSIGNMENTS.index((args.layout_index, args.arm))
    if position:
        index, arm = ASSIGNMENTS[position-1]
        if not (base/ROOT.format(index=index, arm=arm)/'continuous_native_arrival_evaluation.json').is_file():
            raise ValueError('complete and evaluate the preceding fixed assignment')
    if shutil.disk_usage(base).free < 4 * 1024**3:
        raise ValueError('four GiB free required for one full-budget recording')
    gyro = SimpleNamespace(**(vars(study.cohort.gyro) | dict(initialize_obstacles=previous.initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(study.cohort) | dict(gyro=gyro)))
    bind(study.main, ROOT=ROOT, ARMS=ARMS, layouts=layouts, INVENTORY=INVENTORY,
        INVENTORY_SHA256=INVENTORY_SHA256, FreshCameraSession=FreshCameraSession,
        annotate=annotate, cohort=cohort, initialize_pose=initialize_pose,
        initialize_registration=previous.initialize_registration,
        SeededPredictiveRuntime=SignedLearnedRuntime, ReactiveRuntime=SignedReactiveRuntime)()


if __name__ == '__main__': main()

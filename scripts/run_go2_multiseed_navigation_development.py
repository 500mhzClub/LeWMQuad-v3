"""Fixed 22-mission comparison of three training methods and three seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.floor_reacquisition_development import FloorReacquisitionRuntimeMixin, initialize_registration
from lewm.seeded_motion_correction_development import SeededMotionCorrectionMixin, registry, registry_identity, REGISTRY_SHA256
from lewm.process_registered_round_trip_development import registration_ready
from lewm import multiseed_navigation_layouts_development as layouts
from scripts import run_go2_stopping_projection_transfer_development as transfer
from scripts.run_go2_batched_consensus_followup_development import initialize_pose

cohort = transfer.cohort
INVENTORY = Path('docs/go2_multiseed_navigation_layout_inventory_2026-09-15.json')
INVENTORY_SHA256 = 'bf831e45cee5435c982b34a817e059cc16c351de17a698c432878c42e586d512'
SEEDS = (2026091001, 2026091401, 2026091402)
METHODS = ('jepa', 'direct', 'supervised_rollout')
ARMS = tuple(f'seed_{s}_full_{c}' for s in SEEDS for c in METHODS) + ('pose_command', 'reactive')
# Each arm runs once on each layout. Offset the second layout by five arms.
PAIRS = tuple(((0, arm), (1, ARMS[(i+5) % len(ARMS)])) for i,arm in enumerate(ARMS))
ROOT = 'go2_multiseed_navigation_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class FreshPhysicalInit(cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(cohort.LiveDepthNoiseMixin, cohort.CompactDepthRetentionMixin,
        cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class SeededPredictiveRuntime(FloorReacquisitionRuntimeMixin, SeededMotionCorrectionMixin,
        transfer.stopping.StoppingAwareRuntime):
    pass


class ReactiveRuntime(FloorReacquisitionRuntimeMixin, transfer.CurrentReactiveRuntime):
    pass


def assignment_for(arm):
    return 'seed_2026091001_full_supervised_rollout' if arm == 'pose_command' else arm


def finish(name, value):
    if name == 'launch.json':
        assignment = assignment_for(ARM)
        variant = 'no_rgb' if '_no_rgb_' in assignment else 'full'
        entry = None if ARM == 'reactive' else registry(variant)[assignment]
        value = value | dict(experiment='multiseed_navigation_development_v1',
            comparison='three_training_methods_three_seeds_and_fitted_reactive_controls',
            study_arm=ARM, model_assignment=assignment,
            training_seed=None if entry is None else entry['seed'],
            training_condition=None if entry is None else entry['condition'],
            reference_training_condition=None,
            frozen_model_state_sha256=None if entry is None else entry['model_state_sha256'],
            prediction_head=None if entry is None else entry['prediction_head'],
            closed_loop_motion_residual_fit_sha256=None if entry is None else entry['fit_sha256'],
            motion_residual_correction_root=None if entry is None else entry['root_name'],
            correction_base_model=None if entry is None else assignment,
            frozen_model_registry_sha256=registry_identity(variant)[1],
            model_input_variant=variant,
            planned_conditions=list(ARMS), planned_layout_indices=[0,1], planned_layout_count=2,
            planned_native_assignments=22, fixed_dispatch_pairs=PAIRS,
            fresh_layout_inventory=layouts.build_inventory(), frozen_layout_inventory_sha256=INVENTORY_SHA256,
            layout_novelty_scope='distinct_from_explicit_76_layout_development_registry',
            tracker='BatchedConsensusMotion', batched_gyro_proposal_scatter=True,
            registration='ReacquiringFloorRegistration', floor_reacquisition_enabled=True,
            temporary_floor_conflict_is_missing_observation=True,
            consecutive_accepted_poses_before_resuming_planning=4,
            rejected_floor_pose_used_by_map_or_mission=False,
            floor_rejection_resets_arrival_dwell=True, floor_rejection_cancels_pending_commands=True,
            global_budget_continues_during_missingness=True,
            floor_candidate_and_acceptance_thresholds_unchanged=True,
            actual_runtime_class='ReactiveRuntime' if entry is None else 'SeededPredictiveRuntime',
            training_method_effect_includes_fixed_correction_procedure=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/multiseed_navigation_layouts_development.py',
                    'lewm/seeded_motion_correction_development.py',
                    'lewm/floor_reacquisition_development.py',
                    'lewm/batched_gyro_consensus_development.py',
                    'lewm/batched_consensus_tracking_development.py',
                    'scripts/run_go2_batched_consensus_followup_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    treatment = ARM if ARM in ('pose_command','reactive') else 'learned'
    bind(transfer.annotate, TREATMENT=treatment,
        RAW_WRITE=bind(finish, ARM=ARM, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(2), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args(); i = args.layout_index
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[i % 2]:
        raise ValueError('assigned layout CPU group required')
    if (hashlib.sha256(INVENTORY.read_bytes()).hexdigest() != INVENTORY_SHA256
            or json.loads(INVENTORY.read_text()) != layouts.build_inventory()):
        raise ValueError('fixed two-layout inventory required')
    assignment = assignment_for(args.arm)
    variant = 'no_rgb' if '_no_rgb_' in assignment else 'full'
    entry = None if assignment == 'reactive' else registry(variant)[assignment]
    condition = 'reactive' if entry is None else entry['condition']
    output = cohort.stable.source.BASE / ROOT.format(index=i, arm=args.arm)
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve every fixed assignment')
    writer = bind(cohort.make_writer, annotate=bind(annotate, ARM=args.arm))(output,i,condition)
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(model, **kwargs):
            if kwargs.get('condition') != condition: raise ValueError('fixed training condition required')
            extra = dict(registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02)
            if entry is None:
                return ReactiveRuntime(model, **extra, **kwargs)
            return SeededPredictiveRuntime(model, training_seed=entry['seed'],
                motion_prediction_source='pose_command' if args.arm=='pose_command' else 'learned',
                **extra, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=i,
            specification=layouts.specification, public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT=assignment,
            PacedNativeSession=partial(FreshCameraSession, noise_layout_index=i, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles, OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()

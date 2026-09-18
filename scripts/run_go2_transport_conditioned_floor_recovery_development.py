"""Fixed supervised/reactive tests of consumer-conditioned floor candidates."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_auxiliary_turn_recovery_development as previous

study = previous.study
ARMS = ('seed_2026091402_full_supervised_rollout', 'reactive')
ROOT = 'go2_transport_conditioned_floor_recovery_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def initialize_registration():
    from lewm.floor_reacquisition_development import initialize_registration as original
    from lewm import process_registered_round_trip_development as process
    from lewm.transport_conditioned_partial_floor_development import TransportConditionedReacquiringRegistration
    original()
    process._registration = TransportConditionedReacquiringRegistration()


def initialize_obstacles():
    from lewm import independent_depth_process_development as process
    from lewm.gyro_conditioned_partial_floor_consumers_development import GyroConditionedAuxiliaryObstacles
    previous.initialize_obstacles()
    process._observer = GyroConditionedAuxiliaryObstacles()


def finish(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='transport_conditioned_floor_recovery_v1',
            comparison='weak_floor_candidates_conditioned_on_each_consumer_normal',
            reference_root_name=previous.ROOT.format(index=1, arm=value['study_arm']),
            planned_conditions=list(ARMS), fixed_sequential_assignments=[[1,a] for a in ARMS],
            registration='TransportConditionedReacquiringRegistration',
            independent_obstacle_observer='GyroConditionedAuxiliaryObstacles',
            original_floor_acceptance_and_reacquisition_unchanged=False,
            floor_candidate_selection_changed=True, floor_acceptance_thresholds_unchanged=True,
            floor_reacquisition_runtime_and_command_guards_unchanged=True,
            raw_tracker_changed=False, tracker='BatchedConsensusMotion',
            pose_candidates_use_transported_reference_normal=True,
            obstacle_candidates_use_existing_gyro_normal=True,
            only_original_weak_extent_candidates_pruned=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/gyro_conditioned_partial_floor_candidates_development.py',
                    'lewm/gyro_conditioned_partial_floor_consumers_development.py',
                    'lewm/transport_conditioned_partial_floor_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(previous.annotate, ARM=ARM, RAW_WRITE=bind(finish, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(1,), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    base = study.cohort.stable.source.BASE
    reference = base/study.ROOT.format(index=1, arm=ARMS[0])
    probe = json.loads((reference/'transport_conditioned_partial_floor_consumers_replay_820_v1/effective_availability_diagnostic_v1.json').read_text())
    if not probe['new_jointly_available_frames_with_four_pose_streak'] or probe['failure'] is not None:
        raise ValueError('usable sequential floor-recovery evidence required')
    if args.arm == ARMS[1] and not (base/ROOT.format(index=1, arm=ARMS[0])/'continuous_native_arrival_evaluation.json').is_file():
        raise ValueError('complete and evaluate fixed supervised follow-up first')
    gyro = SimpleNamespace(**(vars(study.cohort.gyro) | dict(initialize_obstacles=initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(study.cohort) | dict(gyro=gyro)))
    bind(study.main, ROOT=ROOT, annotate=annotate, cohort=cohort,
        initialize_registration=initialize_registration,
        SeededPredictiveRuntime=previous.LearnedRecoveryRuntime,
        ReactiveRuntime=previous.ReactiveRecoveryRuntime)()


if __name__ == '__main__': main()

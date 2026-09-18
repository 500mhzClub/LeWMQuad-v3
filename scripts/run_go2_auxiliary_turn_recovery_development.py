"""Two fixed follow-ups to primary-depth blindness, after the 22-run comparison."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.auxiliary_only_turn_recovery_development import AuxiliaryTurnDispatch, initialize_obstacles
from scripts import run_go2_multiseed_navigation_development as study

ARMS = ('reactive', 'seed_2026091402_full_supervised_rollout')
ROOT = 'go2_auxiliary_only_turn_recovery_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class RecoveryViewMixin:
    def request(self, *, now_ns):
        result = super().request(now_ns=now_ns)
        if result['reason'] == 'PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO':
            with self.lock:
                if self.view_recovery is None:
                    self.view_recovery = dict(trigger_ns=now_ns,
                        mission_generation=self.mission_generation, target_heading_rad=None,
                        source='primary_depth_unavailable_translation_veto')
                result = result | dict(view_recovery=dict(self.view_recovery))
        return result


class LearnedRecoveryRuntime(RecoveryViewMixin, study.SeededPredictiveRuntime, AuxiliaryTurnDispatch):
    pass


class ReactiveRecoveryRuntime(RecoveryViewMixin, study.ReactiveRuntime, AuxiliaryTurnDispatch):
    pass


def mark(name, value):
    if name == 'launch.json':
        arm = value['study_arm']; index = value['layout_index']
        value = value | dict(experiment='auxiliary_only_turn_recovery_development_v1',
            comparison='current_auxiliary_obstacles_allow_turns_during_primary_depth_blindness',
            reference_root_name=study.ROOT.format(index=index, arm=arm),
            planned_conditions=list(ARMS), planned_layout_indices=[1], planned_layout_count=1,
            planned_native_assignments=2, fixed_dispatch_pairs=None,
            fixed_sequential_assignments=[[1, a] for a in ARMS],
            actual_runtime_class='ReactiveRecoveryRuntime' if arm == 'reactive' else 'LearnedRecoveryRuntime',
            independent_obstacle_observer='AuxiliaryTurnObstacles',
            auxiliary_only_turn_recovery=True, translation_requires_both_cameras=True,
            primary_depth_translation_veto_requests_new_view=True,
            original_floor_acceptance_and_reacquisition_unchanged=True,
            nominal_disk_and_observation_age_guards_unchanged=True,
            repeated_exposed_development_maze=True, new_independent_development_layout=False,
            layout_novelty_scope='repeat_of_exposed_multiseed_layout_1',
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/auxiliary_only_turn_recovery_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(study.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(1,), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    base = study.cohort.stable.source.BASE
    if not (base/'go2_multiseed_navigation_complete_comparison_v1_attempt_001/result.json').is_file():
        raise ValueError('finish and compare all 22 fixed assignments before the follow-ups')
    reference = base/study.ROOT.format(index=1, arm=args.arm)
    if not (reference/'auxiliary_only_turn_guard_probe_v1/result.json').is_file():
        raise ValueError('saved failure guard probe required')
    if args.arm == ARMS[1]:
        first = base/ROOT.format(index=1, arm=ARMS[0])
        if not (first/'continuous_native_arrival_evaluation.json').is_file():
            raise ValueError('complete and evaluate the first fixed follow-up')
    gyro = SimpleNamespace(**(vars(study.cohort.gyro) | dict(initialize_obstacles=initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(study.cohort) | dict(gyro=gyro)))
    bind(study.main, ROOT=ROOT, cohort=cohort, annotate=annotate,
        SeededPredictiveRuntime=LearnedRecoveryRuntime, ReactiveRuntime=ReactiveRecoveryRuntime)()


if __name__ == '__main__': main()

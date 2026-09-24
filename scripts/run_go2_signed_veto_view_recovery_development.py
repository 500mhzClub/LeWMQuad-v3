"""One reactive test changing only the actual translation-veto view direction."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.signed_veto_view_recovery_development import SignedVetoViewMixin
from scripts import run_go2_transport_conditioned_floor_recovery_development as previous

ARMS = ('reactive',)
ROOT = 'go2_signed_veto_view_recovery_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class SignedRecoveryRuntime(SignedVetoViewMixin, previous.previous.ReactiveRecoveryRuntime):
    pass


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='signed_veto_view_recovery_v1',
            comparison='vetoed_arc_direction_for_measured_view_recovery',
            reference_root_name=previous.ROOT.format(index=1, arm='reactive'),
            planned_conditions=list(ARMS), planned_native_assignments=1,
            fixed_sequential_assignments=[[1, 'reactive']],
            actual_runtime_class='SignedRecoveryRuntime',
            vetoed_arc_view_direction=True, view_recovery_angle_degrees=45,
            view_recovery_completion_tolerance_rad=.1,
            straight_veto_and_primary_blind_view_direction_unchanged=True,
            floor_consumers_unchanged_from_reference=True,
            raw_tracker_changed=False, tracker='BatchedConsensusMotion',
            command_guards_unchanged=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/signed_veto_view_recovery_development.py',
                    'lewm/veto_view_round_trip_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(previous.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    if sys.argv[1:] != ['--layout-index', '1', '--arm', 'reactive']:
        raise ValueError('fixed reactive layout-1 follow-up required')
    study = previous.study
    reference = study.cohort.stable.source.BASE / previous.ROOT.format(index=1, arm='reactive')
    if not (reference/'auxiliary_turn_recovery_evaluation_v1.json').is_file():
        raise ValueError('evaluate completed reactive floor-consumer reference first')
    gyro = SimpleNamespace(**(vars(study.cohort.gyro) | dict(initialize_obstacles=previous.initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(study.cohort) | dict(gyro=gyro)))
    bind(study.main, ROOT=ROOT, annotate=annotate, cohort=cohort,
        initialize_registration=previous.initialize_registration,
        ReactiveRuntime=SignedRecoveryRuntime)()


if __name__ == '__main__': main()

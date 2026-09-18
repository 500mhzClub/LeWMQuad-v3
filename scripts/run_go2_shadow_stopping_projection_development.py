"""Two exposed-maze followups disabling only planned stopping enforcement."""
import argparse
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.shadow_stopping_projection_development import ShadowStoppingProjectionMixin
from scripts import run_go2_shared_recovery_transfer_development as reference

study = reference.study
ARMS = ('seed_2026091001_full_supervised_rollout',)
ASSIGNMENTS = ((0, ARMS[0]), (1, ARMS[0]))
ROOT = 'go2_shadow_stopping_projection_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class ShadowStoppingRuntime(ShadowStoppingProjectionMixin, reference.SignedLearnedRuntime):
    pass


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='shadow_stopping_projection_v1',
            comparison='same_frozen_controller_planned_stopping_enforcement_off',
            reference_root_name=reference.ROOT.format(index=INDEX, arm=value['study_arm']),
            planned_conditions=list(ARMS), planned_native_assignments=2,
            fixed_sequential_assignments=list(ASSIGNMENTS),
            new_independent_development_layout=False, exposed_development_layout=True,
            repeated_exposed_development_maze=True,
            layout_novelty_scope='repeated_shared_recovery_development_layouts',
            actual_runtime_class='ShadowStoppingRuntime', raw_tracker_changed=False,
            planned_stopping_projection_enforced=False,
            planned_stopping_projection_shadow_computed=True,
            other_predictive_selection_unchanged=True,
            full_online_rollout_ablation=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/shadow_stopping_projection_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(reference.annotate, ARM=ARM,
        RAW_WRITE=bind(mark, INDEX=INDEX, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    base = study.cohort.stable.source.BASE
    preceding = ASSIGNMENTS[:ASSIGNMENTS.index((args.layout_index,args.arm))]
    for index, arm in preceding:
        if not (base/ROOT.format(index=index,arm=arm)/'continuous_native_arrival_evaluation.json').is_file():
            raise ValueError('complete and evaluate the preceding fixed assignment')
    if shutil.disk_usage(base).free < 4 * 1024**3:
        raise ValueError('four GiB free required for one full-budget recording')
    gyro = SimpleNamespace(**(vars(study.cohort.gyro) | dict(
        initialize_obstacles=reference.previous.initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(study.cohort) | dict(gyro=gyro)))
    bind(study.main, ROOT=ROOT, ARMS=ARMS, layouts=reference.layouts,
        INVENTORY=reference.INVENTORY, INVENTORY_SHA256=reference.INVENTORY_SHA256,
        FreshCameraSession=reference.FreshCameraSession,
        annotate=bind(annotate, INDEX=args.layout_index), cohort=cohort,
        initialize_pose=reference.initialize_pose,
        initialize_registration=reference.previous.initialize_registration,
        SeededPredictiveRuntime=ShadowStoppingRuntime)()


if __name__ == '__main__': main()

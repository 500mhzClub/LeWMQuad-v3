"""Fixed fresh-maze full/no-RGB comparison, using the shared predictive runtime."""
import argparse
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.neural_input_treatment_development import NeuralInputTreatmentMixin
from lewm import neural_rgb_transfer_layouts_development as layouts
from scripts import run_go2_shared_recovery_transfer_development as reference

study = reference.study
SEEDS = (2026091001, 2026091401, 2026091402)
METHODS = ('jepa', 'direct', 'supervised_rollout')
ARMS = tuple(f'seed_{s}_{v}_{m}' for s in SEEDS for m in METHODS for v in ('full','no_rgb'))
ASSIGNMENTS = tuple((i, f'seed_{s}_{v}_{m}') for s in SEEDS for m in METHODS
    for i in (0,1) for v in (('full','no_rgb') if i == 0 else ('no_rgb','full')))
ROOT = 'go2_neural_rgb_transfer_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
INVENTORY = Path('docs/go2_neural_rgb_transfer_layout_inventory_2026-09-15.json')
INVENTORY_SHA256 = '6b572ea6cd183ec2fa04b4b345db7a1cfb6841a6d893091c388e4a46e13f105f'


class FreshPhysicalInit(study.cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(study.cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(study.cohort.LiveDepthNoiseMixin, study.cohort.CompactDepthRetentionMixin,
        study.cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class InputCheckedRuntime(NeuralInputTreatmentMixin, reference.SignedLearnedRuntime):
    pass


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='neural_rgb_transfer_v1',
            comparison='matched_full_no_rgb_three_methods_three_seeds',
            planned_conditions=list(ARMS), planned_native_assignments=len(ASSIGNMENTS),
            fixed_sequential_assignments=list(ASSIGNMENTS),
            fresh_layout_inventory=layouts.build_inventory(), frozen_layout_inventory_sha256=INVENTORY_SHA256,
            layout_novelty_scope='distinct_from_explicit_80_layout_development_registry',
            actual_runtime_class='InputCheckedRuntime', raw_tracker_changed=False,
            neural_rgb_input_checked_at_every_forward=True,
            input_treatment_includes_matched_training_and_motion_correction=True,
            camera_based_tracking_and_mapping_retained=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/neural_rgb_transfer_layouts_development.py',
                    'lewm/neural_input_treatment_development.py',
                    'lewm/seeded_motion_correction_development.py',
                    'lewm/matched_motion_residual_runtime_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(reference.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    base = study.cohort.stable.source.BASE
    position = ASSIGNMENTS.index((args.layout_index,args.arm))
    if position:
        index, arm = ASSIGNMENTS[position-1]
        if not (base/ROOT.format(index=index,arm=arm)/'continuous_native_arrival_evaluation.json').is_file():
            raise ValueError('complete and evaluate the preceding fixed assignment')
    if shutil.disk_usage(base).free < 4*1024**3:
        raise ValueError('four GiB free required for one full-budget recording')
    gyro = SimpleNamespace(**(vars(study.cohort.gyro) | dict(initialize_obstacles=reference.previous.initialize_obstacles)))
    cohort = SimpleNamespace(**(vars(study.cohort) | dict(gyro=gyro)))
    bind(study.main, ROOT=ROOT, ARMS=ARMS, layouts=layouts, INVENTORY=INVENTORY,
        INVENTORY_SHA256=INVENTORY_SHA256, FreshCameraSession=FreshCameraSession,
        annotate=annotate, cohort=cohort, initialize_pose=reference.initialize_pose,
        initialize_registration=reference.previous.initialize_registration,
        SeededPredictiveRuntime=InputCheckedRuntime)()


if __name__ == '__main__':
    main()

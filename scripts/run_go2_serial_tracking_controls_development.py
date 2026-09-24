"""Two fixed original-tracker controls for the successful cache follow-ups."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_multiseed_navigation_development as study
from scripts import evaluate_go2_multiseed_navigation_development as evaluation

ARMS = ('seed_2026091402_full_direct', 'seed_2026091401_full_jepa')
ROOT = 'go2_serial_original_tracker_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='serial_original_tracker_controls_v1',
            comparison='original_tracker_without_second_native_simulation',
            reference_root_name=study.ROOT.format(index=1, arm=ARM),
            cached_reference_root_name=f'go2_cached_floor_moments_{ARM}_noise_2mm_native_layout01_4800_v1_attempt_001',
            planned_conditions=list(ARMS), planned_layout_indices=[1], planned_layout_count=1,
            planned_native_assignments=2, fixed_dispatch_pairs=None,
            fixed_sequential_assignments=[[1, arm] for arm in ARMS],
            tracker='BatchedConsensusMotion', auxiliary_only_turn_recovery=False,
            floor_moments_cache_enabled=False, original_comparison_failure_replaced=False,
            new_independent_development_layout=False, exposed_development_layout=True,
            layout_novelty_scope='repeat_of_exposed_multiseed_layout_1',
            other_native_simulation_planned=False,
            extra_sources=value['extra_sources'] | {
                __file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(mark, ARM=ARM, RAW_WRITE=RAW_WRITE)
    bind(study.annotate, ARM=ARM, RAW_WRITE=emit)(name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(1,), required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        source = SimpleNamespace(**(vars(evaluation.study) | dict(ROOT=ROOT)))
        bind(evaluation.evaluate, study=source)(1, args.arm)
    else:
        bind(study.main, ROOT=ROOT, annotate=annotate)()


if __name__ == '__main__': main()

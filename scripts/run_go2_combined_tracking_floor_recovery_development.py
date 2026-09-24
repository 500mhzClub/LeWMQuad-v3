"""One supervised tracking follow-up after the fixed floor-consumer experiment."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_transport_conditioned_floor_recovery_development as previous

ARM = 'seed_2026091402_full_supervised_rollout'
ARMS = (ARM,)
ROOT = 'go2_combined_tracking_floor_recovery_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.cached_moments_deferred_copy_tracking_development import CachedMomentsDeferredCopyMotion
    previous.study.cohort.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = CachedMomentsDeferredCopyMotion()


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='combined_tracking_floor_recovery_v1',
            comparison='cached_floor_moments_and_deferred_copy_with_same_floor_consumers',
            reference_root_name=previous.ROOT.format(index=1, arm=ARM),
            planned_conditions=list(ARMS), planned_native_assignments=1,
            fixed_sequential_assignments=[[1,ARM]],
            tracker='CachedMomentsDeferredCopyMotion', raw_tracker_changed=True,
            floor_consumers_unchanged_from_reference=True, intended_tracker_change_only=True,
            reference_policy_changed=False, pose_acceptance_thresholds_changed=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/cached_moments_deferred_copy_tracking_development.py',
                    'lewm/cached_floor_moments_development.py',
                    'lewm/deferred_registration_copy_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(previous.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    if sys.argv[1:] != ['--layout-index','1','--arm',ARM]:
        raise ValueError('fixed supervised-seed-1402 layout-1 tracking follow-up required')
    base = previous.study.cohort.stable.source.BASE
    reactive = base/previous.ROOT.format(index=1,arm='reactive')
    if not (reactive/'continuous_native_arrival_evaluation.json').is_file():
        raise ValueError('complete and evaluate both fixed floor-consumer attempts first')
    reference = base/previous.ROOT.format(index=1,arm=ARM)
    comparison = json.loads((reference/'gyro_coherent_floor_cached_moments_deferred_copy_replay_v1/exact_match_and_cost_comparison_v1.json').read_text())
    if not comparison['all_replay_estimates_and_receipts_exact']:
        raise ValueError('combined tracker must preserve the recorded estimator results')
    study = SimpleNamespace(**(vars(previous.study) | dict(
        main=bind(previous.study.main, initialize_pose=initialize_pose))))
    bind(previous.main, ROOT=ROOT, study=study, annotate=annotate)()


if __name__ == '__main__': main()

"""One exposed-maze tracking-throughput follow-up; preserve original failure."""
import hashlib
from pathlib import Path
import sys
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_multiseed_navigation_development as study

ARM = 'seed_2026091402_full_direct'
ROOT = 'go2_cached_floor_moments_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.cached_floor_moments_development import CachedFloorMomentsMotion
    study.cohort.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = CachedFloorMomentsMotion()


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='cached_floor_moments_tracking_followup_v1',
            comparison='reuse_raw_floor_cloud_statistics_with_original_pair_fit_acceptance',
            reference_root_name=study.ROOT.format(index=1, arm=ARM),
            planned_conditions=[ARM], planned_layout_indices=[1], planned_layout_count=1,
            planned_native_assignments=1, fixed_dispatch_pairs=None,
            tracker='CachedFloorMomentsMotion', raw_floor_moments_cache_capacity=32,
            reference_policy_changed=False, pose_acceptance_thresholds_changed=False,
            auxiliary_only_turn_recovery=False, original_comparison_failure_replaced=False,
            new_independent_development_layout=False, exposed_development_layout=True,
            layout_novelty_scope='repeat_of_exposed_multiseed_layout_1',
            extra_sources=value['extra_sources'] | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/cached_floor_moments_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(study.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    if sys.argv[1:] != ['--layout-index', '1', '--arm', ARM]:
        raise ValueError('fixed direct-seed-1402 layout-1 tracking follow-up required')
    bind(study.main, ROOT=ROOT, annotate=annotate, initialize_pose=initialize_pose)()


if __name__ == '__main__': main()

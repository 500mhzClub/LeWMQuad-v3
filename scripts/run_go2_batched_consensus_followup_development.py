"""Exposed-maze follow-up of the recorded return-phase tracking overflow."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_stopping_projection_transfer_development as transfer

ROOT = 'go2_batched_consensus_pose_command_noise_2mm_native_layout02_4800_v1_attempt_001'
REFERENCE = transfer.ROOT.format(index=2, condition='pose_command')


def initialize_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    from lewm.batched_consensus_tracking_development import BatchedConsensusMotion
    transfer.cohort.stable.floor.configure()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = BatchedConsensusMotion()


def annotate_followup(name, value):
    if name == 'launch.json':
        if value['layout_index'] != 2 or value['comparison_condition'] != 'pose_command':
            raise ValueError('fixed fitted-motion layout-2 follow-up required')
        value = value | dict(experiment='batched_consensus_tracking_followup_v1',
            comparison='batched_gyro_proposals_with_same_reference_policy_and_acceptance',
            reference_root_name=REFERENCE, planned_conditions=['pose_command'],
            planned_layout_indices=[2], planned_layout_count=1, planned_native_assignments=1,
            fixed_dispatch_pairs=None, new_independent_development_layout=False,
            exposed_development_layout=True, layout_novelty_scope='previously_exposed_transfer_layout_2',
            tracker='BatchedConsensusMotion', batched_gyro_proposal_scatter=True,
            reference_policy_changed=False, pose_acceptance_thresholds_changed=False,
            original_comparison_failure_replaced=False,
            extra_sources=value['extra_sources'] | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/batched_gyro_consensus_development.py',
                    'lewm/batched_consensus_tracking_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(annotate_followup, RAW_WRITE=RAW_WRITE)
    bind(transfer.annotate, TREATMENT='pose_command', RAW_WRITE=emit)(name, value)


def main():
    import sys
    if sys.argv[1:] != ['--layout-index', '2', '--condition', 'pose_command']:
        raise ValueError('use --layout-index 2 --condition pose_command')
    # Bind a private initializer; the completed cohort and other launchers keep
    # their original tracker and source. All controller settings are inherited.
    stopping = SimpleNamespace(StoppingAwareRuntime=transfer.stopping.StoppingAwareRuntime,
        previous=SimpleNamespace(initialize_pose=initialize_pose))
    bind(transfer.main, ROOT=ROOT, annotate=annotate, stopping=stopping)()


if __name__ == '__main__':
    main()

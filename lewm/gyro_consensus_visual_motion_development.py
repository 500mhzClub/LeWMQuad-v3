"""Current gyro-conditioned RGB-D pose for experimental registered navigation."""
from lewm.pair_local_plane_consensus_development import PairLocalPlaneConsensusVisualMotion
from lewm.gyro_consensus_pair_pose_development import GyroConsensusPairPose


class GyroConsensusVisualMotion(PairLocalPlaneConsensusVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = GyroConsensusPairPose()

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            observer_variant='gyro_consensus_dual_camera_v1',
            gyro_conditioned_image_consensus_estimator=True,
            measured_plane_constrained_estimator=False,
            original_pair_acceptance_checks_unchanged=False,
            floor_plane_conflict_rejection_scope='image_reference_pair',
            gyro_bias_estimated=False, gyro_noise_model_validated=False,
            translation_requires_current_rgbd_correspondences=True)


def initialize_pose():
    import cv2
    import torch
    from lewm.two_cm_floor_extent_development import configure
    from lewm import process_mapped_runtime_development as process
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = GyroConsensusVisualMotion()

"""Gyro-consensus tracking with identical chained image-link reuse."""
from lewm.chained_flow_memo_development import ChainedFlowMemo
from lewm.gyro_consensus_pair_pose_development import GyroConsensusPairPose
from lewm.gyro_consensus_visual_motion_development import GyroConsensusVisualMotion


class ReusedFlowGyroPose(GyroConsensusPairPose):
    def __init__(self):
        super().__init__()
        self.flow_memo = ChainedFlowMemo()

    def observe(self, *args, **kwargs):
        with self.flow_memo.observation():
            return super().observe(*args, **kwargs)


class ReusedFlowGyroVisualMotion(GyroConsensusVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = ReusedFlowGyroPose()


def initialize_pose():
    import cv2
    import torch
    from lewm.two_cm_floor_extent_development import configure
    from lewm import process_mapped_runtime_development as process
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = ReusedFlowGyroVisualMotion()

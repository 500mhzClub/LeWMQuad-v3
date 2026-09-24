"""Remove accumulated matrix roundoff from the integrated gyro rotation."""
import numpy as np
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.reused_flow_gyro_visual_motion_development import ReusedFlowGyroPose, ReusedFlowGyroVisualMotion


def orthonormalize(rotation):
    u, _, vt = np.linalg.svd(rotation)
    return u @ vt


class OrthonormalFastOrientation(FastRelativeOrientation):
    def step(self, *args, **kwargs):
        # The original integration and finite/proper-rotation checks run first.
        super().step(*args, **kwargs)
        self.rotation = orthonormalize(self.rotation)
        return self._result()


class OrthonormalGyroPose(ReusedFlowGyroPose):
    def __init__(self):
        super().__init__()
        self.gyro = OrthonormalFastOrientation()


class OrthonormalGyroVisualMotion(ReusedFlowGyroVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = OrthonormalGyroPose()

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(gyro_rotation_reorthogonalized_each_camera_interval=True)


def initialize_pose():
    import cv2
    import torch
    from lewm.two_cm_floor_extent_development import configure
    from lewm import process_mapped_runtime_development as process
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    process._motion = OrthonormalGyroVisualMotion()

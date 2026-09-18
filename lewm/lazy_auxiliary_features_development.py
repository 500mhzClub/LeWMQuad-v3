"""Defer auxiliary corner descriptors until a pose fit actually needs them.

Own the measured pixels immediately; plane fitting and image-chain caching can
read depth and grayscale without computing descriptors. A later fallback sees
the same saved pixels. Deferred work can increase fallback latency, which must
be measured separately from the common primary-camera path.
"""
from copy import deepcopy
from types import FunctionType

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.full_consensus_tracker_development import FullConsensusPose, FullConsensusVisualMotion


class LazyCornerSupportFeatureFrame:
    def __init__(self, rgb, depth):
        if not isinstance(rgb, np.ndarray) or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise SensorContractError('exact current RGB frame required')
        self.rgb = rgb.copy(); self.depth = deepcopy(depth)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
        self._computed = None

    def _features(self):
        if self._computed is None:
            self._computed = CornerSupportFeatureFrame(self.rgb, self.depth)
        return self._computed

    @property
    def keypoints(self): return self._features().keypoints

    @property
    def descriptors(self): return self._features().descriptors

    def witness(self): return self._features().witness()


class _LazyAuxiliaryDual(DualCameraAnchorPose):
    def observe(self, policy, depth, fast, *, auxiliary_rgb, auxiliary_depth, now_ns):
        def feature_frame(rgb, camera_depth):
            cls = LazyCornerSupportFeatureFrame if camera_depth is auxiliary_depth else CornerSupportFeatureFrame
            return cls(rgb, camera_depth)
        original = DualCameraAnchorPose.observe
        function = FunctionType(original.__code__, original.__globals__ |
            dict(CornerSupportFeatureFrame=feature_frame), original.__name__,
            original.__defaults__, original.__closure__)
        return function(self, policy, depth, fast, auxiliary_rgb=auxiliary_rgb,
            auxiliary_depth=auxiliary_depth, now_ns=now_ns)


class LazyAuxiliaryPose(FullConsensusPose, _LazyAuxiliaryDual):
    pass


class LazyAuxiliaryVisualMotion(FullConsensusVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = LazyAuxiliaryPose()

"""Empty-state map optimization with unchanged dual-camera controller behavior."""
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.single_pass_later_floor_controller_development import SinglePassLaterFloorMap


class SinglePassDualCameraController(DualCameraSettledController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = SinglePassLaterFloorMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface

"""Existing measured tracking fallback inside the original contact-scoring policy."""
from lewm.commitment_contact_anchored_controller_development import CommitmentContactAnchoredController
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion

CONTROLLER = 'direct_flow_commitment_contact_anchored_controller_v1'
FLAG = 'direct_corner_flow_missingness_fallback_enabled'


class DirectFlowCommitmentContactController(CommitmentContactAnchoredController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = DirectFlowDualCameraVisualMotion(identity=(0,0,0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | {'controller':CONTROLLER, FLAG:True}

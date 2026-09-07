"""Frozen-response comparison on B; predictions never select commands."""
from copy import deepcopy

from lewm.action_motion_identification_development import MotionIdentificationController, command_schedule
from lewm.action_response_model_development import model_identity, position_persistence, predict_response
from lewm.causal_sensor_state import SensorContractError


class MotionValidationController(MotionIdentificationController):
    def __init__(self, owner, model, identity):
        if model_identity(model) != identity:
            raise SensorContractError('frozen response model identity required')
        super().__init__(owner)
        self.schedule = command_schedule('validation')
        self.model = deepcopy(model)
        self.model_sha256 = identity

    def observe(self, policy, depth, fast, *, now_ns):
        row = super().observe(policy, depth, fast, now_ns=now_ns)
        row['position_persistence_prediction'] = None
        row['response_prediction'] = None
        if row['prediction'] is not None:
            future = (self.schedule + [[0., 0., 0.]] * 4)[row['motion_index']:row['motion_index'] + 4]
            row['position_persistence_prediction'] = position_persistence(row['prediction'])
            row['response_prediction'] = predict_response(
                self.owner, policy, future, self.model, now_ns=now_ns, model_sha256=self.model_sha256)
        if self.status == 'COMPLETE_IDENTIFICATION_SCHEDULE':
            self.status = 'COMPLETE_VALIDATION_SCHEDULE'
            row['status'] = self.status
        self.last = deepcopy(row)
        return row

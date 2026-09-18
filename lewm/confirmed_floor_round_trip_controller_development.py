"""View-recovery controller with additional observed auxiliary floor confirmation."""
from lewm.view_reentry_round_trip_controller_development import ViewReentryRoundTripController
from lewm.confirmed_auxiliary_floor_memory_development import ConfirmedAuxiliaryFloorMap


class ConfirmedFloorRoundTripController(ViewReentryRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = ConfirmedAuxiliaryFloorMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='confirmed_floor_round_trip_controller_v1',
            current_primary_floor_confirmation_enabled=True)

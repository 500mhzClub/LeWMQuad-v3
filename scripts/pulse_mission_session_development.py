"""Distinct maze session for the measured pulse command domain.

Not launched or a new scene authorization: a new protocol/initializer must fix
the actual next scene. Native sampling, guards, gait and command slew remain
inherited. The old .35rad/s experiment and its validator are unchanged.
"""
from lewm.command_pulse_response_development import validate_command
from scripts.fresh_maze_session_development import MissionRGBDSession
from scripts.rgbd_session_development import RGBDSession


class PulseMissionRGBDSession(MissionRGBDSession):
    def command_tick(self, requested):
        # Deliberately enter the common acquisition/physics layer only after
        # validation against this new .20/.45 domain, not the old .20/.35 one.
        return RGBDSession.command_tick(self, validate_command(requested))

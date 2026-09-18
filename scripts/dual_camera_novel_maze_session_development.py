"""Add the already captured downward RGB to the existing paired sensor packet."""
from lewm.causal_auxiliary_rgb_observation_development import validate_rgb
from scripts.novel_maze_round_trip_session_development import NovelMazeRoundTripSession
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition


class DualCameraNovelMazeSession(NovelMazeRoundTripSession):
    def sensor_packets(self):
        policy, depth, fast, auxiliary, now = super().sensor_packets()
        index = len(self.model_manifest)-1
        image, _ = packet(self.output, index, policy, public_acquisition(self.auxiliary_audit[index]), now_ns=now)
        validate_rgb(image, auxiliary, policy, now_ns=now)
        return policy, depth, fast, auxiliary, image, now

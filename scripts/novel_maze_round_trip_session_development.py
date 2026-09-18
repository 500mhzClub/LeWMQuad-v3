"""Native prospective maze acquisition with bounded paired auxiliary frames."""
from lewm.novel_maze_round_trip_contract_development import MAX_OBSERVATIONS
from scripts.novel_maze_round_trip_physical_session_development import NovelMazeBaseSession
from scripts.auxiliary_downward45_depth_capture_development import capture
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.run_go2_successive_choice_maze_development_v1 import write_json


class NovelMazeRoundTripSession(NovelMazeBaseSession):
    def __init__(self, *args, **kwargs):
        self.auxiliary_audit = []
        super().__init__(*args, **kwargs)

    def sensor_packets(self):
        # Reject over-budget calls before primary acquisition can write a frame.
        next_index = (len(self.samples)-750)//50
        if not 0 <= next_index < MAX_OBSERVATIONS:
            raise ValueError('bounded prospective maze acquisition required')
        policy, depth, fast, now = super().sensor_packets()
        index = len(self.model_manifest)-1
        if index != next_index: raise ValueError('consecutive actual maze observations required')
        if len(self.auxiliary_audit) == index:
            row = capture(self, self.output, index)
            if row['physical_sample_index'] != 749+50*index or row['measured_ns'] != now:
                raise ValueError('exact primary/auxiliary physical pairing required')
            self.auxiliary_audit.append(row)
        if len(self.auxiliary_audit) != index+1:
            raise ValueError('uninterrupted auxiliary acquisition required')
        auxiliary = packet(self.output, index, policy, public_acquisition(self.auxiliary_audit[index]), now_ns=now)
        return policy, depth, fast, auxiliary, now

    def persist_observations(self, output):
        super().persist_observations(output)
        write_json(output/'auxiliary_camera_audit.json', self.auxiliary_audit)

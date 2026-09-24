"""Prospective paired acquisition with one auxiliary depth reconstruction.

No launcher or live-session replacement. Captures and public packet builders
are the existing implementations; only their session composition differs.
"""
from lewm.causal_auxiliary_rgb_observation_development import validate_rgb
from scripts.auxiliary_downward45_depth_capture_development import capture
from scripts.dual_camera_novel_maze_session_development import DualCameraNovelMazeSession
from scripts.novel_maze_round_trip_physical_session_development import NovelMazeBaseSession
from scripts.novel_maze_auxiliary_rgb_packet_development import public_acquisition
from scripts.renderer_witness_dual_camera_maze_session_development import RendererWitnessDualCameraMazeSession
from scripts import extended_budget_anchored_maze_development as extended

MAX_OBSERVATIONS = extended.MAX_OBSERVATIONS
rgb_packet = extended.rgb_packet


class SingleReadAuxiliaryDualSession(DualCameraNovelMazeSession):
    def sensor_packets(self):
        # Preserve the original guard before primary capture can write a frame.
        next_index = (len(self.samples)-750)//50
        if not 0 <= next_index < MAX_OBSERVATIONS:
            raise ValueError('bounded prospective maze acquisition required')
        policy, depth, fast, now = NovelMazeBaseSession.sensor_packets(self)
        index = len(self.model_manifest)-1
        if index != next_index:
            raise ValueError('consecutive actual maze observations required')
        if len(self.auxiliary_audit) == index:
            row = capture(self, self.output, index)
            if row['physical_sample_index'] != 749+50*index or row['measured_ns'] != now:
                raise ValueError('exact primary/auxiliary physical pairing required')
            self.auxiliary_audit.append(row)
        if len(self.auxiliary_audit) != index+1:
            raise ValueError('uninterrupted auxiliary acquisition required')
        image, auxiliary = rgb_packet(self.output, index, policy,
            public_acquisition(self.auxiliary_audit[index]), now_ns=now)
        validate_rgb(image, auxiliary, policy, now_ns=now)
        return policy, depth, fast, auxiliary, image, now


class SingleReadAuxiliaryRendererSession(
        RendererWitnessDualCameraMazeSession, SingleReadAuxiliaryDualSession):
    # The original renderer wrapper's super() enters the new paired method.
    # Keep its complete capture witnesses, failure latch and persistence.
    capture_fixed_rgb = extended.ExtendedBudgetRendererSession.capture_fixed_rgb
    sensor_packets = extended.ExtendedBudgetRendererSession.sensor_packets

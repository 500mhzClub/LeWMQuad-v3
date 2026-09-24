"""Continuous mission acquisition with a causal, non-actuating depth observer."""
import json

import numpy as np
from PIL import Image

from lewm.depth_relative_motion_development import DepthRelativeState
from scripts.single_sample_rgbd_session_development import SingleSampleRGBDSession


class MovingRGBDSession(SingleSampleRGBDSession):
    def __init__(self, spec, output):
        self.relative_state = DepthRelativeState()
        self.relative_observations = []
        super().__init__(spec, output)

    def capture_observation(self, output):
        super().capture_observation(output)
        index = len(self.packet_rows)-1
        now = int(self.packet_rows[index]['decision_ns'])
        if now % 100_000_000:
            # Native contact can terminate between policy ticks. Preserve its
            # raw RGBD without inventing a complete high-rate decision history.
            result = {'measured_ns': now, 'status': 'NON_DECISION_TERMINAL_CAPTURE'}
        else:
            with Image.open(output/f'rgb_{index:04d}.png') as image: pixels = np.array(image)
            policy = self.observations.packet(pixels, now)
            result = self.relative_state.observe(policy, self.latest_depth,
                self.fast_buffer.packet(now_ns=now), now_ns=now)
        self.relative_observations.append({'observation_index': index, 'observer': result})

    def persist_observations(self, output):
        super().persist_observations(output)
        (output/'relative_state_observations.json').write_text(
            json.dumps(self.relative_observations, indent=2, allow_nan=False)+'\n')

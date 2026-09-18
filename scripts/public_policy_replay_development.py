"""Read retained RGB/body policy observations without requiring depth archives."""
from numbers import Integral
from pathlib import Path

from lewm.causal_rgb_dataset_development import _leaf, _protected
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.in_memory_public_replay_development import PublicReplay


class PublicPolicyReplay:
    """Policy inputs only; deliberately provides no full sensor packet method."""
    _archive = PublicReplay._archive

    def __init__(self, directory):
        directory = Path(directory).absolute()
        if _protected(directory) or _protected(directory.resolve()):
            raise ValueError('protected policy input forbidden')
        self.directory = directory.resolve()
        fields = {'image_ns', 'decision_ns'} | {
            f'{s.name}_{k}' for s in SCHEMAS
            for k in ('values', 'valid', 'measured_ns', 'available_ns')}
        self.body = self._archive('policy_histories.npz', fields)

    def policy_packet(self, frame):
        if isinstance(frame, bool) or not isinstance(frame, Integral) or not 0 <= frame < len(self.body['decision_ns']):
            raise ValueError('actual retained policy frame required')
        _leaf(self.directory, f'rgb_{frame:04d}.png')
        return PublicReplay.policy_packet(self, frame)

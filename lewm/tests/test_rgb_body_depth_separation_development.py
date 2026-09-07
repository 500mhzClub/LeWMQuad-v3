"""Tensor dependency test only; synthetic packets do not grant data eligibility."""
from functools import lru_cache
import torch
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.tests.test_terminal_event_coverage_development import fixture
from lewm.tests.test_independent_pulse_context_development import policy


class PolicyOnlyPacket:
    def __init__(self, policy_packet, depth, shadow):
        self.policy_packet, self.depth, self.shadow = policy_packet, depth, shadow

    def __getitem__(self, index):
        if index != 0:
            raise AssertionError('RGB/body learner attempted to read a non-policy packet element')
        return self.policy_packet


def assert_same_tree(left, right):
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_same_tree(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    else:
        assert left == right


def test_changed_depth_and_shadow_cannot_change_learner_inputs_or_target_join():
    data = fixture('contact')
    window, targets = data['window'], data['labels']
    dataset = PulseTimedDataset([window], [targets], {'synthetic': dict(layout_id='synthetic', role='train')})
    cached_policy = lru_cache(maxsize=None)(policy)
    class Reader:
        def __init__(self, depth, shadow): self.depth, self.shadow = depth, shadow
        def packet(self, i): return PolicyOnlyPacket(cached_policy(i), self.depth, self.shadow)
    first = dataset.sample(0, {'synthetic': Reader({'depth_m': 1., 'valid': True}, {'pose': [0., 0., 0.]})})
    other = dataset.sample(0, {'synthetic': Reader({'depth_m': float('nan'), 'valid': False}, {'failure': 'tracking lost'})})
    assert_same_tree(first, other)
    assert set(first['inputs']) == {'observation_history', 'known_action_blocks', 'known_action_valid'}
    assert set(first['inputs']['observation_history']) == {'rgb', 'body', 'control'}
    assert first['targets']['contact_valid'].any() and (first['targets']['contact'] == 1).any()

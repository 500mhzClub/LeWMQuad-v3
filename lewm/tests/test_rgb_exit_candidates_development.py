import math

import numpy as np
import pytest

from lewm.rgb_exit_candidates_development import candidates_from_floor_points, observe_exit_candidates, ExitCandidate
from lewm.gravity_feedback_ground_development import CausalGravityFeedbackGround
from lewm.memory.observed_exploration_development import ObservedExploration, PlaceFix
from lewm.tests.test_relative_gyro_turn_development import initialized, packet


def points_for(angles, ranges=(1.05, 1.15, 1.25, 1.35)):
    return np.array([[r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), -.3]
                     for a in angles for r in ranges])


def test_contiguous_multi_depth_angular_support_proposes_a_bearing_not_a_graph_edge():
    points = points_for((-5, -3, -1, 1, 3, 5))
    result = candidates_from_floor_points(points, np.ones(len(points), bool), timestamp_ns=10, observation_id='frame')
    assert len(result['candidates']) == 1
    candidate = result['candidates'][0]
    assert candidate.bearing_body_rad == pytest.approx(0)
    assert candidate.supported_angle_bins == 6 and candidate.support_points == 24
    assert not candidate.qualified_exit and not candidate.qualified_traversal
    memory = ObservedExploration()
    memory.observe_place(PlaceFix('fix', 10, 'start', True))
    with pytest.raises(ValueError, match='typed exit'):
        memory.observe_exit(candidate)
    assert not memory.exits


@pytest.mark.parametrize('angles,ranges', [((-3, -1, 1), (1.05, 1.15, 1.25)),
                                         ((-5, -3, -1, 1, 3, 5), (1.05, 1.15)),
                                         ((-5, -3, -1, 1, 3, 5), (.5, .6, .7, .8))])
def test_tiny_or_shallow_floor_patch_does_not_become_an_opening(angles, ranges):
    points = points_for(angles, ranges)
    result = candidates_from_floor_points(points, np.ones(len(points), bool), timestamp_ns=10, observation_id='frame')
    assert not result['candidates'] and not result['absence_means_closed']


def test_gaps_remain_unknown_instead_of_being_filled_into_a_single_proposal():
    points = points_for((-15, -13, -11, -9, 9, 11, 13, 15))
    result = candidates_from_floor_points(points, np.ones(len(points), bool), timestamp_ns=10, observation_id='frame')
    assert len(result['candidates']) == 2
    assert result['candidates'][0].angular_upper_rad < result['candidates'][1].angular_lower_rad


def test_empty_or_unobserved_points_are_unknown_not_free_or_closed():
    points = np.full((4, 3), np.nan)
    result = candidates_from_floor_points(points, np.zeros(4, bool), timestamp_ns=0, observation_id='empty')
    assert result['observed_point_count'] == 0 and not result['candidates']
    assert not result['absence_means_closed']
    with pytest.raises(ValueError):
        candidates_from_floor_points(points, np.ones(4, bool), timestamp_ns=0, observation_id='bad')


def test_actual_packet_interface_requires_palette_evidence_and_fresh_ground():
    p = packet(initialized(), 80)
    now = p['image']['measured_ns']
    ground = CausalGravityFeedbackGround('transported_feedback').begin(p, now_ns=now)
    p['image']['rgb'][:] = [115, 120, 108]
    result = observe_exit_candidates(p, ground, now_ns=now, observation_id='green')
    assert result['candidates'] and not result['metric_clearance_qualified']
    p['image']['rgb'][:] = 100
    result = observe_exit_candidates(p, ground, now_ns=now, observation_id='gray')
    assert not result['candidates'] and not result['absence_means_closed']
    with pytest.raises(ValueError):
        observe_exit_candidates(p, ground, now_ns=now + 1, observation_id='stale')


def test_candidate_cannot_assert_qualification_or_fabricated_support():
    with pytest.raises(ValueError):
        ExitCandidate('x', 0, 0., -.1, .1, 4, 12, 3, qualified_exit=True)
    with pytest.raises(ValueError):
        ExitCandidate('x', 0, 0., -.1, .1, 4, 1, 3)

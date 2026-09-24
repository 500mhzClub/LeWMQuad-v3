from threading import Lock
import numpy as np
import pytest
from lewm.auxiliary_only_turn_recovery_development import dispatch_request, auxiliary_only_obstacles
from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles
from lewm import fine_obstacle_round_trip_development as fine


def obstacle(*, cells=((60, 0),), counts=(0, 1000)):
    return CurrentObstacles(4, 1_900_000_000, (0., 0., 0.), tuple(map(tuple, np.eye(3))),
        frozenset(cells), counts, fine.FRAME)


def command(action):
    return ScheduledCommand.prepare(action, observed_ns=1_600_000_000,
        completed_ns=1_800_000_000, delay_ticks=3, commit_ticks=4)


def test_turn_allowed_but_forward_and_arcs_vetoed_with_auxiliary_only():
    for action in ('left_turn', 'right_turn'):
        result = dispatch_request(command(action), obstacle(), now_ns=1_900_000_000)
        assert result['requested_command'] == list(command(action).command)
        assert result['reason'] == 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'
        assert result['missing_primary_rays_inferred_free'] is False
    for action in ('forward', 'left_arc', 'right_arc'):
        result = dispatch_request(command(action), obstacle(), now_ns=1_900_000_000)
        assert result['requested_command'] == [0., 0., 0.]
        assert result['reason'] == 'PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO'


@pytest.mark.parametrize('current,now,reason', [
    (None, 1_900_000_000, 'CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'),
    (obstacle(), 2_120_000_000, 'CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'),
    (obstacle(cells=((40, 0),)), 1_900_000_000, 'CURRENT_OBSERVED_OBSTACLE_VETO'),
    (obstacle(), 2_300_000_000, 'OUTSIDE_COMMITTED_INTERVAL')])
def test_original_turn_vetoes_still_apply(current, now, reason):
    result = dispatch_request(command('left_turn'), current, now_ns=now)
    assert result['requested_command'] == [0., 0., 0.] and result['reason'] == reason


def test_paired_current_evidence_keeps_original_dispatch_result():
    current = obstacle(counts=(800, 1000))
    for action in ('hold', 'forward', 'left_turn', 'right_arc'):
        plan = command(action)
        assert dispatch_request(plan, current, now_ns=1_900_000_000) == fine.dispatch_request(
            plan, current, now_ns=1_900_000_000)


def test_missing_floor_remains_unavailable_and_old_plane_is_rejected():
    receipt = dict(measured_ns=1_900_000_000, joint_plane=dict(available=False))
    assert auxiliary_only_obstacles(None, None, None, receipt, now_ns=1_900_000_000) is None
    with pytest.raises(ValueError, match='same current'):
        auxiliary_only_obstacles(None, None, None, receipt, now_ns=2_000_000_000)


def test_primary_blind_translation_veto_requests_one_new_view():
    from scripts.run_go2_auxiliary_turn_recovery_development import RecoveryViewMixin
    class Base:
        def request(self, *, now_ns):
            return dict(reason='PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO',
                requested_command=[0., 0., 0.], view_recovery=None)
    class Runtime(RecoveryViewMixin, Base):
        pass
    runtime = Runtime()
    runtime.lock = Lock(); runtime.view_recovery = None; runtime.mission_generation = 2
    first = runtime.request(now_ns=100)
    assert first['view_recovery'] == dict(trigger_ns=100, mission_generation=2,
        target_heading_rad=None, source='primary_depth_unavailable_translation_veto')
    assert runtime.request(now_ns=120)['view_recovery'] == first['view_recovery']

from dataclasses import replace

import numpy as np

from lewm.auxiliary_only_turn_recovery_development import dispatch_request as original
from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.fresh_obstacle_dispatch_development import CurrentObstacles
from lewm.pipeline_age_dispatch_development import (
    PipelineAgeRuntime, PipelineAgeDispatch, dispatch_request)
from lewm.continuous_commitment_runtime_development import ContinuousCommitmentRuntime


def inputs(cells=()):
    plan = ScheduledCommand.prepare('forward', observed_ns=0, completed_ns=200_000_000,
        delay_ticks=3, commit_ticks=4)
    current = CurrentObstacles(1, 100_000_000, (0., 0., 0.),
        tuple(map(tuple, np.eye(3))), frozenset(cells), (1000, 1000),
        'current_body_1cm_grid')
    return plan, current


def test_gap_is_admitted_with_actual_age_in_stopping_projection():
    plan, current = inputs()
    assert original(plan, current, now_ns=320_000_000)['reason'] == 'CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'
    result = dispatch_request(plan, current, now_ns=320_000_000)
    assert result['reason'] == 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'
    assert result['observation_age_allowance_s'] == .22
    assert any(result['requested_command'])
    assert result['stopping_allowance_s'] == .5


def test_longer_delay_and_absent_observation_still_stop():
    plan, current = inputs()
    for value, now in ((current, 360_000_000), (None, 320_000_000)):
        result = dispatch_request(plan, value, now_ns=now)
        assert result['reason'] == 'CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE'
        assert not any(result['requested_command'])


def test_obstacle_and_missing_primary_still_veto_translation():
    plan, current = inputs(((50, 0),))
    assert not any(dispatch_request(plan, current, now_ns=320_000_000)['requested_command'])
    plan, current = inputs(((64, 0),))
    result = dispatch_request(plan, current, now_ns=320_000_000)
    assert result['reason'] == 'CURRENT_STOPPING_MARGIN_VETO'
    assert not any(result['requested_command'])
    plan, current = inputs()
    result = dispatch_request(plan, replace(current, valid_return_counts=(0, 1000)),
        now_ns=320_000_000)
    assert result['reason'] == 'PRIMARY_DEPTH_UNAVAILABLE_TRANSLATION_VETO'
    assert not any(result['requested_command'])


def test_runtime_retains_commitment_checks_before_new_dispatch():
    mro = PipelineAgeRuntime.__mro__
    assert mro.index(ContinuousCommitmentRuntime) < mro.index(PipelineAgeDispatch)
    assert PipelineAgeDispatch.request.__globals__['dispatch_request'] is dispatch_request

"""A complete negative learned outcome is admissible; live/failed work is not."""
from copy import deepcopy

import pytest

from scripts import nominal_measured_plane_native_inputs_development as inputs


def fixture(success=False):
    launch = dict(source_sha256={'fixed': 'source'})
    result = dict(status='MEASURED_PLANE_DISPATCH_RECOVERY_V1_COMPLETE',
        source_sha256=deepcopy(launch['source_sha256']), artifact_sha256={'launch.json': inputs.LEARNED_LAUNCH_SHA},
        conditions=[{'verified_round_trip': success}], automatic_retry=False,
        original_failed_waiters_preserved=True,
        controller_completion_sha256=inputs.learned.original.prefix.COMPLETION_SHA,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False,
        measured_round_trip_successes=int(success))
    return result, launch


@pytest.mark.parametrize('success', [False, True])
def test_actual_outcome_is_not_selected_by_success(success):
    result, launch = fixture(success)
    assert inputs.require_learned_result(result, launch) is result['conditions'][0]


@pytest.mark.parametrize('fault', ['status', 'sources', 'launch', 'population', 'retry', 'failures', 'prefix', 'claim', 'count'])
def test_incomplete_or_changed_learned_result_is_rejected(fault):
    result, launch = fixture()
    if fault == 'status': result['status'] = 'RUNNING'
    elif fault == 'sources': result['source_sha256']['changed'] = 'source'
    elif fault == 'launch': result['artifact_sha256']['launch.json'] = 'other'
    elif fault == 'population': result['conditions'].append(result['conditions'][0])
    elif fault == 'retry': result['automatic_retry'] = True
    elif fault == 'failures': result['original_failed_waiters_preserved'] = False
    elif fault == 'prefix': result['controller_completion_sha256'] = 'other'
    elif fault == 'claim': result['navigation_qualified'] = True
    elif fault == 'count': result['measured_round_trip_successes'] = 1
    with pytest.raises(ValueError): inputs.require_learned_result(result, launch)


def test_live_original_owner_prevents_reading_future_completion(monkeypatch):
    monkeypatch.setattr(inputs, 'learned_launch', lambda: {'owner': inputs.LEARNED_OWNER})
    monkeypatch.setattr(inputs.run, 'owner_live', lambda owner: owner == inputs.LEARNED_OWNER)
    monkeypatch.setattr(inputs.run, 'read_json', lambda *a, **k: pytest.fail('future result read while owner live'))
    with pytest.raises(ValueError, match='must end'):
        inputs.admit('future-sha', {})

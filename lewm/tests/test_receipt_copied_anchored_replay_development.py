from copy import deepcopy
import pytest
from scripts.replay_go2_receipt_copied_anchored_prefix_v1 import (
    normalize_candidate, execution_order, timing_summary, FRAMES)


def test_normalization_removes_only_explicit_implementation_metadata():
    original = dict(controller='receipt_copied_residual_anchored_continuation_controller_v1',
        anchored_selection_receipt_copy_enabled=True, requested_command=[.2, 0., 0.],
        new_selection={'prediction': [1., 2.], 'evidence': {'keep': True}})
    before = deepcopy(original); result = normalize_candidate(original)
    assert original == before
    assert result == dict(controller='residual_anchored_continuation_controller_v1',
        requested_command=[.2, 0., 0.], new_selection={'prediction': [1., 2.], 'evidence': {'keep': True}})
    for patch in ({'controller': 'other'}, {'anchored_selection_receipt_copy_enabled': False}):
        with pytest.raises(ValueError): normalize_candidate(original | patch)


def rows():
    return [dict(frame=i,execution_order=list(execution_order(i)),original_controller_s=.4,candidate_controller_s=.2)
        for i in range(FRAMES)]


def test_fixed_windows_balance_order_and_do_not_claim_100ms_from_speed_ratio():
    data = rows(); result = timing_summary(data)
    assert result['post_warmup_prefix']['observations'] == 402
    for name in ('early_navigation', 'repeated_hold'):
        window = result[name]; selected = data[window['first_frame']:window['last_frame']+1]
        assert window['observations'] == 10
        assert sum(r['execution_order'][0] == 0 for r in selected) == 5
        assert window['median_ratio'] == 2.
        assert window['candidate_over_100ms'] == 10


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'order', 'nan', 'zero', 'bool'])
def test_incomplete_or_invalid_paired_timings_are_rejected(fault):
    data = rows()
    if fault == 'missing': data.pop()
    elif fault == 'duplicate': data[-1]['frame'] = 403
    elif fault == 'order': data[3]['execution_order'] = [0, 1]
    else: data[3]['candidate_controller_s'] = {'nan':float('nan'),'zero':0.,'bool':True}[fault]
    with pytest.raises(ValueError): timing_summary(data)


@pytest.mark.parametrize('frame', [-1, 405, True, 1.5])
def test_execution_order_rejects_outside_prefix(frame):
    with pytest.raises(ValueError): execution_order(frame)

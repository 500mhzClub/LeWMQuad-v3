from copy import deepcopy
import numpy as np
import pytest
from lewm.causal_executed_residual_diagnosis_development import replay_bias


def rows():
    return [dict(tick=i, available_tick=i+1, predicted_body_xy_m=[.02, .01],
        observed_body_xy_m=[.01, -.01]) for i in range(12)]


def test_constant_bias_is_estimated_only_after_first_observation_and_window_is_bounded():
    r = replay_bias(rows())
    assert r[0]['correction_xy_m'] == [0., 0.] and r[0]['residual_source_ticks'] == []
    np.testing.assert_allclose(r[1]['corrected_body_xy_m'], [.01, -.01], atol=1e-15)
    assert r[-1]['residual_source_ticks'] == list(range(3, 11))
    assert r[-1]['residual_available_ticks'] == list(range(4, 12))
    assert all(x['observed_residual_samples'] <= 8 for x in r)


def test_current_and_future_targets_cannot_change_any_earlier_corrected_prediction():
    source = rows(); before = deepcopy(source); expected = replay_bias(source)
    for row in source[5:]: row['observed_body_xy_m'] = [1., -1.]
    changed = replay_bias(source)
    assert changed[:6] == expected[:6] and changed[6] != expected[6]
    assert replay_bias(before) == expected


def test_missing_outcomes_are_not_imputed_or_retained_past_clock_window():
    source = [rows()[0], rows()[11]]; r = replay_bias(source)
    assert r[1]['residual_source_ticks'] == [] and r[1]['correction_xy_m'] == [0., 0.]


def test_native_fields_and_invalid_label_chronology_are_rejected():
    for mutate in (lambda r: r.update(native_body_xy_m=[0., 0.]),
            lambda r: r.update(available_tick=0),
            lambda r: r.update(observed_body_xy_m=[float('nan'), 0.])):
        source = rows(); mutate(source[0])
        with pytest.raises(ValueError): replay_bias(source)
    with pytest.raises(ValueError): replay_bias([rows()[1], rows()[0]])

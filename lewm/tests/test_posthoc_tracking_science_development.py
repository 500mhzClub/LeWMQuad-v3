"""Summary accounting tests with tiny synthetic rows only."""
import pytest
from scripts.read_go2_tracking_posthoc_science_v1 import paired_rows


def population():
    stream = dict(arm_availability=dict(original=dict(first_failure=1),
        temporal_anchor=dict(first_failure=None)), score=dict(frames=3))
    estimates = [dict(frame=i, arms={a: dict(observer_wall_ms=ms,
        observer_update_attempted=attempt) for a, ms, attempt in (
        ('original', [80., 120., .001][i], i <= 1),
        ('temporal_anchor', [95., 105., 90.][i], True))}) for i in range(3)]
    errors = {'position_m': .01, 'orientation_rad': .001,
              'incremental_position_m': None, 'incremental_orientation_rad': None}
    evaluated = [dict(frame=i, availability='both' if i == 0 else 'temporal_anchor_only',
        errors=dict(original=errors, temporal_anchor=errors)) for i in range(3)]
    return estimates, evaluated, stream


def test_failed_update_included_and_terminal_noops_excluded():
    result = paired_rows(*population())
    times = result['actual_update_timing']
    assert times['original']['statistics_ms']['count'] == 2
    assert times['original']['statistics_ms']['mean'] == 100.
    assert times['original']['above_100ms'] == 1
    assert times['temporal_anchor']['statistics_ms']['count'] == 3
    contrast = result['shared_support_candidate_minus_original']
    assert contrast['position_m']['count'] == 1
    assert contrast['position_m']['mean'] == 0.
    assert contrast['incremental_position_m']['count'] == 0


@pytest.mark.parametrize('fault', ['missing', 'extra', 'update', 'frame'])
def test_partial_or_misaccounted_stream_is_rejected(fault):
    rows, evaluated, stream = population()
    if fault == 'missing': evaluated.pop()
    elif fault == 'extra': rows.append(rows[-1])
    elif fault == 'update': rows[2]['arms']['original']['observer_update_attempted'] = True
    else: evaluated[-1]['frame'] = 10
    with pytest.raises(ValueError):
        paired_rows(rows, evaluated, stream)

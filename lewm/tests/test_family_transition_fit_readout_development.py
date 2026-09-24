from copy import deepcopy
import pytest
from scripts.read_go2_family_transition_fits_v1 import summarize, VARIANTS, CONDITIONS


def fixture():
    return {f'{v}_{c}': {r: dict(role=r, clusters=[dict(scope=s, cluster=str(i), motion_targets=2,
        contact_targets=3, contact_positives=1, undefined_yaw=0, position_error_m=float(i),
        yaw_error_rad=.1, contact_brier=.2) for s in ('all', 'initial', 'moving') for i in (0, 1)])
        for r in ('train', 'geometry_transfer')} for v in VARIANTS for c in CONDITIONS}


def test_complete_cluster_macros_and_every_predefined_contrast():
    scores = fixture(); report = summarize(scores)
    assert len(report['rows']) == 36 and len(report['contrasts']) == 42
    assert all(r['macro']['position_error_m'] == .5 for r in report['rows'])
    assert all(all(v == 0 for v in r['left_minus_right'].values()) for r in report['contrasts'])
    assert report['learned_benefit_established'] is False


def test_missing_yaw_and_population_changes_cannot_hide_in_complete_macro():
    scores = fixture(); scores['full_jepa']['geometry_transfer']['clusters'][0]['yaw_error_rad'] = None
    report = summarize(scores)
    row = next(r for r in report['rows'] if (r['method'], r['role'], r['scope']) == ('full_jepa', 'geometry_transfer', 'all'))
    assert row['macro']['yaw_error_rad'] is None
    bad = deepcopy(scores); bad['no_rgb_jepa']['train']['clusters'][0]['motion_targets'] -= 1
    with pytest.raises(ValueError, match='populations'): summarize(bad)

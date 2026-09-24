"""Independent distribution reference, fixed denominators and negative outcomes."""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations
import pytest
from scipy.stats import binomtest
from lewm import independent_round_trip_exact_comparison_analysis_development as analysis


def synthetic_report(success):
    rows=[dict(case=c.name,layout_index=c.layout_index,arm=c.arm_name,
        readout=dict(verified_round_trip=success(c.layout_index,c.arm_name)),
        startup_comparison=dict(status=analysis.MATCHED)) for c in analysis.CASES]
    lookup={(r['layout_index'],r['arm']):r['readout']['verified_round_trip'] for r in rows}
    arms=[]
    for arm in analysis.ARMS:
        count=sum(lookup[i,arm.name] for i in range(8))
        arms.append(dict(arm=arm.name,planned_layouts=8,completed_layouts=8,verified_round_trips=count,
            verified_round_trip_rate=count/8,scientific_failures=8-count,
            cases=[next(r['case'] for r in rows if r['layout_index']==i and r['arm']==arm.name) for i in range(8)]))
    comparisons=[]
    for name,left,right in analysis.COMPARISONS:
        wins=sum(lookup[i,left] and not lookup[i,right] for i in range(8))
        losses=sum(lookup[i,right] and not lookup[i,left] for i in range(8))
        comparisons.append(dict(comparison=name,left_arm=left,right_arm=right,layout_pairs=8,
            left_only_successes=wins,right_only_successes=losses,tied_outcomes=8-wins-losses,
            paired_success_rate_difference=(wins-losses)/8))
    return dict(status='INDEPENDENT_ROUND_TRIP_MULTIARM_COMPLETE',all_fixed_cases_executed=True,
        planned_episodes=32,completed_episodes=32,independent_layout_units=8,ordered_case_readouts=rows,
        arms=arms,comparisons=comparisons,measured_round_trip_successes=sum(r['verified_round_trips'] for r in arms),
        all_startups_matched=True,scientific_failures_retained=True,outcome_based_layout_replacement=False,
        treatment_repetitions_counted_as_independent_layouts=False)


@pytest.mark.parametrize('wins,losses',[(w,l) for w in range(9) for l in range(9-w)])
def test_exact_discordant_probability_matches_independent_scipy_reference(wins,losses):
    p=analysis.exact_paired_pvalue(wins,losses)
    assert p==analysis.exact_paired_pvalue(losses,wins)
    expected=1. if wins+losses==0 else binomtest(wins,wins+losses,.5,alternative='two-sided').pvalue
    assert float(p)==pytest.approx(expected,abs=1e-15)


@pytest.mark.parametrize('successes',range(9))
@pytest.mark.parametrize('alpha',[Fraction(1,20),Fraction(1,80)])
def test_interval_matches_independent_exact_binomial_reference(successes,alpha):
    expected=binomtest(successes,8).proportion_ci(confidence_level=1-float(alpha),method='exact')
    actual=analysis.exact_interval(successes,alpha=alpha)
    assert actual==pytest.approx(dict(low=expected.low,high=expected.high),abs=1e-11)


@pytest.mark.parametrize('order',list(permutations(range(3))))
def test_holm_preserves_comparison_order_and_monotonic_step_down(order):
    raw=[Fraction(1,100),Fraction(4,100),Fraction(5,100)]
    expected=[Fraction(3,100),Fraction(8,100),Fraction(8,100)]
    assert analysis.holm_three([raw[i] for i in order])==[expected[i] for i in order]


def test_all_successes_do_not_prove_difference_or_high_population_reliability():
    report=synthetic_report(lambda i,arm:True);before=deepcopy(report)
    out=analysis.analyze_complete_readout(report)
    assert report==before and out['independent_layout_units']==8
    assert all(not r['conditional_reject_at_0_05'] and r['exact_two_sided_p']['value']==1 for r in out['comparisons'])
    for arm in out['arms']:
        assert arm['marginal_95pct_exact_interval']['low']==pytest.approx(.6305833524471808)
        assert arm['four_arm_family_95pct_bonferroni_exact_interval']['low']<.54
    assert not out['sampling_assumptions_verified'] and not out['goal_achieved']


def test_all_failures_remain_in_denominators_and_do_not_establish_advantage():
    out=analysis.analyze_complete_readout(synthetic_report(lambda i,arm:False))
    assert all(arm['layouts']==8 and arm['successes']==0 for arm in out['arms'])
    assert all(row['tied_outcomes']==8 and not row['conditional_reject_at_0_05'] for row in out['comparisons'])


def test_maximal_difference_is_adjusted_over_all_three_hypotheses():
    out=analysis.analyze_complete_readout(synthetic_report(lambda i,arm:arm=='persistent_jepa'))
    for row in out['comparisons']:
        assert row['exact_two_sided_p']['value']==1/128
        assert row['holm_adjusted_p']['value']==3/128 and row['conditional_reject_at_0_05']
    assert 'whole predictive method' in out['comparisons'][1]['interpretation']
    assert 'other histories retained' in out['comparisons'][2]['interpretation']
    assert not out['navigation_qualified'] and out['upstream_artifact_authentication_required']


def test_unmatched_startup_withholds_all_comparisons_without_dropping_layouts():
    report=synthetic_report(lambda i,arm:i<4)
    report['ordered_case_readouts'][17]['startup_comparison']['status']='DIVERGED'
    report['all_startups_matched']=False
    out=analysis.analyze_complete_readout(report)
    assert out['comparisons']==[] and not out['inferential_comparisons_available']
    assert all(arm['layouts']==8 and arm['successes']==4 for arm in out['arms'])


@pytest.mark.parametrize('fault',['missing_case','duplicate_case','row_order','outcome_integer','layout_bool',
    'aggregate_success','arm_count','arm_rate_bool','pair_count','pair_rate_bool','startup_summary',
    'completed_count','denominator','failure_exclusion','replacement','independent_treatments'])
def test_changed_roster_denominator_or_summary_is_rejected(fault):
    r=synthetic_report(lambda i,arm:False)
    if fault=='missing_case':r['ordered_case_readouts'].pop()
    elif fault=='duplicate_case':r['ordered_case_readouts'][1]=deepcopy(r['ordered_case_readouts'][0])
    elif fault=='row_order':r['ordered_case_readouts'].reverse()
    elif fault=='outcome_integer':r['ordered_case_readouts'][0]['readout']['verified_round_trip']=0
    elif fault=='layout_bool':r['ordered_case_readouts'][0]['layout_index']=False
    elif fault=='aggregate_success':r['measured_round_trip_successes']=1
    elif fault=='arm_count':r['arms'][0]['verified_round_trips']=1
    elif fault=='arm_rate_bool':r['arms'][0]['verified_round_trip_rate']=False
    elif fault=='pair_count':r['comparisons'][0]['left_only_successes']=1
    elif fault=='pair_rate_bool':r['comparisons'][0]['paired_success_rate_difference']=False
    elif fault=='startup_summary':r['all_startups_matched']=False
    elif fault=='completed_count':r['completed_episodes']=31
    elif fault=='denominator':r['independent_layout_units']=32
    elif fault=='failure_exclusion':r['scientific_failures_retained']=False
    elif fault=='replacement':r['outcome_based_layout_replacement']=True
    elif fault=='independent_treatments':r['treatment_repetitions_counted_as_independent_layouts']=True
    with pytest.raises(ValueError):analysis.analyze_complete_readout(r)


@pytest.mark.parametrize('wins,losses',[(-1,0),(9,0),(4,5),(True,0),(0,1.)])
def test_invalid_discordant_counts_rejected(wins,losses):
    with pytest.raises(ValueError):analysis.exact_paired_pvalue(wins,losses)


def test_reduced_comparison_family_cannot_be_selected_after_readout():
    with pytest.raises(ValueError):analysis.holm_three([Fraction(1,128)])

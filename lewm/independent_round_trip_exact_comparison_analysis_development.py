"""Prospective statistical supplement for the fixed eight-layout, four-arm study.

The caller must authenticate the complete readout upstream. This module checks
its outcome projection; it does not authenticate artifacts or verify sampling
assumptions, approve a population definition, or establish navigation readiness.
"""
from fractions import Fraction
from math import comb
from lewm.independent_round_trip_comparison_study_development import ARMS, CASES
from lewm.independent_round_trip_population_readout_development import COMPARISONS, MATCHED

LAYOUTS=8
ALPHA=Fraction(1,20)
INTERPRETATIONS={
    'training_objective':'fixed full-RGB JEPA versus supervised-rollout models',
    'predictive_method':'whole predictive method versus reactive observed-route method',
    'planning_grid_persistence':'accumulated planning cells; other histories retained',
}


def exact_paired_pvalue(wins,losses):
    if any(type(v) is not int or v<0 for v in (wins,losses)) or wins+losses>LAYOUTS:
        raise ValueError('zero to eight discordant independent-layout outcomes required')
    n=wins+losses
    return min(Fraction(1),Fraction(2*sum(comb(n,j) for j in range(min(wins,losses)+1)),2**n))


def holm_three(pvalues):
    if len(pvalues)!=3 or any(type(p) is not Fraction or not 0<=p<=1 for p in pvalues):
        raise ValueError('all three exact predeclared comparison p-values required')
    adjusted=[None]*3;running=Fraction(0)
    for rank,index in enumerate(sorted(range(3),key=lambda i:pvalues[i])):
        running=max(running,min(Fraction(1),(3-rank)*pvalues[index]))
        adjusted[index]=running
    return adjusted


def exact_interval(successes,*,alpha=ALPHA):
    """Two-sided Clopper-Pearson bounds for the fixed eight-layout denominator."""
    if type(successes) is not int or not 0<=successes<=LAYOUTS:
        raise ValueError('zero to eight successes required')
    if type(alpha) is not Fraction or not 0<alpha<1:
        raise ValueError('explicit rational error probability required')
    target=float(alpha/2)
    def root(lower):
        lo,hi=0.,1.
        for _ in range(64):
            p=(lo+hi)/2
            js=range(successes,LAYOUTS+1) if lower else range(successes+1)
            tail=sum(comb(LAYOUTS,j)*p**j*(1-p)**(LAYOUTS-j) for j in js)
            if (tail<target if lower else tail>target):lo=p
            else:hi=p
        return (lo+hi)/2
    return dict(low=0. if successes==0 else root(True),high=1. if successes==LAYOUTS else root(False))


def fraction_receipt(value):
    return dict(numerator=value.numerator,denominator=value.denominator,value=float(value))


def outcome_projection(report):
    if (report['status']!='INDEPENDENT_ROUND_TRIP_MULTIARM_COMPLETE'
            or report['all_fixed_cases_executed'] is not True
            or report['scientific_failures_retained'] is not True
            or report['outcome_based_layout_replacement'] is not False
            or report['treatment_repetitions_counted_as_independent_layouts'] is not False
            or any(type(report[k]) is not int or report[k]!=v for k,v in
                (('planned_episodes',32),('completed_episodes',32),('independent_layout_units',8)))
            or len(report['ordered_case_readouts'])!=32):
        raise ValueError('complete fixed 32-case readout with eight layout units required')
    lookup={};matched=True
    for case,row in zip(CASES,report['ordered_case_readouts'],strict=True):
        if (row['case']!=case.name or type(row['layout_index']) is not int
                or row['layout_index']!=case.layout_index or row['arm']!=case.arm_name
                or type(row['readout']['verified_round_trip']) is not bool):
            raise ValueError('exact ordered cases and boolean verified outcomes required')
        lookup[case.layout_index,case.arm_name]=row['readout']['verified_round_trip']
        matched &= row['startup_comparison']['status']==MATCHED
    if report['all_startups_matched'] is not matched:
        raise ValueError('startup summary must agree with every retained case')
    counts={arm.name:sum(lookup[i,arm.name] for i in range(LAYOUTS)) for arm in ARMS}
    if (type(report['measured_round_trip_successes']) is not int
            or report['measured_round_trip_successes']!=sum(counts.values())
            or len(report['arms'])!=4):
        raise ValueError('all aggregate successes must reconstruct')
    for arm,row in zip(ARMS,report['arms'],strict=True):
        if (row['arm']!=arm.name or any(type(row[k]) is not int or row[k]!=v for k,v in
                (('planned_layouts',8),('completed_layouts',8),('verified_round_trips',counts[arm.name]),
                 ('scientific_failures',8-counts[arm.name])))
                or type(row['verified_round_trip_rate']) is not float or row['verified_round_trip_rate']!=counts[arm.name]/8
                or row['cases']!=[next(c.name for c in CASES if c.layout_index==i and c.arm_name==arm.name) for i in range(8)]):
            raise ValueError('per-arm denominator and success aggregates must reconstruct')
    if len(report['comparisons'])!=3:raise ValueError('all three original comparisons required')
    for (name,left,right),comparison in zip(COMPARISONS,report['comparisons'],strict=True):
        wins=sum(lookup[i,left] and not lookup[i,right] for i in range(8))
        losses=sum(lookup[i,right] and not lookup[i,left] for i in range(8))
        if (comparison['comparison']!=name or comparison['left_arm']!=left or comparison['right_arm']!=right
                or any(type(comparison[k]) is not int or comparison[k]!=v for k,v in
                    (('layout_pairs',8),('left_only_successes',wins),('right_only_successes',losses),('tied_outcomes',8-wins-losses)))
                or type(comparison['paired_success_rate_difference']) is not float
                or comparison['paired_success_rate_difference']!=(wins-losses)/8):
            raise ValueError('paired aggregate outcomes must reconstruct')
    return lookup,counts,matched


def analyze_complete_readout(report):
    lookup,counts,matched=outcome_projection(report)
    arms=[dict(arm=arm.name,successes=counts[arm.name],layouts=8,rate=counts[arm.name]/8,
        marginal_95pct_exact_interval=exact_interval(counts[arm.name]),
        four_arm_family_95pct_bonferroni_exact_interval=exact_interval(counts[arm.name],alpha=ALPHA/4)) for arm in ARMS]
    pairs=[]
    if matched:
        pvalues=[]
        for name,left,right in COMPARISONS:
            wins=sum(lookup[i,left] and not lookup[i,right] for i in range(8))
            losses=sum(lookup[i,right] and not lookup[i,left] for i in range(8))
            p=exact_paired_pvalue(wins,losses);pvalues.append(p)
            pairs.append(dict(comparison=name,interpretation=INTERPRETATIONS[name],left_arm=left,right_arm=right,
                layout_pairs=8,left_only_successes=wins,right_only_successes=losses,tied_outcomes=8-wins-losses,
                paired_success_rate_difference=(wins-losses)/8,exact_two_sided_p=fraction_receipt(p)))
        for row,p in zip(pairs,holm_three(pvalues),strict=True):
            row.update(holm_adjusted_p=fraction_receipt(p),conditional_reject_at_0_05=p<=ALPHA,
                observed_direction='left' if row['left_only_successes']>row['right_only_successes'] else
                    'right' if row['right_only_successes']>row['left_only_successes'] else 'tie')
    return dict(schema='independent_round_trip_exact_comparison_analysis_development.v1',
        planned_episodes=32,independent_layout_units=8,arms=arms,comparisons=pairs,
        all_cases_retained=True,inferential_comparisons_available=matched,
        comparisons_withheld_reason=None if matched else 'one or more startups unmatched; no pair exclusion',
        family_alpha=fraction_receipt(ALPHA),comparison_family_size=3,
        assumptions=['independent representative layout draws for binomial coverage',
            'independent paired discordance signs with equal direction probabilities under each null',
            'matched implementations and startup evidence; fixed hypotheses before outcome access'],
        sampling_assumptions_verified=False,upstream_artifact_authentication_required=True,
        raw_audits_reexecuted=False,population_definition_selected=False,
        navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False)

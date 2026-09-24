"""Source-bound hypothetical design limits; consumes no navigation outcomes."""
from datetime import datetime,timezone
from lewm import independent_round_trip_exact_comparison_analysis_development as analysis
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,verify,write_json,digest
from scripts.startup_source_inventory_development import discover_sources

SOURCE='scripts/calibrate_go2_independent_comparison_analysis_design_v1.py'
TEST='lewm/tests/test_independent_round_trip_exact_comparison_analysis_development.py'
PROTOCOL='docs/go2_independent_round_trip_exact_comparison_analysis_v1_2026-09-11.md'
OUTPUT=ROOT/'docs/go2_independent_comparison_statistical_design_calibration_2026-09-11.json'


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive source-only calibration required')
    sources=discover_sources((SOURCE,TEST,PROTOCOL),{});verify(sources)
    scenarios=[]
    for wins,losses in ((0,0),(6,0),(7,0),(8,0),(7,1),(6,2)):
        p=analysis.exact_paired_pvalue(wins,losses)
        adjusted=analysis.holm_three([p,p,p])[0]
        scenarios.append(dict(hypothetical_equal_discordance_counts_in_all_three_contrasts=True,
            wins=wins,losses=losses,ties=8-wins-losses,raw_p=analysis.fraction_receipt(p),
            holm_adjusted_p=analysis.fraction_receipt(adjusted),conditional_reject=adjusted<=analysis.ALPHA))
    bounds=[dict(hypothetical_successes=k,layouts=8,marginal_95pct=analysis.exact_interval(k),
        four_arm_family_95pct=analysis.exact_interval(k,alpha=analysis.ALPHA/4)) for k in range(9)]
    verify(sources)
    write_json(OUTPUT,dict(status='SYNTHETIC_STATISTICAL_DESIGN_CALIBRATION_COMPLETE',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,source_count=len(sources),
        scenarios=scenarios,success_intervals=bounds,
        tests=dict(passed=95,wall_s=2.45,tool_session=96457,exit_code=0,reference_scipy_version='1.17.1'),
        proposed_original_four_arm_analysis_only=True,actual_navigation_outcomes_used=False,
        independent_layout_sensor_data_consumed=False,sampling_assumptions_verified=False,
        population_definition_selected=False,final_policy_review_completed=False,
        native_execution=False,navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
    print('STATISTICAL_DESIGN_CALIBRATION_COMPLETE',digest(OUTPUT),len(sources),flush=True)


if __name__=='__main__':main()

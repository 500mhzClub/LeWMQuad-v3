"""Summarize every assignment in the fixed four-run exposed-maze batch."""
import json
from scripts import run_go2_view_replan_repeatability_development as run


def main():
    output=run.BASE/'go2_view_replan_repeatability_readout_v1_attempt_001'
    rows=[]
    for number,arm in enumerate(run.ARMS,1):
        root=run.BASE/run.root_name(number)
        read=lambda name:json.loads((root/name).read_text())
        result=read('view_replan_repeatability_readout_v1.json')
        launch=read('launch.json');navigation=result['navigation'];backtrack=result['physical_backtracking']
        assert result['batch_assignment']==number and result['arm']==arm and launch['study_arm']==arm
        physical=read('continuous_native_arrival_evaluation.json')
        forecasts=read('saved_short_pulse_same_window_xy_v1.json')
        yaw=read('saved_short_pulse_yaw_evaluation_v1.json')
        rows.append(dict(assignment=number,arm=arm,root=str(root),
            round_trip=navigation['round_trip'],contacts=navigation['contacts'],
            arrivals=navigation['arrivals'],terminal=navigation['terminal'],failure=navigation['failure'],
            simulation_s=navigation['simulation_s'],plans=navigation['plans'],plans_on_time=navigation['plans_on_time'],
            physical_backtracking=backtrack,physical_arrival_checks=physical['round_trip_arrival_checks_passed'],
            interrupted_views=len(result['interrupted_view_events']),coverage_rejections=result['coverage_rejections'],
            observed_coverage_patches=result['observed_coverage_patches'],actions=result['actions'],
            pipeline_faults=result['pipeline_faults'],forecast_windows=forecasts['windows'],
            executed_window_xy_rmse_mm=forecasts.get('rmse_mm'),executed_window_yaw_rmse_deg=yaw['rmse_deg']))
    result=dict(schema='view_replan_repeatability_complete_result.v1',plan=str(run.PLAN),rows=rows,
        completed_assignments=len(rows),round_trips=sum(r['round_trip'] for r in rows),
        contacts=sum(r['contacts'] for r in rows),
        by_arm={a:dict(attempts=sum(r['arm']==a for r in rows),
            round_trips=sum(r['round_trip'] for r in rows if r['arm']==a)) for a in dict.fromkeys(run.ARMS)},
        all_failures_included=True,independent_layout_replication=False,
        intervention_causal_effect_established=False,jepa_advantage_established=False,
        simulation_not_real_time_or_hardware_qualification=True)
    output.mkdir(exist_ok=False);run.save(output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k not in
        ('arrivals','physical_backtracking','root','pipeline_faults','actions','failure')}))


if __name__=='__main__':main()

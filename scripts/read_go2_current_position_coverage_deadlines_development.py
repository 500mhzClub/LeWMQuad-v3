"""Separate missing-view recovery from deadline starvation in the saved failure."""
from collections import Counter
import json
import numpy as np

from scripts.run_go2_current_position_coverage_view_development import BASE,ROOT


def main():
    root=BASE/ROOT; output=root/'planning_deadline_diagnosis_v1.json'
    if output.exists():raise ValueError('preserve completed diagnosis')
    read=lambda name:json.loads((root/name).read_text())
    plans=[r for r in read('planning.json') if 'selection' in r]
    profiles=read('live_planning_profile.json');requests=read('requests.json')
    summaries=[]
    for low,high in ((0,600),(600,1600),(1600,4466)):
        rows=[r for r in plans if low<=r['frame']<high]
        prof=[r for r in profiles if low<=r['frame']<high]
        dispatched=[r for r in requests if 1_500_000_000+low*100_000_000<=r['simulator_ns']<1_500_000_000+high*100_000_000]
        costs={}
        for component in ('route','model_forward','action_selection','predictive_clearance'):
            costs[component]={measure:dict(zip(('median','p95','maximum'),
                np.percentile([p['components'][component][measure]/1e6 for p in prof
                    if component in p['components']],[50,95,100]).tolist()))
                for measure in ('wall_ns','thread_cpu_ns')}
        summaries.append(dict(camera_frame_interval=[low,high],plans=len(rows),
            on_time=sum(r['on_time'] for r in rows),actions=dict(Counter(r['action'] for r in rows)),
            all_candidates_nominally_blocked=sum(all(not c['nominal_predicted_path_clear']
                for c in r['selection']['memory_forecast_candidates']) for r in rows),
            coverage_rejections=sum(r['selection'].get('translation_footprint_coverage',{}).get('rejected',False) for r in rows),
            observation_to_plan_completion_ms=dict(zip(('median','p95','maximum'),np.percentile(
                [(r['completed_ns']-r['measured_ns'])/1e6 for r in rows],[50,95,100]).tolist())),
            component_ms=costs,components_nested_not_additive=True,
            requested_intervals=len(dispatched),nonzero_requested_intervals=sum(any(r['requested_command']) for r in dispatched),
            dispatch_reasons=dict(Counter(r['reason'] for r in dispatched))))
    events=read('stage_events.json')
    tail={stage:[e for e in events if e['stage']==stage][-4:]
        for stage in ('registration','mapping','planning')}
    result=dict(schema='current_position_coverage_deadlines.v1',intervals=summaries,
        terminal_stage_events=tail,failure=read('failure.json'),pipeline_faults=read('pipeline_faults.json'),
        queue_failure_interpretation='registration enqueued mapping frame 4464, then overflowed the two-slot planning queue while frame 4452 was active and 4456/4460 were waiting',
        queue_identity_inferred_from_source_and_recorded_stage_order=True,
        queue_identity_not_in_original_exception=True,
        narrower_routing_hotspot_not_yet_isolated=True,
        alternative_navigation_outcome_proven=False)
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in summaries:
        print(json.dumps({k:v for k,v in row.items() if k not in ('component_ms','dispatch_reasons')},indent=2))
        print('component_median_wall_ms',{k:v['wall_ns']['median'] for k,v in row['component_ms'].items()})
        print('dispatch_reasons',row['dispatch_reasons'])


if __name__=='__main__':main()

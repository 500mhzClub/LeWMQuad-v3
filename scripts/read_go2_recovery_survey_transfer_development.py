"""Summarize one completed fixed survey-transfer assignment and all saved outcomes."""
import json, pathlib, statistics, collections, sys
base=pathlib.Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
fixed=json.loads(pathlib.Path('docs/go2_recovery_survey_transfer_plan_2026-09-16.json').read_text())
assignment=int(sys.argv[1]); index,condition=fixed['assignments'][assignment-1]
root=base/f'go2_recovery_survey_transfer_{condition}_jepa_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
read=lambda f:json.loads((root/f).read_text())
plans=[x for x in read('planning.json') if 'selection' in x]
profiles={x['frame']:x for x in read('live_planning_profile.json')}
bins=[]
for lo in range(0,5000,500):
 rows=[x for x in plans if lo<=x['frame']<lo+500]
 if not rows:continue
 times={}
 for name in ['route','action_selection','model_forward','whole_plan_including_release_wait']:
  samples=[profiles[x['frame']]['components'][name] for x in rows if name in profiles[x['frame']]['components']]
  times[name]={f:statistics.median(s[f]/1e6 for s in samples) for f in ['wall_ns','thread_cpu_ns']} if samples else None
 bins.append(dict(frame_start=lo,frame_end_exclusive=lo+500,plans=len(rows),on_time=sum(x['on_time'] for x in rows),route_status=dict(collections.Counter(x['route_status'] for x in rows)),median_component_ms=times))
survey=read('initial_survey.json');events=read('visual_dispatch_events.json');requests=read('requests.json')
violations=[];ties=[]
for event in events:
 for r in requests:
  if r.get('command_observation_ns',event['trigger_ns'])>=event['trigger_ns'] or not any(r['requested_command']):continue
  if r['now_ns']>event['published_ns']:
   violations.append(dict(event=event,request=r))
  elif r['now_ns']==event['published_ns']:
   ties.append(dict(event=event,request=r,gate_observed_new_threshold=r.get('visual_recovery_plan_minimum_ns',-1)>=event['trigger_ns']))
treatment=dict(condition=condition,survey_complete=survey.get('complete'),completed_views=len(survey.get('completed_view_stages',[])),survey_deferred=survey.get('deferred',False),deferral=survey.get('deferral'),recovery_publications=len(events),cancelled_windows=sum(len(x['cancelled_plan_observations_ns']) for x in events),old_nonzero_requests_strictly_after_publication=violations,same_clock_ties=ties)
result=dict(assignment=assignment,layout_index=index,condition=condition,root=root.name,outcome=read('short_pulse_navigation_evaluation_v1.json'),timing=read('planning_latency_stress_diagnosis_v1.json'),treatment=treatment,latency_by_frame_interval=bins,forecast_xy={k:v for k,v in read('saved_short_pulse_same_window_xy_v1.json').items() if k!='rows'},forecast_yaw={k:v for k,v in read('saved_short_pulse_yaw_evaluation_v1.json').items() if k!='rows'},timing_components_inclusive=True)
if (root/'physical_return_corridor_readout_v1.json').exists():result['backtracking']={k:v for k,v in read('physical_return_corridor_readout_v1.json').items() if k not in ['transitions','reverse_edges']}
with (root/'survey_transfer_scientific_readout_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
out=base/'go2_recovery_survey_transfer_comparison_v1_attempt_001';out.mkdir(exist_ok=True)
completed=[]
for n,(i,c) in enumerate(fixed['assignments'],1):
 p=base/f'go2_recovery_survey_transfer_{c}_jepa_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'/'survey_transfer_scientific_readout_v1.json'
 if p.is_file():completed.append(json.loads(p.read_text()))
summary=dict(schema='recovery_survey_transfer_comparison.v1',planned=4,evaluated=len(completed),complete=len(completed)==4,round_trips=sum(r['outcome']['round_trip'] for r in completed),contacts=sum(r['outcome']['contacts'] for r in completed),records=completed,model_changed=False,training_seed=fixed['training_seed'],final_evaluation=False,hardware_validated=False)
(out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(assignment=assignment, outcome=result['outcome'], survey_deferred=treatment['survey_deferred'], completed_views=treatment['completed_views'], recovery_publications=len(events), cancelled_windows=treatment['cancelled_windows'], late_old_requests=len(violations), clock_ties=len(ties), aggregate_evaluated=len(completed))))


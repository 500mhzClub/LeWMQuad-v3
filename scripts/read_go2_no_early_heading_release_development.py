"""Read full outcomes and actual intervention receipts for the fixed ablations."""
import json
from pathlib import Path
import sys

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
plan = json.loads(Path('docs/go2_no_early_heading_release_plan_2026-09-16.json').read_text())
assignment = int(sys.argv[1])
case = plan['cases'][assignment-1]
root = BASE/f"go2_no_early_heading_release_{case['name']}_jepa_noise_2mm_native_layout01_4800_v1_attempt_001"
reference = BASE/case['reference']
read = lambda p, n: json.loads((p/n).read_text())
plans = [r for r in read(root, 'planning.json') if 'selection' in r]
suppressed = [r for r in plans if 'suppressed_early_heading_release' in r['selection']]
early_releases = [r for r in plans if r['selection'].get('clearance_turn', {}).get('event')
                  == 'FULL_RESERVE_PREFERRED_HEADING_REJOINS_ROUTE_OR_VIEW']
assert not early_releases
requests = {r['now_ns']: r for r in read(root, 'requests.json')}
reference_requests = {r['now_ns']: r for r in read(reference, 'requests.json')}
common = sorted(requests.keys() & reference_requests.keys())
first_difference = next((t for t in common if requests[t]['requested_command']
                         != reference_requests[t]['requested_command']), None)
first_suppression = min((r['completed_ns'] for r in suppressed), default=None)
result = dict(schema='no_early_heading_release_readout.v1', assignment=assignment,
    case=case['name'], reference_root=reference.name, root=root.name,
    reference_outcome=read(reference, 'short_pulse_navigation_evaluation_v1.json'),
    outcome=read(root, 'short_pulse_navigation_evaluation_v1.json'),
    timing=read(root, 'planning_latency_stress_diagnosis_v1.json'),
    eligible_releases_suppressed=len(suppressed),
    suppressed_plans_on_time=sum(r['on_time'] for r in suppressed),
    suppressed_plans_marked_committed=sum(r.get('committed', False) for r in suppressed),
    suppressed_plan_frames=[r['frame'] for r in suppressed],
    actual_early_heading_release_events=len(early_releases),
    first_suppression_plan_completed_ns=first_suppression,
    first_common_tick_requested_command_difference_ns=first_difference,
    common_request_ticks_compared=len(common),
    command_difference_preceded_first_suppression=(first_difference < first_suppression
        if first_difference is not None and first_suppression is not None else None),
    survey=read(root, 'initial_survey.json'),
    forecast_xy={k:v for k,v in read(root, 'saved_short_pulse_same_window_xy_v1.json').items() if k!='rows'},
    forecast_yaw={k:v for k,v in read(root, 'saved_short_pulse_yaw_evaluation_v1.json').items() if k!='rows'},
    single_exposed_execution=True, counterfactual_navigation_success_inferred=False)
if (root/'physical_return_corridor_readout_v1.json').is_file():
    result['backtracking']={k:v for k,v in read(root, 'physical_return_corridor_readout_v1.json').items()
                            if k not in ('transitions', 'reverse_edges')}
with (root/'no_early_heading_release_readout_v1.json').open('x') as f:
    json.dump(result, f, indent=2); f.write('\n')
records=[]
for c in plan['cases']:
    p=BASE/f"go2_no_early_heading_release_{c['name']}_jepa_noise_2mm_native_layout01_4800_v1_attempt_001"/'no_early_heading_release_readout_v1.json'
    if p.is_file(): records.append(json.loads(p.read_text()))
output=BASE/'go2_no_early_heading_release_comparison_v1_attempt_001'
output.mkdir(exist_ok=True)
(output/'result.json').write_text(json.dumps(dict(planned=2, evaluated=len(records),
    complete=len(records)==2, records=records, exposed_development_only=True), indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in
    ('timing', 'survey', 'forecast_xy', 'forecast_yaw', 'backtracking', 'suppressed_plan_frames')}, indent=2))

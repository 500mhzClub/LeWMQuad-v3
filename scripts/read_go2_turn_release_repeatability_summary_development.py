"""Summarize the completed four-run experiment, including command divergence."""
import collections
import itertools
import json
from pathlib import Path

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
OUTPUT = BASE/'go2_turn_release_repeatability_comparison_v1_attempt_001'
study = json.loads((OUTPUT/'result.json').read_text())
assert study['complete'] and study['evaluated'] == study['planned'] == 4
records = study['records']
requests = {}
plans = {}
rows = []
for record in records:
    root = BASE/record['root']
    case = record['case']
    requests[case] = {r['now_ns']: r for r in json.loads((root/'requests.json').read_text())}
    plans[case] = [p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p]
    events = collections.Counter(p['selection'].get('clearance_turn', {}).get('event')
                                 for p in plans[case])
    outcome = record['outcome']
    rows.append(dict(case=case, condition=record['condition'],
        round_trip=outcome['round_trip'], contacts=outcome['contacts'],
        simulation_s=outcome['simulation_s'], plans=outcome['plans'],
        plans_on_time=outcome['plans_on_time'],
        eligible_release_decisions=record['eligible_release_decisions'],
        applied_releases=record['release_decisions_applied'],
        suppressed_releases=record['eligible_releases_suppressed'],
        alternative_turn_latches=events['CLEAR_ALTERNATIVE_TURN_LATCHED'],
        translation_requested_s=record['timing']['nonzero_translation_requested_seconds'],
        forecast_xy_rmse_mm=record['forecast_xy']['rmse_mm'],
        forecast_yaw_rmse_deg=record['forecast_yaw']['rmse_deg'],
        physical_backtracking=record['backtracking']['physical_backtracking_observed']))

pairs = []
for a, b in itertools.combinations(requests, 2):
    common = sorted(requests[a].keys() & requests[b].keys())
    tick = next((t for t in common if requests[a][t]['requested_command']
                 != requests[b][t]['requested_command']), None)
    pair = dict(cases=[a, b], common_ticks=len(common), first_command_difference_ns=tick)
    if tick is not None:
        pair['requests_at_first_difference'] = {c: requests[c][tick] for c in (a, b)}
        # Preserve the nearby publications without attributing a unique cause.
        pair['nearby_plans'] = {c: [{k: p.get(k) for k in
            ('frame', 'measured_ns', 'completed_ns', 'action', 'on_time', 'committed', 'route_status')}
            for p in plans[c] if tick-600_000_000 <= p['measured_ns'] <= tick]
            for c in (a, b)}
    pairs.append(pair)

total_plans = sum(r['plans'] for r in rows)
on_time = sum(r['plans_on_time'] for r in rows)
result = dict(schema='heading_release_repeatability_complete_summary.v1',
    records=rows, round_trips=sum(r['round_trip'] for r in rows),
    total_contacts=sum(r['contacts'] for r in rows),
    total_plans=total_plans, plans_on_time=on_time, fraction_on_time=on_time/total_plans,
    total_eligible_release_decisions=sum(r['eligible_release_decisions'] for r in rows),
    first_command_differences=pairs, independent_layouts=1, model_training_seeds=1,
    all_runs_preserved=True, final_evaluation=False, hardware_validated=False,
    real_time_qualified=False, heading_release_benefit_established=False,
    additional_repetitions_in_this_batch=False,
    interpretation='Repeated physical navigation evidence on one exposed maze; '
        'zero eligible release decisions means the intended treatment was never exercised. '
        'Timing and trajectory differences cannot establish a release-rule benefit. '
        'Prior failures and long turning episodes remain separate preserved results.')
with (OUTPUT/'complete_scientific_readout_v1.json').open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(json.dumps({k: v for k, v in result.items() if k != 'first_command_differences'}, indent=2))
print('First command differences:', [(p['cases'], p['first_command_difference_ns']) for p in pairs])

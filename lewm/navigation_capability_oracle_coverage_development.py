"""Ensure every admitted oracle command lies in its selected verified prefix."""
from collections import Counter
import json


def coverage(root):
    plans={p['measured_ns']:p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p}
    requests=json.loads((root/'requests.json').read_text())
    verification=json.loads((root/'oracle_prefix_check.json').read_text())
    prefixes={(r['observed_ns'],r['candidate']):r for r in verification['comparisons']}
    failures=[];admitted=0;origins=set();excluded=Counter()
    for row in requests:
        origin=row.get('command_observation_ns')
        if row['reason']!='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED':
            if origin in plans:excluded[row['reason']]+=1
            continue
        admitted+=1;origins.add(origin)
        plan=plans.get(origin)
        branch=prefixes.get((origin,plan['selection']['action_index'])) if plan else None
        if branch is None or not branch['passed'] or row['simulator_ns']+20_000_000>origin+branch['matched_prefix_ms']*1_000_000:
            failures.append(dict(simulator_ns=row['simulator_ns'],origin=origin,reason='Executed selected candidate outside verified branch prefix'))
    return dict(schema='navigation_capability_oracle_executed_coverage.v1',passed=bool(admitted) and not failures and verification['passed'],
        selected_plans=len(plans),actually_admitted_plans=len(origins),admitted_20ms_intervals=admitted,failures=failures,
        non_admission_reasons=dict(excluded),selected_prefix_duration_ms_counts=dict(Counter(str(prefixes[(stamp,p['selection']['action_index'])]['matched_prefix_ms']) for stamp,p in plans.items())),
        all_branch_prefix_comparisons=len(prefixes),physics_steps=0,model_calls=0)

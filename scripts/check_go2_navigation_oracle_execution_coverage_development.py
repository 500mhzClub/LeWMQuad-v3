"""Require branch fidelity for every actually admitted C0 command interval."""
import argparse
from collections import Counter
import json
from pathlib import Path

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import PROTOCOL, save, sha


def check(root):
    base=Path(json.loads(PROTOCOL.read_text())['output_root']);output.install(base)
    assert root.resolve().is_relative_to((base/'runs').resolve())
    assert json.loads((root/'config.json').read_text())['controller']=='C0'
    prefix=json.loads((root/'oracle_prefix_check.json').read_text())
    comparisons={(r['observed_ns'],r['candidate']):r for r in prefix['comparisons']}
    plans={r['measured_ns']:r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r}
    requests=json.loads((root/'requests.json').read_text());failures=[];counts=Counter();overrides=Counter()
    for request in requests:
        origin=request.get('command_observation_ns')
        if request['reason']!='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED':
            if origin in plans:overrides[request['reason']]+=1
            continue
        if origin not in plans:
            failures.append(dict(time=request['simulator_ns'],reason='admitted command lacks selected plan'));continue
        candidate=plans[origin]['selection']['action_index'];row=comparisons[origin,candidate]
        required_ns=request['simulator_ns']+20_000_000-origin
        if not row['passed'] or required_ns>row['matched_prefix_ms']*1_000_000:
            failures.append(dict(time=request['simulator_ns'],observed_ns=origin,candidate=candidate,
                required_prefix_ms=required_ns/1_000_000,checked_prefix_ms=row['matched_prefix_ms']))
        counts[origin]+=1
    result=dict(schema='navigation_capability_oracle_executed_coverage.v1',passed=bool(counts) and not failures,
        selected_plans=len(plans),actually_admitted_plans=len(counts),admitted_20ms_intervals=sum(counts.values()),
        failures=failures,non_admission_reasons=dict(overrides),
        selected_prefix_duration_ms_counts=dict(Counter(comparisons[t,r['selection']['action_index']]['matched_prefix_ms'] for t,r in plans.items())),
        all_branch_prefix_comparisons=len(comparisons),
        note='Common pre-dispatch prefixes alone do not qualify an executed candidate; each admitted command interval must fall inside its selected branch comparison.',
        inputs={n:sha(root/n) for n in ['config.json','oracle_prefix_check.json','planning.json','requests.json']},
        physics_steps=0,model_calls=0)
    save(root/'oracle_execution_coverage.json',result);print(output.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);check(p.parse_args().root)

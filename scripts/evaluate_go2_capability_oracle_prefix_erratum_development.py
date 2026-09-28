"""Evaluator-side C0 executed-prefix qualification under the 28 September erratum.

Reads only preserved records: the frozen checker's rows, dispatch requests,
branch receipts and planning. Harness, oracle and frozen checker are unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from lewm import decision_headroom_json_v42_development as output

REPO = Path(__file__).resolve().parents[1]
ERRATUM = REPO/'docs/go2_navigation_capability_oracle_prefix_erratum_2026-09-28.json'
OUTPUT = 'oracle_prefix_erratum_evaluation.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(root):
    erratum = json.loads(ERRATUM.read_text())
    causes, ordinary, tick = erratum['dispatch_substitution_causes'], set(erratum['ordinary_dispatch_reasons']), erratum['tick_ns']
    check = json.loads((root/'oracle_prefix_check.json').read_text())
    requests = json.loads((root/'requests.json').read_text())
    receipts = json.loads((root/'model_calls.json').read_text())
    planning = json.loads((root/'planning.json').read_text())
    assert check['position_tolerance_m'] == erratum['position_tolerance_m']
    assert check['yaw_tolerance_degrees'] == erratum['yaw_tolerance_degrees']
    by_time = {r['simulator_ns']: r for r in requests}
    served_by_origin = {}
    for r in requests:
        served_by_origin.setdefault(r.get('command_observation_ns'), []).append(r)
    plans = {r['measured_ns']: r for r in planning if 'selection' in r}
    rows_by_origin = {}
    for row in check['comparisons']:
        rows_by_origin.setdefault(row['observed_ns'], []).append(row)
    stops = []
    receipt_origins = [r['observed_ns'] for r in receipts]
    if len(set(receipt_origins)) != len(receipt_origins) or set(receipt_origins) != set(rows_by_origin):
        stops.append('branch receipts and comparison origins differ')
    comparable = non_executed = 0
    max_position = max_yaw = 0.
    no_match = []
    for receipt in receipts:
        origin = receipt['observed_ns']
        rows = sorted(rows_by_origin.get(origin, []), key=lambda r: r['candidate'])
        if [r['candidate'] for r in rows] != list(range(6)):
            stops.append(f'{origin}: expected six candidate rows')
            continue
        first = by_time.get(origin)
        matched = [r for r in rows if r['matched_prefix_ms'] > 0]
        for r in rows:
            if r['matched_prefix_ms'] > 0:
                comparable += 1
                max_position = max(max_position, r['maximum_position_error_m'])
                max_yaw = max(max_yaw, r['maximum_yaw_error_deg'])
                if not r['passed']:
                    stops.append(f'{origin}/{r["candidate"]}: comparable prefix outside tolerance')
            else:
                # Independent confirmation from the retained first-tick branch command.
                branch_first = receipt['applied_commands'][r['candidate']][0]
                if first is not None and all(abs(a-b) <= 1e-7 for a, b in zip(first['applied_command'], branch_first)):
                    stops.append(f'{origin}/{r["candidate"]}: zero-length row but first tick matches')
                if matched:
                    non_executed += 1
        if matched:
            continue
        if first is None:
            stops.append(f'{origin}: no matching branch and no dispatch record')
            continue
        reason = first['reason']
        if reason in ordinary or reason not in causes:
            stops.append(f'{origin}: unexplained no-match (first tick {reason})')
            continue
        sequence = []
        for k in range(40):
            row = by_time.get(origin+k*tick)
            if row is None:
                break
            if row['reason'] in ordinary and k:
                break
            sequence.append(row['reason'])
        plan = plans.get(origin)
        no_match.append(dict(observed_ns=origin, cause=causes[reason], first_tick_reason=reason,
            substitution_sequence=dict(Counter(sequence)), substitution_ticks=len(sequence),
            selected_action=None if plan is None else plan['action'],
            served_requests=len(served_by_origin.get(origin, [])),
            served_reasons=dict(Counter(r['reason'] for r in served_by_origin.get(origin, [])))))
    substituted = Counter(r['reason'] for r in requests if r['reason'] in causes)
    vetoed = []
    for origin in sorted(rows_by_origin):
        served = served_by_origin.get(origin, [])
        first = by_time.get(origin)
        kinds = {causes.get(r['reason']) for r in served} | ({causes.get(first['reason'])} if first else set())
        if kinds & {'veto', 'missing_current_observation'}:
            plan = plans.get(origin)
            vetoed.append(dict(observed_ns=origin, selected_action=None if plan is None else plan['action'],
                reasons=dict(Counter(r['reason'] for r in served))))
    record = dict(schema='navigation_capability_oracle_prefix_erratum_evaluation.v1',
        assignment=root.name, passed_under_erratum=not stops, stops=stops,
        frozen_checker_passed=check['passed'], decisions=len(receipts), branch_rows=len(check['comparisons']),
        comparable_rows=comparable, comparable_maximum_position_error_m=max_position,
        comparable_maximum_yaw_error_deg=max_yaw, non_executed_rows_with_other_match=non_executed,
        no_matching_branch_decisions=len(no_match), no_matching_branch_by_cause=dict(Counter(r['cause'] for r in no_match)),
        no_matching_branch_details=no_match,
        dispatch_substitution_ticks=dict(substituted),
        dispatch_substitution_ticks_by_cause=dict(Counter(causes[r['reason']] for r in requests if r['reason'] in causes)),
        vetoed_selections=len(vetoed), vetoed_movement_selections=sum(v['selected_action'] not in (None, 'hold') for v in vetoed),
        vetoed_selection_details=vetoed,
        erratum_sha256=sha(ERRATUM), evaluator_sha256=sha(__file__),
        input_sha256={name: sha(root/name) for name in ('oracle_prefix_check.json', 'requests.json', 'model_calls.json', 'planning.json', 'result.json')})
    return record


def write(root):
    path = root/OUTPUT
    record = evaluate(root)
    with path.open('x') as stream:
        json.dump(record, stream, separators=(',', ':'))
        stream.write('\n')
    assert json.loads(path.read_text()) == json.loads(json.dumps(record))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    protocol = json.loads((REPO/'docs/go2_navigation_capability_completed_support_v4_2026-09-27.json').read_text())
    base = Path(protocol['output_root'])
    assert args.root.resolve().is_relative_to((base/'runs').resolve())
    output.install(base)
    r = evaluate(args.root) if args.dry_run else write(args.root)
    print(json.dumps({k: v for k, v in r.items() if not k.endswith('_details') and k != 'input_sha256'}))

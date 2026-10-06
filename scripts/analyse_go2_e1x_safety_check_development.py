"""E1 exploratory-arm safety gate: results (declared 30 Sep 2026, commit a435cbcc).

Read-only over the preserved records of `cohorts/e1x_safety_check`. No sealed-set access.

**Gate:** C3-v3 and C4-v3 both have zero disallowed contacts and zero hard violations.

Reported per controller (C1 is a reference):
- round trips;
- disallowed contacts;
- hard violations, confirmed and unresolved;
- operating-margin violations (the 20-mm threshold), confirmed and unresolved;
- the minimum separation lower bound;
- selected-hold rates per leg, pooled and as a per-mission mean;
- hold decisions by the reader's category;
- every 480-s timeout labelled by the frozen mechanism rules, with the latch-active fraction
  of its final window;
- other failures with their source error (for example, pose loss).
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.diagnose_go2_capability_validation_timeouts_development import diagnose, FINAL_WINDOW_S

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
PHASES = ('OUTBOUND', 'RETURN')


def mission(run):
    ev = json.loads((run/'episode_evaluation.json').read_text())
    result = json.loads((run/'result.json').read_text())
    row = dict(run=run.name, round_trip=ev['round_trip_success'], contacts=ev['disallowed_contact_samples'],
               hard=ev['safety']['hard']['confirmed_violation_samples'], hard_unresolved=ev['safety']['hard']['unresolved_sampled_samples'],
               operating=ev['safety']['operating']['confirmed_violation_samples'],
               operating_unresolved=ev['safety']['operating']['unresolved_sampled_samples'],
               minimum_separation_lower_m=ev['safety']['hard']['minimum_separation_lower_m'],
               holds={p: ev['stall_by_phase'].get(p, {}).get('holds', 0) for p in PHASES},
               selected={p: ev['stall_by_phase'].get(p, {}).get('selected_plans', 0) for p in PHASES},
               hold_categories=ev['hold_categories'], source_error=ev['source_error'], simulated_s=result['simulated_s'],
               wall_s=result['wall_s'], bytes=sum(f.stat().st_size for f in run.rglob('*') if f.is_file()), mechanism=None)
    if result['policy_steps'] >= 24000 and not ev['round_trip_success']:
        d = diagnose(run)
        late = [r for r in json.loads((run/'planning.json').read_text())
                if 'selection' in r and r['measured_ns']/1e9-1.5 >= d['terminal_s']-FINAL_WINDOW_S]
        row.update(mechanism=d['mechanism'], remaining_at_480s_m=d['remaining_shortest_path_at_480s_m'],
                   latch_active_final_window=sum(bool((r['selection'].get('clearance_turn') or {}).get('active')) for r in late)/max(1, len(late)))
    return row


def summary(rows):
    out = dict(missions=len(rows), round_trips=sum(r['round_trip'] for r in rows), disallowed_contacts=sum(r['contacts'] for r in rows),
               hard_violations=sum(r['hard'] for r in rows), hard_unresolved=sum(r['hard_unresolved'] for r in rows),
               operating_margin_violations=sum(r['operating'] for r in rows), operating_unresolved=sum(r['operating_unresolved'] for r in rows),
               minimum_separation_lower_m=min(r['minimum_separation_lower_m'] for r in rows),
               timeouts_by_frozen_mechanism=dict(Counter(r['mechanism'] for r in rows if r['mechanism'])),
               latch_active_by_timeout={r['run']: round(r['latch_active_final_window'], 3) for r in rows if r['mechanism']},
               other_failures={r['run']: r['source_error'] for r in rows if not r['round_trip'] and not r['mechanism']},
               hold_decisions_by_category=dict(sum((Counter(r['hold_categories']) for r in rows), Counter())),
               mean_wall_s=float(np.mean([r['wall_s'] for r in rows])), median_wall_s=float(np.median([r['wall_s'] for r in rows])),
               mean_bytes=float(np.mean([r['bytes'] for r in rows])))
    for p in PHASES:
        holds, selected = sum(r['holds'][p] for r in rows), sum(r['selected'][p] for r in rows)
        rates = [r['holds'][p]/r['selected'][p] for r in rows if r['selected'][p]]
        out[f'{p.lower()}_hold_rate'] = dict(pooled=holds/selected if selected else None, holds=holds, selected=selected,
                                              per_mission_mean=float(np.mean(rates)) if rates else None)
    return out


def main():
    output.install(BASE)
    cohort = json.loads((BASE/'cohorts/e1x_safety_check/result.json').read_text())
    groups = {}
    for arm in ('C1', 'C3', 'C4'):
        rows = [mission(BASE/'runs'/r['assignment']) for r in cohort['rows'] if r['controller'] == arm]
        groups[arm] = dict(summary=summary(rows), missions=rows) if rows else None
    gate = all(groups[a] and groups[a]['summary']['disallowed_contacts'] == 0 and groups[a]['summary']['hard_violations'] == 0
               and groups[a]['summary']['missions'] == 10 for a in ('C3', 'C4'))
    result = dict(schema='e1x_safety_gate.v1', declaration_commit='a435cbcc', cohort_complete=cohort['complete'], cohort_stops=cohort['stops'],
                  gate_passed=bool(gate and cohort['complete'] and not cohort['stops']), versions=dict(C1='unchanged', C3='C3-v3', C4='C4-v3'),
                  groups=groups, analyser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    owner.save(BASE/'cohorts/e1x_safety_check/gate.json', result)
    print(json.dumps(dict(gate_passed=result['gate_passed'], stops=cohort['stops'],
                          summary={a: g['summary'] if g else None for a, g in groups.items()}), indent=1))


if __name__ == '__main__':
    main()

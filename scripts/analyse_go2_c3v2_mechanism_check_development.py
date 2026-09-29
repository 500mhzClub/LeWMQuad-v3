"""Mechanism check: C3-v2 on the fresh check versus C3-v1 on validation (Andrew, 29 Sep 2026).

A mechanism check on 10 fresh mazes, not a result. The two sets are different mazes and are
not paired. C1 on both sets is shown as a reference for how much the maze sets alone differ.
Read-only over preserved records.

1. Hold rates. Selected holds over selected plans per leg, from the frozen reader's
   `stall_by_phase`, pooled over missions (the capability report's definition) and as a
   per-mission mean.
2. Deadlocks and hold stalls by trap.
   - Every 480-s timeout is labelled with the frozen mechanism rules
     (`diagnose_go2_capability_validation_timeouts_development.diagnose`). The final-window
     fraction of decisions with the latched clearance turn active is added, as trap 3 is
     reported both ways.
   - Hold decisions are totalled by the reader's hold category.
3. Predicted forward travel from rest in closed loop. These are decisions whose selected
   action is forward, with zero applied commands over the preceding 1.0 s. The acceptance
   rest rule is used, and the prediction is the forward candidate's 800-ms predicted XY
   translation.
   - True travel is the physics-trace XY translation in the decision body frame over the
     same 800 ms.
   - The primary ratio uses only decisions whose executed applied tape equals the predicted
     candidate tape for all eight steps, so prediction and truth refer to the same commands.
   - The ratio over all forward-from-rest decisions is secondary.
   - Also secondary: the same ratio for all tape-matched forward decisions, from rest or not,
     because decisions from rest are rare (about one per mission).
   - C1's prediction slot logs a placeholder, not its command-history forecast, so this item
     covers C3, with C4 as a reference.
4. The median logged 800-ms predicted translation per candidate action over all decisions
   (C3 and C4, both versions). This checks which readout was active in each set.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from lewm.c3v2_offline_pipeline_development import Recording
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts.diagnose_go2_capability_validation_timeouts_development import diagnose, FINAL_WINDOW_S

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
GROUPS = {'C3-v2 fresh check': 'c3v2_check_C3_chk*_ep0_attempt001', 'C3-v1 validation': 'v4_completed_support_validation_C3_val*_ep0_attempt001',
          'C1 fresh check (reference)': 'c3v2_check_C1_chk*_ep0_attempt001', 'C1 validation (reference)': 'v4_completed_support_validation_C1_val*_ep0_attempt001',
          'C4-v2 fresh check (reference)': 'c3v2_check_C4_chk*_ep0_attempt001', 'C4-v1 validation (reference)': 'v4_completed_support_validation_C4_val*_ep0_attempt001'}
PREDICTION_SETS = {'C3-v2 fresh check': 'c3v2_check_C3_chk*_ep0_attempt001', 'C3-v1 validation': 'v4_completed_support_validation_C3_val*_ep0_attempt001',
                   'C4-v2 fresh check': 'c3v2_check_C4_chk*_ep0_attempt001', 'C4-v1 validation': 'v4_completed_support_validation_C4_val*_ep0_attempt001'}
ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
PHASES = ('OUTBOUND', 'RETURN')
FORWARD = 1  # canonical candidate order: hold, forward, left_arc, right_arc, left_turn, right_turn


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_decisions(run):
    if '_C1_' in run.name:
        return []
    calls = json.loads((run/'model_calls.json').read_text())
    plans = {r['measured_ns']: r for r in json.loads((run/'planning.json').read_text()) if 'selection' in r}
    native = run/'native'
    recording = Recording(native/'policy_histories.npz', native/'policy_observations.json', None, native/'physics_trace.npz')
    meta = json.loads((native/'in_memory_camera_observations.json').read_text())['frames']
    rows = []
    for call in calls:
        plan = plans.get(call['observed_ns'])
        if plan is None or plan['action'] != 'forward':
            continue
        frame = (call['observed_ns']-1_500_000_000)//100_000_000
        if frame+8 >= len(meta) or not recording.valid[frame].all():
            continue
        try:
            executed = recording.executed_tape(frame)
        except ValueError:
            continue
        candidate = np.asarray(call['applied_commands'][FORWARD], np.float32)
        predicted = float(np.linalg.norm(np.asarray(call['motion_xy_yaw'][FORWARD][7][:2])))
        true = float(np.linalg.norm(recording.targets(frame, meta)[7, :2]))
        rows.append(dict(frame=int(frame), predicted_m=predicted, true_m=true, tape_matched=bool(np.array_equal(executed, candidate)),
                         from_rest=bool(np.all(recording.values[frame][-10:] == 0))))
    return rows


def run_row(run):
    ev = json.loads((run/'episode_evaluation.json').read_text())
    result = json.loads((run/'result.json').read_text())
    row = dict(run=run.name, round_trip=ev['round_trip_success'],
               holds={p: ev['stall_by_phase'].get(p, {}).get('holds', 0) for p in PHASES},
               selected={p: ev['stall_by_phase'].get(p, {}).get('selected_plans', 0) for p in PHASES},
               hold_categories=ev['hold_categories'], mechanism=None, latch_active_final_window=None)
    if result['policy_steps'] >= 24000 and not ev['round_trip_success']:
        diagnosis = diagnose(run)
        end = diagnosis['terminal_s']
        late = [r for r in json.loads((run/'planning.json').read_text())
                if 'selection' in r and r['measured_ns']/1e9-1.5 >= end-FINAL_WINDOW_S]
        row.update(mechanism=diagnosis['mechanism'], remaining_at_480s_m=diagnosis['remaining_shortest_path_at_480s_m'],
                   latch_active_final_window=sum(bool((r['selection'].get('clearance_turn') or {}).get('active')) for r in late)/max(1, len(late)))
    row['forward'] = forward_decisions(run)
    return row


def summary(rows):
    out = dict(missions=len(rows), round_trips=sum(r['round_trip'] for r in rows))
    for p in PHASES:
        holds, selected = sum(r['holds'][p] for r in rows), sum(r['selected'][p] for r in rows)
        rates = [r['holds'][p]/r['selected'][p] for r in rows if r['selected'][p]]
        out[f'{p.lower()}_hold_rate'] = dict(pooled=holds/selected if selected else None, holds=holds, selected=selected,
                                              per_mission_mean=float(np.mean(rates)) if rates else None)
    out['timeouts_by_frozen_mechanism'] = dict(Counter(r['mechanism'] for r in rows if r['mechanism']))
    out['latch_active_final_window_by_timeout'] = {r['run']: round(r['latch_active_final_window'], 3) for r in rows if r['mechanism']}
    out['hold_decisions_by_category'] = dict(sum((Counter(r['hold_categories']) for r in rows), Counter()))
    ratio = lambda ds: float(np.median([d['predicted_m']/d['true_m'] for d in ds])) if ds else None
    mm = lambda ds, k: float(np.median([d[k] for d in ds]))*1000 if ds else None
    forward = [d for r in rows for d in r['forward'] if d['true_m'] > 0]
    rest = [d for d in forward if d['from_rest']]
    matched = [d for d in rest if d['tape_matched']]
    every = [d for d in forward if d['tape_matched']]
    if rows and '_C1_' in rows[0]['run']:
        out['forward_from_rest_800ms'] = 'not applicable: C1 logs a placeholder, not its command-history forecast'
        return out
    out['forward_from_rest_800ms'] = dict(
        decisions=len(rest), tape_matched=len(matched), median_ratio_tape_matched=ratio(matched),
        median_predicted_mm_tape_matched=mm(matched, 'predicted_m'), median_true_mm_tape_matched=mm(matched, 'true_m'),
        median_ratio_all_from_rest_secondary=ratio(rest),
        all_forward_tape_matched_secondary=dict(decisions=len(every), median_ratio=ratio(every),
                                               median_predicted_mm=mm(every, 'predicted_m'), median_true_mm=mm(every, 'true_m')))
    return out


def logged_predictions(pattern):
    motions = []
    for run in sorted((BASE/'runs').glob(pattern)):
        path = run/'model_calls.json'
        calls = json.loads(path.read_text()) if path.exists() else []
        if calls:
            motions.append(np.asarray([c['motion_xy_yaw'] for c in calls])[:, :, 7, :2])
    if not motions:
        return None
    translation = np.linalg.norm(np.concatenate(motions), axis=2)
    return dict(missions=len(motions), decisions=len(translation),
                median_800ms_translation_mm={a: float(np.median(translation[:, i]))*1000 for i, a in enumerate(ACTIONS)})


def main():
    output.install(BASE)
    report = dict(schema='c3v2_mechanism_check.v1',
                  label='Mechanism check on 10 fresh mazes, not a result; the fresh-check and validation mazes differ and are not paired',
                  groups={}, runs={})
    for name, pattern in GROUPS.items():
        rows = [run_row(run) for run in sorted((BASE/'runs').glob(pattern)) if (run/'episode_evaluation.json').exists()]
        report['groups'][name] = summary(rows)
        report['runs'][name] = [{k: v for k, v in r.items() if k != 'forward'} | dict(forward_decisions=len(r['forward']),
                                forward_from_rest_decisions=sum(d['from_rest'] for d in r['forward'])) for r in rows]
    report['logged_predictions'] = {name: logged_predictions(pattern) for name, pattern in PREDICTION_SETS.items()}
    report['analyser_sha256'] = sha(__file__)
    root = BASE/'analysis/c3v2_mechanism_check'
    root.mkdir(parents=True, exist_ok=True)
    count = report['groups']['C3-v2 fresh check']['missions']
    owner.save(root/f'result_{count}of10.json', report)
    print(json.dumps(dict(groups=report['groups'], logged_predictions=report['logged_predictions']), indent=1))


if __name__ == '__main__':
    main()

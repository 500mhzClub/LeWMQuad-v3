"""Merge the gap-diagnosis scorer subsets and summarise the like-for-like by tape shape.

Diagnosis on the check mazes only. The scorer ran as three parallel processes on disjoint
fresh-check missions (`score_go2_c3v2_gap_pipeline_development.py --only ... --out
pipeline_part_i.json`, unchanged code). This script combines their pipeline-test results.

**Forward decisions differ in what was executed.** Most executed forward candidates on C4
missions are single 100-ms terminal pulses at the goal (tape 000F0000, about 1 mm of true
travel), where a predicted/true ratio is meaningless. So each forward decision is classed by
its forward candidate's tape:
- **full forward:** at least four 100-ms forward steps;
- **pulse:** fewer.
Each class is split into from rest (zero applied commands in the preceding 1.0 s) and moving.
The ratio is reported for full-forward commands, and the XY error for both classes, with
counts everywhere.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
ROOT = BASE/'c3v2_gap_diagnosis_v1'
MODELS = ('C3_v1', 'C3_v2', 'C4_v1', 'C4_v2', 'C1')
FORWARD = 1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_steps(rows):
    calls = {}
    for r in rows:
        if r['run'] not in calls:
            calls[r['run']] = {c['observed_ns']: c for c in json.loads((BASE/'runs'/r['run']/'model_calls.json').read_text())}
        tape = np.asarray(calls[r['run']][1_500_000_000+100_000_000*r['frame']]['applied_commands'][FORWARD])
        r['forward_steps'] = int((tape[:, 0] > 0).sum())
        r['tape'] = ''.join('F' if x[0] > 0 else ('T' if x[2] != 0 else '0') for x in tape)
    return rows


def stats(subset, name, ratio):
    usable = [r for r in subset if r[name] is not None]
    if not usable:
        return None
    p = np.asarray([r[name] for r in usable])
    t = np.asarray([r['true_800_xy'] for r in usable])
    out = dict(n=len(usable), median_predicted_mm=float(np.median(np.linalg.norm(p, axis=1)))*1000,
               median_xy_error_mm=float(np.median(np.linalg.norm(p-t, axis=1)))*1000,
               rmse_xy_mm=float(np.sqrt(np.mean(np.sum((p-t)**2, axis=1))))*1000)
    if ratio:
        out['median_ratio'] = float(np.median(np.linalg.norm(p, axis=1)/np.linalg.norm(t, axis=1)))
    return out


def summarise(rows):
    out = {}
    for kind, keep_kind in (('full_forward', lambda r: r['forward_steps'] >= 4), ('pulse', lambda r: r['forward_steps'] < 4)):
        for split, keep in (('from_rest', lambda r: r['from_rest']), ('moving', lambda r: not r['from_rest']), ('all', lambda r: True)):
            subset = [r for r in rows if keep_kind(r) and keep(r)]
            out[f'{kind}/{split}'] = dict(
                decisions=len(subset), from_C3_missions=sum(r['source_arm'] == 'C3' for r in subset),
                from_C4_missions=sum(r['source_arm'] == 'C4' for r in subset),
                tapes={t: sum(r['tape'] == t for r in subset) for t in sorted({r['tape'] for r in subset})},
                median_true_mm=float(np.median([np.linalg.norm(r['true_800_xy']) for r in subset]))*1000 if subset else None,
                models={name: stats(subset, name, ratio=kind == 'full_forward') for name in MODELS})
    return out


def main():
    output.install(BASE)
    parts = sorted(ROOT.glob('pipeline_part_*.json'))
    loaded = [json.loads(p.read_text()) for p in parts]
    order = [d.name for pattern in ('c3v2_check_C3_chk*_ep0_attempt001', 'c3v2_check_C4_chk*_ep0_attempt001') for d in sorted((BASE/'runs').glob(pattern))]
    per_run = sorted((r for part in loaded for r in part['pipeline_test']['runs']), key=lambda r: order.index(r['run']))
    complete = [r['run'] for r in per_run] == order
    divergences = sorted((part['pipeline_test']['first_divergence'] for part in loaded if part['pipeline_test']['first_divergence']),
                         key=lambda d: (order.index(d['run']), d['decision']))
    passed = complete and all(part['pipeline_test']['passed'] for part in loaded)
    rows = forward_steps([r for part in loaded for r in part['forward_executed_rows']])
    result = dict(schema='c3v2_gap_pipeline_and_like_for_like.v1', label='Diagnosis on check mazes; validation and sealed sets untouched',
                  tolerance=loaded[0]['tolerance'], every_mission_scored_once=complete,
                  pipeline_test=dict(passed=passed, runs=per_run, first_divergence=divergences[0] if divergences else None,
                                     max_abs_xy_m=max(r['max_abs_xy_m'] for r in per_run), max_abs_yaw_rad=max(r['max_abs_yaw_rad'] for r in per_run),
                                     compared_values=sum(r['compared_values'] for r in per_run), decisions=sum(r['decisions'] for r in per_run),
                                     context_unavailable_offline=sum(r['context_unavailable_offline'] for r in per_run)),
                  like_for_like=summarise(rows) if passed else 'not interpreted: pipeline test failed or incomplete',
                  forward_executed_rows=rows, parts={p.name: sha(p) for p in parts},
                  scorer_sha256=sorted({part['scorer_sha256'] for part in loaded}), summariser_sha256=sha(__file__))
    owner.save(ROOT/'pipeline_and_like_for_like.json', result)
    print(json.dumps(dict(pipeline_test={k: v for k, v in result['pipeline_test'].items() if k != 'runs'},
                          like_for_like=result['like_for_like']), indent=1))


if __name__ == '__main__':
    main()

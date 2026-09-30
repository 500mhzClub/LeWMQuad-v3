"""Offline acceptance of C3-v3 against the pre-declared criteria (30 Sep 2026, commit ec2e34c9, section 5).

Everything is scored on the deployed computation: the three-frame encoder batch, then the
frozen predictor with the executed applied tape, then the readout (the C3-v2 evaluator's
`evaluate_contexts`, verified exact on 14,397 fresh-check decisions). Each criterion compares
C3-v3 with C3-v2.

- P (primary): closed-loop moving decisions on the 6 `onpolicy_heldout` C1 missions. A
  decision qualifies if its executed tape equals the forward candidate's tape and has at least
  four forward steps, with non-zero applied commands in the preceding 1.0 s.
  - P1: median 800-ms translation ratio in [0.75, 1.25].
  - P2: median 800-ms XY error at most 50% of C3-v2's.
  - At least 30 decisions must qualify.
- R1 (from rest): on the 72 offline held-out rest-start windows, the median 800-ms translation
  ratio is in [0.75, 1.25].
- N (no regression beyond 5%):
  - N1: held-out groups, XY and yaw RMSE at 500 and 800 ms;
  - N2: the transfer population, XY and yaw RMSE at 500 and 700 ms;
  - N3: in-place-turn spurious translation at most 10 mm.
C4-v3 is scored against C4-v2 with the same measures; reported only. C1 is the logged
command-history forecast.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from lewm.c3v3_onpolicy_data_development import recording_for
from lewm.navigation_capability_supervised_development import DirectMotionPredictor
from scripts import evaluate_go2_c3v2_acceptance_development as v2
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts import train_go2_all_motion_horizon_readout_development as previous

BASE = v2.BASE
OUT = BASE/'c3v3_acceptance_v1'
PREDECLARATION = 'docs/go2_navigation_c3v3_onpolicy_round_predeclaration_2026-09-30.md'
READOUTS = dict(C3_v2=BASE/'c3v2_readout_fit_v1/readout_v2_final.pt', C3_v3=BASE/'c3v3_readout_fit_v1/readout_v3_final.pt')
C4S = dict(C4_v1=BASE/'c4_fit_attempt002/direct_final.pt', C4_v2=BASE/'c4v2_fit_v1/direct_v2_final.pt', C4_v3=BASE/'c4v3_fit_v1/direct_v3_final.pt')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closed_loop(motions, decisions, c1):
    t = np.asarray([d['targets'][7][:2] for d in decisions])
    out = dict(decisions=len(decisions), median_true_mm=float(np.median(np.linalg.norm(t, axis=1)))*1000 if len(t) else None)
    for name, pred in list(motions.items())+[('C1', c1)]:
        p = np.asarray(pred)[:, 7, :2] if len(decisions) else np.zeros((0, 2))
        out[name] = dict(median_ratio=float(np.median(np.linalg.norm(p, axis=1)/np.linalg.norm(t, axis=1))) if len(t) else None,
                         median_xy_error_mm=float(np.median(np.linalg.norm(p-t, axis=1)))*1000 if len(t) else None)
    return out


def main():
    output.install(BASE)
    OUT.mkdir(exist_ok=False)
    model = owner.source.load_dense_navigation_model('action', readout_arm='maze_view_maze_data')
    readouts = dict(C3_v1=model.readout)
    for name, path in READOUTS.items():
        r = previous.prior.previous.load('mixed_data')
        r.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        readouts[name] = r.cuda().eval().requires_grad_(False)
    head = torch.load(json.loads(Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json').read_text())['controllers']['C3']['head_binding']['path'],
                      map_location='cpu', weights_only=False)['model_state_dict']
    c4s = {}
    for name, path in C4S.items():
        c4 = DirectMotionPredictor(head['target_mean'], head['target_scale'])
        c4.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        c4s[name] = c4.cuda().eval()
    with wall.job(BASE, 'C3-v3 round: offline acceptance'):
        # P: closed-loop held-out decisions.
        decisions = json.loads((BASE/'c3v3_data_v1/heldout_onpolicy_decisions.json').read_text())
        full = [d for d in decisions if d['forward_executed'] and d['forward_steps'] >= 4]
        sets = dict(moving=[d for d in full if not d['from_rest']], from_rest=[d for d in full if d['from_rest']])
        recordings, plans = {}, {}
        closed = {}
        for split, subset in sets.items():
            contexts, c1 = [], []
            for d in subset:
                recordings.setdefault(d['run'], recording_for(d))
                if d['run'] not in plans:
                    plans[d['run']] = {r['measured_ns']: r for r in json.loads((Path(d['directory'])/'planning.json').read_text()) if 'selection' in r}
                contexts.append((recordings[d['run']], d['frame']))
                c1.append(np.asarray(plans[d['run']][d['observed_ns']]['motion_correction']['command_history_forecast_xy_yaw'])[1])
            motions = v2.evaluate_contexts(model, readouts, c4s, contexts) if contexts else {n: [] for n in list(readouts)+list(c4s)}
            closed[split] = closed_loop(motions, subset, c1)
        # R and N: offline held-out groups and the transfer population (the C3-v2 evaluator's definitions).
        rows = json.loads((BASE/'c3v2_data_v1/heldout_samples.json').read_text())
        cache, contexts = {}, []
        for r in rows:
            cache.setdefault(r['directory'], v2.Recording.training_case(r['directory']))
            contexts.append((cache[r['directory']], r['frame']))
        motions = v2.evaluate_contexts(model, readouts, c4s, contexts)
        true = np.asarray([r['targets'] for r in rows], np.float32)
        past = np.asarray([r['past_applied'] for r in rows])
        tape = np.asarray([r['applied_tape'] for r in rows])
        rest = np.all(past[:, -10:, :] == 0, axis=(1, 2)) & np.any(tape[:, :, 0] != 0, axis=1)
        turn = ~np.any(tape[:, :, 0] != 0, axis=1) & np.any(tape[:, :, 2] != 0, axis=1)
        groups = {}
        for group, mask in (('rest_start', rest), ('in_place_turn', turn), ('other', ~rest & ~turn)):
            m = {}
            for name, pred in motions.items():
                m[name] = {f'{ms}ms': dict(xy_rmse_m=v2.rmse(pred[mask, h], true[mask, h], [0, 1]), yaw_rmse_deg=v2.yaw_rmse(pred[mask, h], true[mask, h]))
                           for ms, h in ((500, 4), (800, 7))}
                p, t = pred[mask, 7], true[mask, 7]
                m[name]['800ms'].update(median_translation_ratio=float(np.median(np.linalg.norm(p[:, :2], axis=1)/np.maximum(np.linalg.norm(t[:, :2], axis=1), 1e-9))),
                                        median_excess_translation_m=float(np.median(np.linalg.norm(p[:, :2], axis=1)-np.linalg.norm(t[:, :2], axis=1))))
            groups[group] = dict(windows=int(mask.sum()), metrics=m)
        windows = json.loads((v2.TRANSFER/'transfer_targets.json').read_text())
        tcache, tcontexts, index = {}, [], {}
        for w in windows:
            key = (w['case'], w['frame'])
            if key not in index:
                directory = v2.TRANSFER/f"case_{w['case']:02d}"
                tcache.setdefault(directory, v2.Recording.training_case(directory))
                index[key] = len(tcontexts)
                tcontexts.append((tcache[directory], w['frame']))
        tmotions = v2.evaluate_contexts(model, readouts, c4s, tcontexts)
        transfer = defaultdict(dict)
        for ms in (500, 700):
            sel = [w for w in windows if w['horizon_ms'] == ms]
            t = np.asarray([w['actual'] for w in sel], np.float32)
            for name in tmotions:
                p = np.asarray([tmotions[name][index[(w['case'], w['frame'])], ms//100-1] for w in sel])
                transfer[f'{ms}ms'][name] = dict(xy_rmse_m=v2.rmse(p, t, [0, 1]), yaw_rmse_deg=v2.yaw_rmse(p, t))

        def at(group, name, ms, key):
            return groups[group]['metrics'][name][f'{ms}ms'][key]

        def measures(old, new):
            moving = closed['moving']
            return {
                'P_decidable_at_least_30': moving['decisions'] >= 30,
                'P1_moving_ratio_in_0.75_1.25': moving['decisions'] >= 30 and 0.75 <= moving[new]['median_ratio'] <= 1.25,
                'P2_moving_xy_error_at_most_half_previous': moving['decisions'] >= 30 and moving[new]['median_xy_error_mm'] <= .5*moving[old]['median_xy_error_mm'],
                'R1_offline_rest_ratio_in_0.75_1.25': 0.75 <= at('rest_start', new, 800, 'median_translation_ratio') <= 1.25,
                'N1_heldout_groups_no_regression_5pct': all(at(g, new, ms, k) <= 1.05*at(g, old, ms, k) for g in groups for ms in (500, 800)
                                                           for k in ('xy_rmse_m', 'yaw_rmse_deg')),
                'N2_transfer_no_regression_5pct': all(transfer[f'{ms}ms'][new][k] <= 1.05*transfer[f'{ms}ms'][old][k] for ms in (500, 700)
                                                      for k in ('xy_rmse_m', 'yaw_rmse_deg')),
                'N3_turn_spurious_translation_at_most_10mm': at('in_place_turn', new, 800, 'median_excess_translation_m') <= .010,
            }
        criteria = measures('C3_v2', 'C3_v3')
        result = dict(schema='c3v3_offline_acceptance.v1', passed=all(criteria.values()), criteria=criteria,
                      c4_report_only=dict(gating=False, comparison='C4-v3 against C4-v2 with the same measures', measures=measures('C4_v2', 'C4_v3')),
                      closed_loop_heldout=closed, heldout_groups=groups, transfer=dict(transfer),
                      predeclaration_sha256=sha(PREDECLARATION), predeclaration_commit='ec2e34c9',
                      checkpoints_sha256={n: sha(p) for n, p in list(READOUTS.items())+list(C4S.items())},
                      evaluator_sha256=sha(__file__), heldout_contexts=len(rows), transfer_windows=len(windows))
        owner.save(OUT/'result.json', result)
        print(json.dumps(dict(passed=result['passed'], criteria=criteria, closed_loop_heldout=closed,
                              c4_report_only=result['c4_report_only']['measures']), indent=1), flush=True)


if __name__ == '__main__':
    main()

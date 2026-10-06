"""Offline acceptance of C3-v2 against the pre-declared criteria (29 Sep 2026, §5).

Run-time path (three-frame encoder batch, frozen predictor with the executed applied tape,
readout) on the held-out rest/turn recordings and on the 240-window development transfer
population. C3-v1 and C3-v2 share the predicted features; only the readout differs.
C4-v1 and C4-v2 are scored with the same measures on the same windows (C4-v2 against C4-v1,
Amendment 1): reported only, never gating.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from lewm import decision_headroom_json_v42_development as output
from lewm import navigation_capability_active_wall_development as wall
from lewm.c3v2_offline_pipeline_development import Recording, pooled_features
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.navigation_capability_supervised_development import DirectMotionPredictor
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
from scripts import train_go2_all_motion_horizon_readout_development as previous

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
TRANSFER = BASE.parent/'go2_maze_view_transfer_v1_attempt_001'
OUT = BASE/'c3v2_acceptance_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmse(pred, true, cols):
    d = pred[:, cols]-true[:, cols]
    return float(np.sqrt(np.mean(np.sum(d**2, axis=1)))) if len(d) else None


def yaw_rmse(pred, true):
    d = np.arctan2(np.sin(pred[:, 2]-true[:, 2]), np.cos(pred[:, 2]-true[:, 2]))
    return float(np.degrees(np.sqrt(np.mean(d**2)))) if len(d) else None


@torch.inference_mode()
def evaluate_contexts(model, readouts, c4s, contexts):
    """contexts: list of (recording, frame, stats) -> per-model (N,8,3) motions."""
    device = next(model.predictor.parameters()).device
    out = {name: [] for name in list(readouts)+list(c4s)}
    for recording, frame in contexts:
        native = recording.native(frame)
        applied = recording.executed_tape(frame)
        current, predicted = pooled_features(model, native, applied)
        for name, readout in readouts.items():
            out[name].append(np.stack([readout(current, predicted[h]).float().cpu().numpy()[0] for h in range(1, 9)]))
        if c4s:
            feats = pool_tokens(F.layer_norm(model.encoder.tokens(native['pixels'].to(device)).float(), (1024,)))[None]
            past = native['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
            past = ((past-model.control_mean)/model.control_std)[None]
            acts = torch.from_numpy(applied[None][:, :, [0, 2]]).to(device)
            for name, c4 in c4s.items():
                out[name].append(np.stack([c4(feats, past, acts, torch.tensor([h], device=device)).float().cpu().numpy()[0] for h in range(1, 9)]))
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    output.install(BASE)
    OUT.mkdir(exist_ok=False)
    model = owner.source.load_dense_navigation_model('action', readout_arm='maze_view_maze_data')
    v2_state = torch.load(BASE/'c3v2_readout_fit_v1/readout_v2_final.pt', map_location='cpu', weights_only=False)
    readout_v2 = previous.prior.previous.load('mixed_data')
    readout_v2.load_state_dict(v2_state['model_state_dict'])
    readouts = dict(C3_v1=model.readout, C3_v2=readout_v2.cuda().eval().requires_grad_(False))
    head = torch.load(json.loads(Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json').read_text())['controllers']['C3']['head_binding']['path'],
                      map_location='cpu', weights_only=False)['model_state_dict']
    c4s = {}
    for name, path in (('C4_v1', BASE/'c4_fit_attempt002/direct_final.pt'), ('C4_v2', BASE/'c4v2_fit_v1/direct_v2_final.pt')):
        c4 = DirectMotionPredictor(head['target_mean'], head['target_scale'])
        c4.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state_dict'])
        c4s[name] = c4.cuda().eval()
    with wall.job(BASE, 'C3-v2 offline acceptance'):
        # Held-out rest/turn recordings.
        rows = json.loads((BASE/'c3v2_data_v1/heldout_samples.json').read_text())
        cache = {}
        contexts = []
        for r in rows:
            cache.setdefault(r['directory'], Recording.training_case(r['directory']))
            contexts.append((cache[r['directory']], r['frame']))
        motions = evaluate_contexts(model, readouts, c4s, contexts)
        true = np.asarray([r['targets'] for r in rows], np.float32)
        past = np.asarray([r['past_applied'] for r in rows])
        tape = np.asarray([r['applied_tape'] for r in rows])
        rest = np.all(past[:, -10:, :] == 0, axis=(1, 2)) & np.any(tape[:, :, 0] != 0, axis=1)
        turn = ~np.any(tape[:, :, 0] != 0, axis=1) & np.any(tape[:, :, 2] != 0, axis=1)
        other = ~rest & ~turn
        H = {500: 4, 800: 7}

        def group_metrics(mask):
            m = {}
            for name, pred in motions.items():
                m[name] = {f'{ms}ms': dict(xy_rmse_m=rmse(pred[mask, h], true[mask, h], [0, 1]), yaw_rmse_deg=yaw_rmse(pred[mask, h], true[mask, h]))
                           for ms, h in H.items()}
                p, t = pred[mask, 7], true[mask, 7]
                m[name]['800ms'].update(median_translation_ratio=float(np.median(np.linalg.norm(p[:, :2], axis=1)/np.maximum(np.linalg.norm(t[:, :2], axis=1), 1e-9))) if mask.any() else None,
                                        median_excess_translation_m=float(np.median(np.linalg.norm(p[:, :2], axis=1)-np.linalg.norm(t[:, :2], axis=1))) if mask.any() else None)
            return dict(windows=int(mask.sum()), metrics=m)
        groups = dict(rest_start=group_metrics(rest), in_place_turn=group_metrics(turn), other=group_metrics(other))
        # Development transfer population (240 windows).
        windows = json.loads((TRANSFER/'transfer_targets.json').read_text())
        tcache, tcontexts, index = {}, [], {}
        for w in windows:
            key = (w['case'], w['frame'])
            if key not in index:
                directory = TRANSFER/f"case_{w['case']:02d}"
                tcache.setdefault(directory, Recording.training_case(directory))
                index[key] = len(tcontexts)
                tcontexts.append((tcache[directory], w['frame']))
        tmotions = evaluate_contexts(model, readouts, c4s, tcontexts)
        transfer = {}
        for ms in (500, 700):
            sel = [w for w in windows if w['horizon_ms'] == ms]
            t = np.asarray([w['actual'] for w in sel], np.float32)
            transfer[f'{ms}ms'] = {}
            for name in tmotions:
                p = np.asarray([tmotions[name][index[(w['case'], w['frame'])], ms//100-1] for w in sel])
                transfer[f'{ms}ms'][name] = dict(xy_rmse_m=rmse(p, t, [0, 1]), yaw_rmse_deg=yaw_rmse(p, t))
                for maze in (0, 1):
                    for group in ('translation', 'turn'):
                        m = np.asarray([w['maze'] == maze and w['group'] == group for w in sel])
                        transfer[f'{ms}ms'][name][f'maze{maze}_{group}_xy_rmse_mm'] = 1000*rmse(p[m], t[m], [0, 1])
        g = groups

        def at(group, name, ms, key):
            return g[group]['metrics'][name][f'{ms}ms'][key]

        def measures(v1, v2):
            return {
                'A1_rest_translation_ratio_in_0.75_1.25': 0.75 <= at('rest_start', v2, 800, 'median_translation_ratio') <= 1.25,
                'A2_rest_xy_rmse_at_most_half_v1': at('rest_start', v2, 800, 'xy_rmse_m') <= .5*at('rest_start', v1, 800, 'xy_rmse_m'),
                'B1_turn_xy_rmse_at_most_v1': at('in_place_turn', v2, 800, 'xy_rmse_m') <= at('in_place_turn', v1, 800, 'xy_rmse_m'),
                'B2_turn_excess_translation_at_most_10mm': at('in_place_turn', v2, 800, 'median_excess_translation_m') <= .010,
                'B3_turn_yaw_rmse_at_most_1.05_v1': at('in_place_turn', v2, 800, 'yaw_rmse_deg') <= 1.05*at('in_place_turn', v1, 800, 'yaw_rmse_deg'),
                'C1_other_no_loss': all(at('other', v2, ms, k) <= 1.05*at('other', v1, ms, k) for ms in (500, 800) for k in ('xy_rmse_m', 'yaw_rmse_deg')),
                'C2_transfer_no_loss': all(transfer[f'{ms}ms'][v2][k] <= 1.05*transfer[f'{ms}ms'][v1][k] for ms in (500, 700) for k in ('xy_rmse_m', 'yaw_rmse_deg')),
            }
        criteria = measures('C3_v1', 'C3_v2')
        c4_report = dict(gating=False, comparison='C4-v2 against C4-v1 with the §5 measures', measures=measures('C4_v1', 'C4_v2'))
        result = dict(schema='c3v2_offline_acceptance.v1', passed=all(criteria.values()), criteria=criteria, c4_report_only=c4_report, heldout_groups=groups,
            transfer=transfer, transfer_v1_reference_progress_report_mm=dict(maze0_700_translation=46.62, maze1_700_translation=52.15),
            predeclaration_sha256=sha('docs/go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md'),
            readout_v2_sha256=sha(BASE/'c3v2_readout_fit_v1/readout_v2_final.pt'), c4_v2_sha256=sha(BASE/'c4v2_fit_v1/direct_v2_final.pt'),
            evaluator_sha256=sha(__file__), heldout_contexts=len(rows), transfer_windows=len(windows))
        owner.save(OUT/'result.json', result)
        print(json.dumps(dict(passed=result['passed'], criteria=criteria, c4_report_only=c4_report['measures'])), flush=True)


if __name__ == '__main__':
    main()

"""Frozen split-calibration transfer to recorded navigation trajectories.

Applies coefficients fitted on the branch assay, unchanged, to executed-action forecasts
on the 2,404 matched navigation windows. The reference mean is taken over the assay's
candidate bank (forward, left_arc, right_arc) sharing each window's recorded committed
prefix. Nothing is refitted here and no outcome for an unexecuted alternative is used.

This tests forecast transfer on observed trajectories. It is NOT a counterfactual
branch-discrimination test: each window supplies one observed future, so the "branch
deviation" is a deviation from the PREDICTED reference-bank mean, not from an observed
counterfactual target mean.

The original evaluator and the completed experiment's bindings are untouched.
"""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.rgb_body_tensor_interface_development import observation_tensors
from scripts import evaluate_go2_refitted_dynamics_navigation_forecasts_development as source
from scripts import train_go2_anchored_visual_dynamics_development as training

REPO = Path('/home/andrewknowles/Workspace/LeWMQuad-v3')
OUTPUT = (REPO / '.generated/navigation_development_artifacts_v1/'
          'go2_anchored_navigation_transfer_v1_attempt_001')
COEFFICIENTS = OUTPUT / 'frozen_coefficients.json'
PLAN = REPO / 'docs/go2_anchored_visual_navigation_plan_2026-09-17.json'
RESULT = REPO / 'docs/go2_anchored_navigation_transfer_result_2026-09-18.json'
BANK = ('forward', 'left_arc', 'right_arc')
CONDITIONS = ('persistence', 'action_raw', 'no_future_action_raw',
              'action_calibrated', 'no_future_action_calibrated')
# Attribution factorial for the action arm only: which component causes which effect.
# raw = (1,1); common_only scales the shared mean and leaves the deviation at 1;
# branch_only scales the deviation and leaves the mean at 1; calibrated scales both.
DECOMPOSED = CONDITIONS + ('action_common_only', 'action_branch_only')
HORIZONS = 7


def bank_indices():
    """Resolve the bank through the canonical ACTIONS mapping, never by literal index."""
    index = tuple(source.ACTIONS.index(a) for a in BANK)
    assert index == (1, 2, 3), f'bank did not resolve to canonical (1,2,3): {index}'
    return index


def metrics(rows, conditions=CONDITIONS):
    if not rows:
        return dict(windows=0, curves={})
    return dict(windows=len(rows), curves={c: [dict(horizon_ms=100 * (h + 1),
                mse=float(np.mean([r['mse'][c][h] for r in rows]))) for h in range(HORIZONS)]
                for c in conditions})


@torch.inference_mode()
def evaluate(limit=None, check=False, decompose=False):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    plan = json.loads(PLAN.read_text())
    frozen = json.loads(COEFFICIENTS.read_text())
    assert tuple(frozen['reference_bank']) == BANK, 'frozen bank differs from this evaluator'
    alpha = {a: (frozen['coefficients'][a]['alpha_common'],
                 frozen['coefficients'][a]['alpha_branch']) for a in training.ARMS}
    index = bank_indices()
    conditions = DECOMPOSED if decompose else CONDITIONS

    model = training.representation.load()
    predictors = {a: training.load(a) for a in training.ARMS}
    started = time.monotonic()
    all_rows = []
    summaries = []
    checks = {}

    for number, record in enumerate(plan['roots'], 1):
        root = Path(record['root'])
        window_path = root / 'saved_executed_motion_forecast_evaluation_v1.json'
        assert source.digest(window_path) == record['window_sha256'], f'input changed: {root}'
        windows = json.loads(window_path.read_text())['rows']
        plans = {p['frame']: p for p in json.loads((root / 'planning.json').read_text())
                 if 'selection' in p}
        reader = source.NoisyPublicReplay(root / 'native')
        packet = lru_cache(maxsize=16)(reader.policy_packet)
        rows = []
        for ordinal, window in enumerate(windows, 1):
            if limit and ordinal > limit:
                break
            frame = window['frame']
            p = plans[frame]
            correction = p['motion_correction']
            history = source.causal_history_tensors(
                [packet(i) for i in range(frame - 3, frame + 1)], p['measured_ns'])
            inputs = source.delayed_candidate_inputs(
                history, p['committed_prefix'], delay_ticks=3, commit_ticks=4)
            if correction['terminal_translation_pulse']:
                inputs['known_action_blocks'] = torch.as_tensor(
                    source.command_sequences(p['committed_prefix'], pulse=True)[:, :, None],
                    dtype=torch.float32) / torch.tensor([.3, 1., .5])
            executed = source.ACTIONS.index(window['action'])
            in_bank = window['action'] in BANK
            select = list(index) + [executed]

            def take(value, rows_wanted):
                if isinstance(value, dict):
                    return {n: v[rows_wanted] for n, v in value.items()}
                return value[rows_wanted]

            chosen = {k: take(v, select) for k, v in inputs.items()}
            h = chosen['observation_history']
            # Shared history is identical across candidates by construction; encode once.
            past = model.encoder({k: v[:1].flatten(0, 1) for k, v in h.items()}).reshape(1, 4, 32)
            anchor = model.target({'rgb': h['rgb'][:1, -1]})
            past4 = past.expand(len(select), -1, -1)
            anchor4 = anchor.expand(len(select), -1)
            args = (past4, anchor4, chosen['known_action_blocks'], chosen['known_action_valid'])

            anchor_np = anchor[0].numpy().astype(np.float64)
            predictions = {}
            for arm, predictor in predictors.items():
                out = predictor(*args)[:, :HORIZONS].numpy().astype(np.float64)
                innovation = out - anchor_np
                reference = innovation[:3].mean(0)
                executed_innovation = innovation[3]
                common, branch = alpha[arm]
                predictions[f'{arm}_raw'] = anchor_np + executed_innovation
                predictions[f'{arm}_calibrated'] = (
                    anchor_np + common * reference + branch * (executed_innovation - reference))
                if decompose and arm == 'action':
                    predictions['action_common_only'] = (
                        anchor_np + common * reference + 1.0 * (executed_innovation - reference))
                    predictions['action_branch_only'] = (
                        anchor_np + 1.0 * reference + branch * (executed_innovation - reference))
                if check:
                    identity = anchor_np + 1.0 * reference + 1.0 * (executed_innovation - reference)
                    checks.setdefault('identity_max_abs', []).append(
                        float(np.abs(identity - predictions[f'{arm}_raw']).max()))
                    zero = anchor_np + 0.0 * reference + 0.0 * (executed_innovation - reference)
                    checks.setdefault('persistence_max_abs', []).append(
                        float(np.abs(zero - np.broadcast_to(anchor_np, zero.shape)).max()))
                    if arm == 'no_future_action':
                        checks.setdefault('no_action_candidate_spread', []).append(
                            float(np.abs(innovation[:3] - innovation[:1]).max()))
            predictions['persistence'] = np.broadcast_to(anchor_np, (HORIZONS, 32)).copy()

            if check:
                rgb = chosen['observation_history']['rgb']
                checks.setdefault('history_identical', []).append(
                    bool(torch.equal(rgb, rgb[:1].expand_as(rgb))))
                blocks = chosen['known_action_blocks']
                prefix_len = len(p['committed_prefix'])
                checks.setdefault('prefix_identical', []).append(
                    bool(torch.equal(blocks[:, :prefix_len], blocks[:1, :prefix_len].expand(
                        blocks.shape[0], -1, *blocks.shape[2:]))))

            future = []
            for offset in range(1, HORIZONS + 1):
                value = packet(frame + offset)
                assert value['image']['measured_ns'] == p['measured_ns'] + offset * 100_000_000
                future.append(observation_tensors(value)['rgb'])
            target = model.target({'rgb': torch.stack(future)}).numpy().astype(np.float64)
            errors = {c: np.square(predictions[c] - target).mean(-1).tolist() for c in conditions}
            rows.append(dict(run=number, frame=frame, action=window['action'],
                             group=window['group'], in_bank=in_bank, mse=errors))
            if ordinal % 400 == 0:
                print('TRANSFER', number, ordinal, flush=True)
        summary = dict(run=number, root=str(root), total=metrics(rows, conditions),
                       in_bank=metrics([r for r in rows if r['in_bank']], conditions),
                       out_of_bank=metrics([r for r in rows if not r['in_bank']], conditions),
                       by_action={a: metrics([r for r in rows if r['action'] == a], conditions)
                                  for a in source.ACTIONS})
        if not check and not decompose:
            training.probe.save(OUTPUT / f'run_{number:02d}.json', dict(summary=summary, rows=rows))
        summaries.append(summary)
        all_rows.extend(rows)
        print('TRANSFER_RUN_COMPLETE', number, len(rows), flush=True)

    result = dict(status='complete', runs=summaries, total=metrics(all_rows, conditions),
                  in_bank=metrics([r for r in all_rows if r['in_bank']], conditions),
                  out_of_bank=metrics([r for r in all_rows if not r['in_bank']], conditions),
                  by_action={a: metrics([r for r in all_rows if r['action'] == a], conditions)
                             for a in source.ACTIONS},
                  conditions=list(conditions), decomposition=decompose,
                  coefficients=frozen['coefficients'], reference_bank=BANK,
                  bank_indices=index, windows=len(all_rows),
                  coefficients_frozen_not_refitted=True,
                  fitted_horizons='pooled 100-800 ms on branch assay',
                  evaluated_horizons='100-700 ms on navigation windows',
                  population_and_horizon_transfer_not_identical_horizon_replication=True,
                  counterfactual_branch_discrimination_not_tested=True,
                  branch_deviation_is_from_predicted_reference_mean=True,
                  overlapping_windows_not_independent=True,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan_sha256=source.digest(PLAN),
                  wall_s=time.monotonic() - started)
    if check:
        result['checks'] = {k: (all(v) if isinstance(v[0], bool) else max(v))
                            for k, v in checks.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--decompose', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    result = evaluate(limit=args.limit, check=args.check, decompose=args.decompose)
    active = result.get('conditions', list(CONDITIONS))
    if args.check:
        print(json.dumps(result['checks'], indent=1))
        print(json.dumps({c: result['total']['curves'][c][-1] for c in active}, indent=1))
        return
    if args.decompose:
        target = OUTPUT / 'component_decomposition'
        target.mkdir(parents=True, exist_ok=True)
        training.probe.save(target / 'result.json', result)
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        training.probe.save(OUTPUT / 'result.json', result)
        training.probe.save(RESULT, result)
    print(json.dumps({c: result['total']['curves'][c][-1] for c in active}, indent=1))


if __name__ == '__main__':
    main()

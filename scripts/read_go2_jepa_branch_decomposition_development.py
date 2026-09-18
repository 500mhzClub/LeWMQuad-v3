"""Post-hoc decomposition; every simple target baseline uses training rows only."""
from collections import defaultdict
import json
import numpy as np
from scripts.probe_go2_jepa_latent_branch_science_development import (
    OUTPUT, ARMS, selected_rows, save)


def main():
    rows = selected_rows()
    train = [i for i,r in enumerate(rows) if r['data_role'] == 'train']
    transfer = [i for i,r in enumerate(rows) if r['data_role'] == 'geometry_transfer']
    with np.load(OUTPUT/'causal_predictions.npz', allow_pickle=False) as archive:
        predictions = {arm:archive[arm+'_predictions'].astype(float) for arm in ARMS}
    with np.load(OUTPUT/'encoded_targets.npz', allow_pickle=False) as archive:
        targets = {arm:archive[arm+'_targets'].astype(float) for arm in ARMS}
    results = {}
    for arm in ARMS:
        p, t = predictions[arm], targets[arm]
        mean = np.broadcast_to(t[train].mean(0), t[transfer].shape)
        prefix = np.stack([t[[j for j in train if rows[j]['prefix_action'] == rows[i]['prefix_action']]].mean(0)
            for i in transfer])
        groups = defaultdict(list)
        for i in transfer:
            groups[rows[i]['cluster'], rows[i]['prefix_action']].append(i)
        centered_prediction, centered_target = [], []
        for indices in groups.values():
            centered_prediction.extend(p[indices]-p[indices].mean(0))
            centered_target.extend(t[indices]-t[indices].mean(0))
        cp, ct = np.asarray(centered_prediction), np.asarray(centered_target)
        curves = []
        for h in range(8):
            mse = lambda a,b: float(np.mean((a-b)**2))
            zero = float(np.mean(ct[:, h]**2))
            error = mse(cp[:, h], ct[:, h])
            curves.append(dict(horizon_ms=(h+1)*100,
                model_mse=mse(p[transfer, h], t[transfer, h]),
                training_horizon_mean_mse=mse(mean[:, h], t[transfer, h]),
                training_prefix_action_mean_mse=mse(prefix[:, h], t[transfer, h]),
                centered_action_effect_prediction_mse=error,
                action_independent_centered_prediction_mse=zero,
                centered_action_error_ratio=None if zero == 0 else error/zero))
        results[arm] = curves
    result = dict(schema='jepa_branch_posthoc_decomposition.v1', results=results,
        exploratory_after_primary_results=True, simple_baselines_use_training_targets_only=True,
        prefix_baseline_does_not_use_future_action_or_transfer_targets=True,
        centered_action_effect_is_diagnostic_not_deployable_prediction=True,
        weights_or_original_results_changed=False)
    save(OUTPUT/'posthoc_decomposition.json', result)
    print(json.dumps({arm:curves[-1] for arm,curves in results.items()}, indent=2))


if __name__ == '__main__':
    main()

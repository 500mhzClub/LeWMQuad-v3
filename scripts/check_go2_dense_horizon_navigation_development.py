"""Compare navigation-boundary inference with completed retained-branch results."""
import json
from pathlib import Path
import numpy as np
import torch

from lewm.dense_horizon_navigation_development import load_dense_navigation_model
from lewm.route_rgb_dataset_development import load_route_observation
from scripts import train_go2_horizon_dense_predictor_development as fit

RESULT = Path('docs/go2_dense_horizon_navigation_check_2026-09-18.json')


def main():
    assert not RESULT.exists()
    torch.set_num_threads(4)
    reference = fit.parent.reference
    rows = reference.selected_rows()
    first = next(r for r in rows if r['data_role']=='train')
    indices = [i for i, r in enumerate(rows) if (r['data_role'], r['cluster'], r['prefix_action'])
        == (first['data_role'], first['cluster'], first['prefix_action'])]
    assert len(indices)==3
    indices *= 2
    directory, _, _ = reference.inputs(first)
    packets = [load_route_observation(directory, i) for i in (3, 8, 13)]
    commands = torch.tensor([rows[i]['known_commands'][:8] for i in indices], dtype=torch.float32)
    expected = json.loads(Path('docs/go2_horizon_dense_predictor_evaluation_2026-09-18.json').read_text())
    records = []
    for arm in fit.ARMS:
        model = load_dense_navigation_model(arm)
        model.set_native_context(packets, observed_ns=packets[-1]['image']['measured_ns'])
        out = model(observation_history={}, known_action_blocks=commands[:, :, None]/torch.tensor([.3, 1., .5]),
            known_action_valid=torch.ones(6, 8, 1, dtype=torch.bool))
        motion = np.asarray(model.receipts[-1]['motion_xy_yaw'])
        target = np.asarray(expected['motion_predictions'][arm])[indices]
        np.testing.assert_allclose(motion, target, rtol=0, atol=2e-5)
        assert torch.equal(out['rollout_outcomes'][:, :3], out['rollout_outcomes'][:1, :3].expand(6, -1, -1))
        if arm=='no_future_action':
            assert torch.equal(out['rollout_outcomes'], out['rollout_outcomes'][:1].expand(6, -1, -1))
        assert not out['contact_prediction_available'] and (out['rollout_outcomes'][:, :, 4]==-1000).all()
        assert model.pending_context is None
        records.append(dict(arm=arm, max_absolute_motion_error=float(np.abs(motion-target).max()),
            identical_prefix_forecasts_exact=True, receipt=model.receipts[-1]))
        del model
        torch.cuda.empty_cache()
    fit.save(RESULT, dict(status='PASS', records=records, training_only=True, new_navigation=False,
        source_sha256={p:fit.digest(p) for p in (__file__, 'lewm/dense_horizon_navigation_development.py')}))
    print('PASS', [(r['arm'], r['max_absolute_motion_error']) for r in records], flush=True)


if __name__=='__main__':
    main()

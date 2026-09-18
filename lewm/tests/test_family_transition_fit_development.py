"""Family-plan identity, durable-step failure and complete raw-score accounting."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lewm.family_transition_fit_development import verified_plan, train, score
from lewm.geometry_progress_family_causal_windows_development import remaining_candidate
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.tests.test_geometry_progress_family_learning_view_development import windows
from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from scripts.run_go2_family_transition_fits_v1 import benchmark_decision


def inputs(view, ids):
    blocks, valid = zip(*(remaining_candidate(view.windows[i]['action'], view.windows[i]['offset_ticks']) for i in ids), strict=True)
    n = len(ids)
    return dict(observation_history={k: torch.zeros((n, *s)) for k, s in
        dict(rgb=(4, 3, 96, 128), body=(4, 20, 63), control=(4, 15, 7)).items()},
        known_action_blocks=torch.stack(blocks), known_action_valid=torch.stack(valid))


def test_original_family_plans_must_match_before_information_removal():
    view = FamilyWindowView(windows()); ids = view.indices('train')[:6]; batch = inputs(view, ids)
    verified_plan(view, ids, batch)
    batch['known_action_blocks'][0, 0, 0, 0] += .01
    with pytest.raises(ValueError, match='exact family'): verified_plan(view, ids, batch)


def test_accounting_failure_retains_actual_update_and_forbids_resume():
    torch.set_num_threads(1)
    view = FamilyWindowView(windows()); schedule = view.schedule(updates=1200, batch_size=6, seed=2026091001)
    def batch(ids):
        data = inputs(view, ids); active, offsets = verified_plan(view, ids, data); n = len(ids)
        targets = dict(motion=torch.zeros(n, 8, 3), contact=torch.zeros(n, 8), motion_valid=active.clone(),
            contact_valid=active.clone(), future_valid=active.clone(), target_offsets_ns=offsets,
            future_observations={k: torch.zeros(n, 8, *v.shape[2:]) for k, v in data['observation_history'].items()})
        return dict(inputs=data, targets=targets)
    trainer = CumulativePulseTrainer('jepa', seed=17, latent_dim=8)
    def failed_receipt(row):
        assert row['update'] == 1 and row['sample_indices'] == schedule['batches'][0]
        raise ValueError('receipt unavailable')
    stream = SimpleNamespace(view=view, training_batch=batch)
    with pytest.raises(ValueError, match='receipt unavailable'):
        train(trainer, stream, schedule, input_variant='full', on_update=failed_receipt, benchmark=True)
    assert trainer.failed and trainer.updates == 1
    with pytest.raises(ValueError, match='fresh'):
        train(trainer, stream, schedule, input_variant='full', on_update=lambda r: None, benchmark=True)


def test_raw_scoring_retains_censoring_and_rejects_wrong_role_population():
    view = FamilyWindowView(windows()); ids = view.indices('geometry_transfer')
    masks, clocks = [], []
    p = np.zeros((len(ids), 8, 5), np.float32); p[:, :, 3] = 1.
    for i in ids:
        active = np.arange(1, 9)*5 <= 40-view.windows[i]['offset_ticks']
        offsets = np.where(active, np.arange(1, 9)*500_000_000, 0)
        view.windows[i]['targets'] = [dict(in_plan=bool(a), offset_ns=int(t),
            motion_valid=bool(a and j == 0), motion=[0., 0., 0.] if a and j == 0 else None,
            contact_valid=bool(a), contact=float(j > 0) if a else None) for j, (a, t) in enumerate(zip(active, offsets))]
        masks.append(active); clocks.append(offsets)
    data = dict(indices=np.asarray(ids), prediction_valid=np.asarray(masks), target_offsets_ns=np.asarray(clocks), rollout_outcomes=p)
    result = score(view, data, role='geometry_transfer', head='rollout_outcomes')
    all_rows = [r for r in result['clusters'] if r['scope']=='all']
    assert sum(r['motion_targets'] for r in all_rows) == len(ids)
    assert all(r['position_error_m'] == r['yaw_error_rad'] == 0. and r['contact_brier'] == .25 for r in all_rows)
    assert sum(r['contact_positives'] for r in all_rows) > 0
    bad = deepcopy(data); bad['indices'] = bad['indices'][::-1]
    with pytest.raises(ValueError, match='complete ordered'): score(view, bad, role='geometry_transfer', head='rollout_outcomes')
    p[:, :, 2:4] = 0.
    assert all(r['yaw_error_rad'] is None and r['undefined_yaw'] > 0 for r in score(view, data, role='geometry_transfer', head='rollout_outcomes')['clusters'])


def phases():
    def rows(prefix):
        return [dict(name=f'{prefix}_{i}', status='FAMILY_TRANSITION_WORKER_COMPLETE', actual_updates=20,
            fit=dict(seed=2026091010+i, model_sha256=str(i)), ledger_sha256=str(i), peak_rss_bytes=2*1024**3) for i in range(4)]
    return dict(wall_s=40., records=rows('serial')), dict(wall_s=15., records=rows('parallel'))


def test_benchmark_requires_exact_updates_and_uses_frozen_speed_and_memory_rule():
    a, b = phases(); assert benchmark_decision(a, b)['selected_workers'] == 4
    b['wall_s'] = 35.; assert benchmark_decision(a, b)['selected_workers'] == 1
    b['wall_s'] = 15.; b['records'][0]['peak_rss_bytes'] = 9*1024**3
    assert benchmark_decision(a, b)['selected_workers'] == 1
    b['records'][0]['ledger_sha256'] = 'changed'
    with pytest.raises(ValueError, match='match exactly'): benchmark_decision(a, b)

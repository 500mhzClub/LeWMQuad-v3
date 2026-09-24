"""Fixed visual-goal cost diagnostic beyond the predictor's 500-ms endpoint.

Retrospective development only. Forecasts use causal context and planned actions;
goal RGB specifies the task. Future motion is used only to score decisions.
"""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from scripts import evaluate_go2_frozen_vjepa_native_branches_development as base
from scripts import train_go2_frozen_vjepa_native_adaptation_development as fit
from lewm.physical_execution_development import rotation_xyzw

OUTPUT = fit.OUTPUT / 'distant_visual_goals_attempt_002'
RESULT = Path('docs/go2_dense_distant_visual_goals_result_2026-09-17.json')
GOAL_FRAMES = (18, 21, 25)


def goal_poses(directory, row):
    cameras = json.loads((directory/'camera_audit.json').read_text())
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
        pose = archive['base_pose_world']
        contact = archive['physics_contact']
        times = archive['timestamp_s']
    start = cameras[13]['physical_sample_index']
    rotation = rotation_xyzw(pose[start, 3:])
    result = {}
    for frame in GOAL_FRAMES:
        at = cameras[frame]['physical_sample_index']
        assert abs(times[at]-times[start]-(frame-13)*.1) < 1e-9
        assert not contact[start:at+1].any()
        delta = rotation.T @ (pose[at, :3]-pose[start, :3])
        relative = rotation.T @ rotation_xyzw(pose[at, 3:])
        result[frame] = np.array([delta[0], delta[1], np.arctan2(relative[1, 0], relative[0, 0])])
        if frame <= 21:
            label = row['targets'][frame-14]
            assert label['motion_valid'] and label['future_observation_index'] == frame
            np.testing.assert_allclose(result[frame], label['motion'], rtol=0, atol=1e-12)
    return result


@torch.inference_mode()
def run():
    terminal = json.loads((fit.OUTPUT/'result.json').read_text())
    assert terminal['status'] == 'COMPLETE'
    assert not RESULT.exists()
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    rows = base.selected_rows()
    prepared = [base.inputs(r) for r in rows]
    plan = dict(source_sha256=base.digest(__file__), predictor_fit_sha256=base.digest(fit.OUTPUT/'result.json'),
        windows_sha256=base.digest(base.data.PULSE/'windows.json'),
        goal_frames=GOAL_FRAMES, forecast_horizon_ms=500, departure_frame=13,
        goal_horizons_ms=[(f-13)*100 for f in GOAL_FRAMES], primary_goal_horizon_ms=800,
        selection='minimum normalized dense feature MSE; uniform expectation over exact ties',
        checkpoint_epoch=23, no_fitting=True, no_navigation=True,
        arms=['action', 'no_future_action', 'persistence', 'observed_future_oracle'],
        physical_labels_used_for_selection=False,
        concurrency='one GPU process, four CPU threads on CPUs 8-11; reuse encoder per image',
        resources_before_launch=dict(available_ram_gib=72, output_free_gib=4.0, root_free_gib=1.2,
            competing_kfd_processes=0, vram_total_gib=31.86, vram_used_gib=1.72),
        parallelism='small shared-encoder task; extra processes would duplicate encoder and context work',
        precision='float32 encoder and predictor; batch-one image encoding, three action candidates',
        purpose='separate distant visual-goal cost failure from forecast error before online integration')
    base.save(OUTPUT/'plan.json', plan)
    models = {}
    for arm in fit.ARMS:
        path = fit.OUTPUT/f'{arm}_latest.pt'
        assert base.digest(path) == terminal['checkpoint_sha256'][arm]
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['epoch'] == 23
        model = base.ProprioActionPredictor(use_proprio=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        models[arm] = model.cuda().eval().requires_grad_(False)
    del state
    encoder = base.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'), torch.float32)
    cache = {}

    def encode(path):
        key = base.digest(path)
        if key not in cache:
            pixels = encoder.preprocess(str(path))[None].cuda()
            value = F.layer_norm(encoder.tokens(pixels).float(), (1024,))[0].cpu()
            assert value.shape == (768, 1024) and torch.isfinite(value).all()
            cache[key] = value
        return cache[key]

    stats = json.loads((base.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean, std = (np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std'))
    context = torch.stack([torch.stack([encode(d/f'rgb_{f:04d}.png') for f in (3, 8, 13)])
                           for d, _, _ in prepared])
    control = torch.from_numpy(np.stack([(c-mean)/std for _, c, _ in prepared]))
    actions = torch.from_numpy(np.stack([a for _, _, a in prepared]))
    predictions = {a: [] for a in fit.ARMS}
    for offset in range(0, len(rows), 3):
        x, a, c = (v[offset:offset+3].cuda() for v in (context, actions, control))
        assert torch.equal(x, x[:1].expand_as(x)) and torch.equal(c, c[:1].expand_as(c))
        mask = torch.ones(3, 768, dtype=torch.bool, device='cuda')
        for arm, model in models.items():
            if arm == 'no_future_action':
                p = F.layer_norm(model(x[:1], torch.zeros_like(a[:1]), mask[:1], control=c[:1]).float(), (1024,))
                p = p.expand(3, -1, -1)
            else:
                p = F.layer_norm(model(x, a, mask, control=c).float(), (1024,))
            assert torch.isfinite(p).all()
            predictions[arm].append(p.cpu())
    predictions = {a: torch.cat(v) for a, v in predictions.items()}
    predictions['persistence'] = context[:, -1]
    base.save(OUTPUT/'causal_forecasts_complete.json', dict(contexts=36, future_rgb_loaded=False))
    print('CAUSAL_FORECASTS_COMPLETE', flush=True)
    goals = {}
    for f in GOAL_FRAMES:
        goals[f] = torch.stack([encode(d/f'rgb_{f:04d}.png') for d, _, _ in prepared])
        print('GOAL_IMAGES_ENCODED', f, 'unique_images', len(cache), flush=True)
    predictions['observed_future_oracle'] = goals[18]
    for directory, _, expected in prepared:
        with np.load(directory/'policy_histories.npz', allow_pickle=False) as archive:
            np.testing.assert_allclose(archive['applied_command_values'][18][-5:, [0, 2]].reshape(10),
                                       expected, rtol=0, atol=1e-6)
    measured = [goal_poses(d, r) for r, (d, _, _) in zip(rows, prepared, strict=True)]
    poses = {f: np.stack([m[f] for m in measured]) for f in GOAL_FRAMES}
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row['data_role'], row['cluster'], row['prefix_action']), []).append(i)
    details = []
    for key, indices in groups.items():
        assert len(indices) == 3
        for f in GOAL_FRAMES:
            target = goals[f][indices]
            actual = goals[18][indices]
            oracle_costs = (actual[:, None]-target[None]).square().mean((-1, -2)).numpy()
            xy_costs = np.linalg.norm(poses[18][indices, None, :2]-poses[f][None, indices, :2], axis=-1)*1000
            yaw_delta = poses[18][indices, None, 2]-poses[f][None, indices, 2]
            yaw_costs = np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))*180/np.pi
            record = dict(role=key[0], cluster=key[1], prefix_action=key[2], goal_horizon_ms=(f-13)*100,
                trials=[rows[i]['trial'] for i in indices], oracle_visual_costs=oracle_costs.tolist(),
                actual_xy_costs_mm=xy_costs.tolist(), actual_yaw_costs_deg=yaw_costs.tolist(), models={})
            for arm, values in predictions.items():
                costs = (values[indices, None]-target[None]).square().mean((-1, -2)).numpy()
                outcomes = []
                for j in range(3):
                    chosen = np.flatnonzero(costs[:, j] == costs[:, j].min())
                    oracle = np.flatnonzero(oracle_costs[:, j] == oracle_costs[:, j].min())
                    outcomes.append(dict(goal_trial=rows[indices[j]]['trial'], chosen_indices=chosen.tolist(),
                        source_branch_probability=float(j in chosen)/len(chosen),
                        oracle_choice_probability=float(np.isin(chosen, oracle).mean()),
                        visual_regret=float(oracle_costs[chosen, j].mean()-oracle_costs[:, j].min()),
                        xy_regret_mm=float(xy_costs[chosen, j].mean()-xy_costs[:, j].min()),
                        yaw_regret_deg=float(yaw_costs[chosen, j].mean()-yaw_costs[:, j].min())))
                record['models'][arm] = dict(costs=costs.tolist(), goals=outcomes)
            details.append(record)
    summaries = {}
    metrics = ('source_branch_probability', 'oracle_choice_probability', 'visual_regret', 'xy_regret_mm', 'yaw_regret_deg')
    for role in ('train', 'geometry_transfer'):
        summaries[role] = {}
        for f in GOAL_FRAMES:
            chosen_groups = [g for g in details if g['role'] == role and g['goal_horizon_ms'] == (f-13)*100]
            summaries[role][str((f-13)*100)] = {a: {k: float(np.mean([v[k] for g in chosen_groups
                for v in g['models'][a]['goals']])) for k in metrics} for a in predictions}
    # The 500-ms task is an exact control reproducing the existing evaluator.
    previous = json.loads(Path('docs/go2_frozen_vjepa_native_adaptation_branch_result_2026-09-17.json').read_text())
    for g in (g for g in details if g['goal_horizon_ms'] == 500):
        original = next(p for p in previous['groups'] if p['trials'] == g['trials'])
        for a in fit.ARMS:
            np.testing.assert_allclose(g['models'][a]['costs'], original['models'][a]['mse_matrix'], rtol=0, atol=1e-6)
    result = dict(status='COMPLETE', plan=plan, summaries=summaries, groups=details,
        prior_scoring_failure='distant_visual_goals_attempt_001/failure.json',
        scoring_correction='derive goal motion directly from recorded full 3D body transforms; no planar composition',
        unique_images=len(cache), wall_s=time.monotonic()-started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), no_dense_cache_retained=True,
        navigation_tested=False, observed_future_oracle_not_deployable=True,
        limitations=['two exposed geometries per role; dependent goals within six history groups',
            'single-tick pulse with subsequent hold, not sustained-command navigation',
            'source-branch identity need not be optimal for a later goal',
            'separate XY/yaw regret minima need not correspond to the same action',
            'goal image is an externally supplied task, not an autonomously obtained navigation subgoal'])
    base.save(OUTPUT/'result.json', result)
    base.save(RESULT, result)
    print('DISTANT_VISUAL_GOALS_COMPLETE', json.dumps(summaries), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            base.save(OUTPUT/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc()))
        raise

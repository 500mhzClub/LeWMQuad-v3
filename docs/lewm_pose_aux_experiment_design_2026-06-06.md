# Pose-Aux Geometry Experiment Design (2026-06-06)

## Decision

Run Path 2C as a controlled **representation-track ablation** from the 8k-run e3
checkpoint. Do not launch the previously drafted `pose_aux_lambda=1.0` fine-tune
unchanged.

Fixed source checkpoint:

`models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt`

The evidence supports changing the representation objective, but not assuming that
an arbitrary pose-loss scale, command-integrated labels, or encoded-pair decode
quality will transfer to deployed MPC.

This experiment is separate from the registered H-JEPA/frozen-LeWM track. It must
be reported as a deliberate supervised-auxiliary departure, not silently folded
into that track.

## Claim Boundary

The target claim is:

> Physical pose labels are used during training to shape a locally metric latent;
> deployment-time planning uses only camera observations, candidate commands, the
> learned predictor, and the learned relative-pose head.

The permitted claim is **no privileged geometry at deployment**. This is not a
self-supervised result and it is not global localization. The learned geometry is
local, within the training-window motion distribution, and must be tested for
transfer to candidate ranking and closed-loop navigation.

Command-integrated labels remain an explicit ablation. They are not the primary
label because commanded motion and realized quadruped motion diverge through slip,
collisions, controller tracking error, and reset boundaries.

## Evidence Lock

The current conclusion is not based on one failed planner:

- Frozen `z_proj` pairwise-distance head: Pearson `+0.054`, Spearman `+0.083`,
  with approximately zero within-band resolution.
- Frozen `z_raw`: weak coarse signal only, Pearson/Spearman approximately `+0.177`,
  with approximately zero fine-resolution correlation and visible overfit.
- First-action ranking and hard candidate selection are approximately chance.
- Metric-cost navigation is at or below random across tested horizons, including
  horizon 1 where execution-horizon mismatch is absent.
- The forward-model result remains strong. The 8k-run e3 checkpoint gives
  `zero-free@h10=+0.201`, `free/persistence@h10=0.395`, and MPC win-rate versus
  zero at h5 `=0.663`.

Therefore:

1. More planner tuning cannot recover geometry absent from the representation.
2. Forward prediction must be protected as a no-regression capability.
3. Encoded-pair pose decoding alone is insufficient evidence; the deployed
   predictor-endpoint to encoded-goal contract must improve.

## Implemented Contract

The pose-aux path now has the following experiment controls:

- Primary labels: aligned physical replay poses from `frames.jsonl`.
- Ablation labels: active-block command integration.
- Pair sampling: all non-identical bidirectional frame pairs.
- Weighting: equal total weight per absolute frame gap, preventing short-gap pair
  count from dominating.
- Yaw loss: wrapped angular residual.
- Encoded contract:
  `RelPoseHead(z_proj[a], z_proj[b]) -> body-relative pose(a,b)`.
- Deployed contract:
  `RelPoseHead(plan_rollout(z_raw[0], cmd[:h])[-1], z_proj[goal]) ->
  body-relative pose(h,goal)`.
- Checkpoint integrity: the pose head, objective weights, label source, rollout
  configuration, and optimizer state are included in resumable checkpoints.
- Evaluation integrity: the per-checkpoint watcher requires valid prior-metric,
  geometry, and navigation JSONs before marking an epoch complete.

The geometry probe reports both contracts, plus fixed distance-band correlations.
This prevents a global correlation driven by coarse near/far separation from being
mistaken for the fine metric resolution needed to rank one-step actions.

## Experiment Ladder

Use a fixed, deterministic, stratified proxy before any full-data continuation.
Start with 300-1000 environment sessions and one epoch per cell. Use the same e3
initial weights, holdout split, batch order, and evaluation caches.

| cell | encoder update | labels | encoded loss | predicted-to-goal loss | purpose |
|---|---|---|---:|---:|---|
| F0 | frozen | actual | head only | head only | frozen-head ceiling; confirms the old representation cannot support the contract |
| C0 | yes | none | 0 | 0 | continuation control; measures drift from e3 without geometry |
| C1 | yes | actual | tuned | 0 | tests whether encoded metric structure can be injected |
| C2 | yes | actual | tuned | tuned | tests the exact deployed planning contract |
| A1 | yes | command | matched to winner | matched to winner | label-source ablation, only after an actual-pose cell is viable |

Do not assume `lambda=1.0`. On a short pilot, log the unweighted base, encoded-pose,
and predicted-pose losses and their encoder gradient norms. Select weights that make
each auxiliary encoder-gradient contribution material but not dominant. A practical
initial screen is approximately `0.03`, `0.1`, and `0.3`, narrowed by gradient ratio
before the one-epoch proxy.

The bounded calibration command is:

```bash
PY=.generated/venvs/genesis_render_vulkan/bin/python
SRC=models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt
$PY scripts/probe_pose_aux_gradient_scale.py \
  --checkpoint "$SRC" --data-root .generated/datagen_full \
  --render-root .generated/datagen_full/render_textured_v03 \
  --seq-len 11 --stride 5 --max-sessions 8 --max-batches 4 \
  --pose-label-source actual --device cpu \
  --output models/pose_aux_proxy_20260606/gradient_scale_actual.json
```

Use the reported `0.1x` and `0.3x` gradient-ratio weights to choose a small
screen, then keep the selected weights fixed across C1, C2, and A1. F0 is
directly runnable with `scripts/train_lewm.py --freeze-model`.

For every proxy cell, preserve the source checkpoint's settled objective:
`seq_len=11`, `stride=5`, `sigreg_lambda=0.09`, `rollout_lambda=0.25`,
`rollout_horizon=10`, and `rollout_gamma=0.9`. Set
`rollout_warmup_epochs=0`; the source is already past warmup, so restarting a
warmup would be an uncontrolled reduction in rollout pressure. Use
`--max-sessions 300` first, then repeat the winning comparison at 1000 sessions.
F0 adds `--freeze-model`; C0 sets both pose lambdas to zero; C1 sets only
`--pose-aux-lambda`; C2 sets both pose lambdas. Every cell gets a separate
output directory and the same shuffle/eval seeds. `--init-from` loads weights
only and intentionally does not inherit these flags; the trainer warns whenever
an init-from run changes source objective/config fields.

A one-batch, two-window CPU smoke on one frontier session suggested
approximately `0.087` encoded and `0.072` predicted for a 10%-of-base encoder
gradient contribution (`0.260` and `0.216` for 30%). Each unweighted auxiliary
gradient exceeded the base encoder-gradient norm. This validates the probe and
rejects `lambda=1.0` as an unmeasured default; the sample is too small and
unrepresentative to select the registered cell weights.

Promote only C1/C2 cells that beat C0 on geometry while preserving forward-model
gates. C2 is the preferred winner because it supervises the contract used by MPC.

## Promotion Gates

All gates use the same deterministic held-out windows/scenes as the baseline.

### Forward-model no-regression

Relative to 8k e3:

- `zero-free@h10 >= +0.18`
- `free/persistence@h10 <= 0.50`
- MPC recorded-action win-rate versus zero at h5 `>= 0.62`

A geometry gain that breaks these gates is not a successful world-model result.

### Geometry

Require improvement over both the frozen-head ceiling and C0:

- encoded-pair physical-pose XY error decreases and distance correlation rises;
- predicted-to-goal physical-pose metrics improve with the same trend;
- correlations improve inside the `0-0.2 m`, `0.2-0.5 m`, and `0.5-1.0 m` bands,
  not only globally;
- bearing and wrapped-yaw errors do not hide a distance-only shortcut.

The predictor-to-goal metrics are the decisive offline gate. If encoded-pair
geometry improves but predictor-to-goal geometry does not, stop and fix the
projection/predictor contract before running navigation.

### Planning and navigation

Before expensive physics:

1. Re-run candidate-ranking/first-action regret. The pose cost must beat random and
   reduce first-action regret; a good pair decoder with chance action ranking fails.
   For promotion, require positive mean first-action Spearman and pose first-action
   regret no more than half the random-pick regret
   (`mean_random_first_dist_m - mean_oracle_first_dist_m`).
2. Run kinematic visible-beacon navigation on at least 32 scenes and 3 seeds, with
   start-yaw jitter. Compare `lewm_pose`, bearing, hold, and random.
3. Keep goal views single-view for the primary test. Multi-view goal minimization
   changes the goal-set semantics while success is still measured against one target
   pose, so it is a separate ablation.
4. Run physics only after the kinematic result clears random with a stable margin.

## Decision Tree

- **C1 and C2 fail geometry:** stop Path 2C at this model/data scale. Report that
  local physical-pose supervision did not produce a usable metric substrate; pursue
  recognition/topological planning or a stronger spatial-token substrate.
- **C1 improves encoded geometry, C2/predictor contract fails:** the encoder can
  retain geometry, but rollout projection destroys it. Redesign or jointly align
  predictor/projector outputs before navigation.
- **C2 improves predictor contract but candidate ranking fails:** inspect candidate
  distribution coverage and train on harder action-conditioned endpoint pairs.
- **Candidate ranking passes but kinematic navigation fails:** diagnose observation
  aliasing/history and closed-loop goal-image semantics.
- **Kinematic passes:** validate physics and scale the winning cell, preserving the
  same forward-model and geometry gates.

## Launch Status

No GPU fine-tune is approved or launched by this design update. The earlier
`pose_aux_lambda=1.0`, command-label-first launch is superseded. First run the
small controlled cells and select loss weights from measured gradient scale.

The current scripts are ready for the frozen and joint-training cells and their
evaluation:

- `scripts/train_lewm.py`
- `scripts/cache_lewm_latents.py`
- `scripts/probe_pose_geometry.py`
- `scripts/probe_pose_aux_gradient_scale.py`
- `scripts/diagnose_nav_cost.py`
- `scripts/watch_finetune_evals.sh`
- `scripts/benchmark_lewm_closed_loop_mpc.py`

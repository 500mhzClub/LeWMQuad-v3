# Go2 DINO true-successor goal-cost V1 result

Date: 2026-08-04

Status: **terminal development result — preregistered gate failed; stop the
frozen-DINO same-patch-cost route.**

## Executive result

The experiment completed all 24 development scenes, all seven paired arms,
and every provenance and integrity check. Giving the planner the *actual*
one-step successor image made frozen DINOv2 same-patch goal costs highly useful
for closed-loop task progress: true successor achieved 14/24 successes, while
shuffled DINO, DINO persistence, hold, and random each achieved 0/24.

That task effect was causal with respect to candidate/score correspondence.
True-successor DINO exceeded shuffled-score DINO by 0.2957 m mean progress,
with a paired whole-scene bootstrap 95% interval of [0.1014, 0.4726] m.

The route nevertheless failed its fixed training-eligibility gate. The
scene-mean geometric first-action-regret reduction against shuffled scores was
0.01574 m, below the registered 0.020 m minimum, and its paired interval for
`true - shuffled` was [-0.04360, 0.01188] m rather than wholly below zero.
Because a learned predictor cannot improve the planning suitability of its
actual target under the same terminal cost, the dense DINO predictor must not
be trained under this protocol.

This is a useful mechanistic result, not evidence that a learned world model
works. Frozen DINO successor-to-goal similarity carries real navigation signal,
but the exact same-position cosine cost does not reliably preserve the local
physical action ranking required by the registered planner gate.

## Frozen scope

- 24 sorted V4 **development-only** medium-maze scenes; one visible-beacon
  trial per scene.
- Kinematic CPU execution, H1, nine registered actions, 12-block cap, seed 7.
- Frozen `dinov2_vits14` on the R9700, exact repository/checkpoint bindings,
  224x224 input, 256x384 patch grid, per-patch L2 normalization.
- Cost: mean same-position patch cosine distance to one goal image.
- Seven arms in fixed order: geometric oracle, true-successor DINO,
  shuffled-score DINO, DINO persistence, bearing, hold, and random.
- 10,000 paired whole-scene bootstrap resamples, seed 2026080402.

The true-successor arms use privileged simulator counterfactual renders. This
is a terminal-cost ceiling, not a deployable policy, learned prediction test,
held-out result, or safety result.

## Main measurements

| Arm | Mean progress (m) | Mean scene regret (m) | Success |
|---|---:|---:|---:|
| geometric `oracle_mpc` | 0.8151 | 0.0000 | 14/24 |
| `dino_true_successor` | 0.6494 | 0.05956 | 14/24 |
| shuffled DINO | 0.3537 | 0.07530 | 0/24 |
| DINO persistence | 0.0000 | 0.13534 | 0/24 |
| bearing ceiling | 0.9000 | 0.02486 | 24/24 |
| hold | 0.0000 | 0.13534 | 0/24 |
| random | 0.4968 | 0.07346 | 0/24 |

Key paired contrasts for `dino_true_successor`:

| Comparator | Progress difference, mean [95%] (m) | Regret difference, mean [95%] (m) |
|---|---:|---:|
| shuffled DINO | +0.29568 [0.10139, 0.47261] | -0.01574 [-0.04360, 0.01188] |
| DINO persistence | +0.64936 [0.48924, 0.79538] | -0.07578 [-0.09611, -0.05455] |
| geometric oracle | -0.16576 [-0.33370, -0.01758] | +0.05956 [0.04057, 0.07991] |

True-successor mean progress also exceeded random by 0.15253 m and hold by
0.64936 m. The random contrast's interval was not a registered gate; only its
mean was required.

## Gate accounting

Passed:

- complete 24-scene x seven-arm panel, no skips, finite scores, and exact
  source/checkpoint/cost/device provenance;
- zero geometric-oracle decision regret;
- true versus shuffled progress advantage at least 0.10 m and lower 95% bound
  above zero;
- true versus persistence progress interval wholly favorable;
- true versus persistence regret interval wholly favorable;
- true mean progress above random and hold.

Failed:

- shuffled-regret reduction: 0.01574 m versus the required 0.020 m;
- shuffled-regret bootstrap upper bound: +0.01188 m versus the required value
  below zero.

Machine verdict:
`FAIL_STOP_FROZEN_DINO_SAME_PATCH_COST_ROUTE`.

The failure is not just a negligible miss on one arbitrary scalar: both the
minimum effect-size criterion and the uncertainty criterion for the same
physical-ranking contrast failed. The strong success/progress effect should be
retained as evidence when designing a materially different successor, but it
must not be used post hoc to waive this gate.

An independent recomputation reproduced the analysis exactly. Per-scene regret
differences were favorable in 12/24 scenes and unfavorable in 12/24, with a
`true - shuffled` median of +0.00131 m (slightly worse), so the missing
reliability is not an analyzer sign error. One design caveat should be retained:
scene regret is measured on each policy's own states after its trajectory has
diverged (226 true-successor decisions versus 288 shuffled decisions). It is a
policy-level physical-ranking comparison, not a pure same-state rank assay.
That caveat narrows interpretation but does not license overriding the fixed
gate; a successor can preregister a same-state diagnostic separately.

## What is established

1. The planner seam can exploit correctly assigned action scores: shuffled
   reassignment preserved every DINO score but destroyed the success effect.
2. Actual visual change matters: true successors strongly beat a persistence
   grid under the same encoder and goal cost.
3. Frozen dense DINO tokens contain task-relevant image-goal navigation signal
   under privileged one-step rendering.
4. Same-position patch cosine distance is not a sufficiently reliable proxy
   for the registered local geometric action ranking.

## What is not established

- No learned DINO predictor was trained or evaluated.
- No evidence establishes action-conditional generalization, multistep latent
  rollout, persistent memory, collision safety, physical robot performance, or
  held-out performance.
- The 14/24 success count is not evidence that token-prediction accuracy would
  retain the ceiling's decision margins.
- Bearing's 24/24 is a privileged task ceiling, not a fair learned-model
  comparator.

## Consequence

Apply the preregistered stop rule exactly:

- do not tune this cost after observing the result;
- do not retry another seed or enlarge this development panel to rescue the
  same claim;
- do not train the already implemented dense DINO temporal predictor for this
  same-patch terminal-cost route;
- do not describe the privileged true-successor ceiling as a learned world
  model result.

A next attempt must change the mechanism materially and receive a new
preregistration. The already named admissible classes are a nonlinear
task-coupled readout, an embodiment-supervised physical target, or a dense
V-JEPA successor. The benchmark, planner seam, paired interventions, and
development custody can be retained.

## Bound artifacts

- Preregistration:
  `docs/lewm_go2_dino_true_successor_goal_cost_v1_preregistration_2026-08-04.md`,
  SHA-256
  `21a2d994c7606a95accf6b793de8ea1e9c99f019fc99f58983bdcce605c80113`.
- Raw 24-scene output:
  `.generated/dino_true_successor_goal_cost_v1/full_development_24scene_h1_seed7.json`,
  3,333,378 bytes, SHA-256
  `1976c73a37a6f2df5db9958f722ffbee7e7d6aaaaecc6d621cd65b0fc989d5fa`.
- Frozen analysis:
  `.generated/dino_true_successor_goal_cost_v1/full_development_24scene_h1_seed7_analysis.json`,
  18,690 bytes, SHA-256
  `69ebc897ab721a2bd9ad9db19e075e34f6899494bbfe04c384eed04d0703844b`.
- Benchmark source SHA-256:
  `c926d54c81bfdf149c1d79baf41b78c1bd05f206414536827e9ddc2d5603d52f`.
- Analyzer source SHA-256:
  `7fa397cce5d0c76e4d2cf8a6203a34fb528683de1e78d97adc20b587238ccfae`.

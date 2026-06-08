# Frozen-Base Task-Aligned Policy Result

Date: 2026-06-08

## Decision

Close frozen-LeWM candidate-scorer and inference-search variants. Pooled raw
features, coarse spatial features, and four-frame history all failed the
registered minimum validity gate in all three seeds.

Do not collect new rollouts. The next bounded experiment should adapt only the
final LeWM vision-encoder blocks using the direct task-aligned objective.

Artifacts: `.generated/task_aligned_policy_v0/`

## Implemented Experiment

The base LeWM checkpoint remained frozen:

`models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt`

`TaskAlignedCandidateScorer` receives a frozen start descriptor, optional
frozen target descriptor, goal-present bit, and one candidate 15-D action
block. It predicts four outcomes separately:

- collision probability;
- target progress;
- heading error;
- final clearance.

All nine registered velocity primitives are scored. Selection combines the
four predicted outcomes, but promotion requires separate regret, collision,
and progress gates so a turn-in-place policy cannot pass by gaming one scalar
cost.

The bounded train and validation sets each contain 16,384 decisions from 32
scene-disjoint, family-balanced scenes. Privileged geometry is used only to
produce training/evaluation targets.

## Frozen Substrates

- `raw`: pooled raw LeWM encoder vector from the current and target images.
- `spatial2`: CLS plus a 2x2 pooled patch-token grid.
- `history4`: current raw vector, four-frame mean vector, and temporal delta.

Each substrate used the same multi-task head, target definitions, three seeds,
and held-out gates.

## Results

Three-seed means:

| frozen substrate | mean regret | collision rate | target progress | minimum gate passes |
|---|---:|---:|---:|---:|
| pooled raw | 0.141 | 15.76% | +0.0030 m | 0 / 3 |
| 2x2 spatial | **0.139** | 15.76% | **+0.0035 m** | 0 / 3 |
| four-frame history | 0.143 | **15.72%** | +0.0015 m | 0 / 3 |
| action-only `yaw_left` prior | **0.116** | **14.04%** | 0.0000 m | control |
| random action | 0.432 | 29.49% | +0.0172 m | control |

Every learned substrate substantially beats random regret, but every substrate
is worse than the trivial action-only prior on regret and collision. None
beats random target progress. No seed passed the minimum gate or promotion
gate.

Training loss continued to improve after validation action selection stopped
improving. This indicates that the heads can fit outcome targets, but the
frozen descriptors do not support a robust held-out action-selection rule
under the current task contract.

## Selection-Rule Checks

Two extra controls were run after the frozen-head failures:

- add a listwise action-ranking auxiliary to the spatial head;
- sweep deployed selection rules over trained raw, spatial, and history heads.

The ranking auxiliary did not help. The best spatial ranking checkpoint still
lost to the `yaw_left` prior and selected negative target progress:

| variant | regret | collision | progress | gate |
|---|---:|---:|---:|---|
| spatial + ranking auxiliary | 0.128 | 15.01% | -0.0059 m | fail |

The selection-grid sweeps also found zero passing settings:

| descriptor | best low-score regret | collision | progress | best progress | progress-mode collision |
|---|---:|---:|---:|---:|---:|
| raw | 0.133 | 15.26% | +0.0003 m | +0.0302 m | 24.10% |
| spatial2 | 0.133 | 15.31% | +0.0019 m | +0.0336 m | 26.54% |
| history4 | 0.130 | 14.89% | -0.0011 m | +0.0285 m | 23.01% |

So this is not just a bad fixed inference cost. The heads can be forced to
choose moving actions, but only by accepting collision and regret far above
the registered gates.

## Interpretation

The failure is not evidence that more inference search is needed. Search can
only rank information exposed by its input representation. The same result
persists after adding coarse spatial layout and short temporal context.

It also does not justify a full from-scratch base-model retrain yet. The next
controlled escalation is a small task-aligned adaptation:

1. initialize from the same LeWM checkpoint;
2. freeze the predictor, projectors, and early vision encoder;
3. unfreeze only the final two vision-transformer blocks and final norm;
4. train them jointly with the multi-task candidate scorer;
5. use pixels, optional target pixels, and candidate actions as deployed
   inputs; privileged labels remain targets only;
6. evaluate on the identical 32-scene validation set and three seeds.

## Registered Gates For Encoder Adaptation

Minimum validity, all three seeds:

- regret below the action-only prior: `< 0.116`;
- collision below the action-only prior: `< 14.04%`;
- positive target progress.

Promotion before closed-loop testing:

- regret ratio versus random `<= 0.5`;
- selected collision rate `<= 5%`;
- target progress above random (`> +0.0172 m`);
- no privileged fields consumed at inference.

If final-block adaptation fails, stop small-head/adapter tuning and review the
task target, goal-conditioning contract, and base training objective before
any large retrain.

## First Adapter Screen

The first bounded adapter run unfroze only the final two ViT blocks and final
norm, for one seed over eight full train/eval epochs:

Artifact: `.generated/task_aligned_policy_v0/adapter_last2_seed20260608.json`

| model | regret | collision | progress | gate |
|---|---:|---:|---:|---|
| best frozen spatial | 0.134 | 15.50% | +0.0035 m | fail |
| final-two-block adapter | 0.134 | 15.25% | +0.0019 m | fail |
| action-only `yaw_left` prior | 0.116 | 14.04% | 0.0000 m | control |

The adapter improved over most frozen heads and trended upward through the
final epoch, but it still failed the minimum gate. More adapter training is
not yet a promotion path; the immediate next review should inspect whether the
task target and goal-conditioning contract over-reward low-motion safe turns
before running larger adapter/retrain jobs.

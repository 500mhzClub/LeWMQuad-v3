# Go2 scene-diversity recurrent replication V1 preregistration

Date frozen: 2026-08-04

Status before execution: `PREREGISTERED_NOT_RUN`

Scope: development-only H1 mechanism test. This document grants no sealed,
held-out, navigation, rollout, planner, promotion, deployment, retry, resume,
or overwrite authority.

## Question and prior result

The predecessor task-coupled recurrent direct-dynamics experiment stopped with
visual regret `0.1536644345` and matched no-vision regret `0.1461774554`.
Visual minus no vision was `+0.007486979`, with a paired 95% interval of
`[-0.0029296875,+0.017903646]`. Visual input therefore generalized in the
wrong direction even though its training regression loss was lower.

This experiment asks one narrower question: does doubling independent scene
diversity, at fixed training-state count and fixed optimization, reverse that
visual-specific generalization failure?

It is not a search over models, encoders, losses, seeds, thresholds, pooling,
PCA width, update count, or evaluation roles. More evaluation samples alone
cannot rescue a wrong-direction mechanism, so the intervention changes scene
clustering while preserving the number and marginal composition of states.

## One fixed intervention

The predecessor used, in each role and family, two scenes with all eight fixed
history pairs per scene. V1 uses four scenes with four history pairs per scene:

- eight ordinary development families;
- four train scenes and four evaluation scenes per family;
- 32 scenes and 128 states per role;
- four states per scene;
- all nine matched counterfactual candidate actions per state;
- 64 scenes, 256 states, 2,304 branches, 768 context frames, and 2,304 target
  frames in total.

The eight predecessor history pairs remain unchanged. Within each role and
family, alternating scene slots receive the even or odd history-index subset,
so every history pair occurs exactly twice. Family, history, action, state,
PCA-cell, minibatch, and update counts therefore match the predecessor; only
the number of independent scene clusters doubles and states per cluster halve.

The ordinary scene-corpus universe, the existing frozen non-scientific scene
exclusion set, and all 32 scenes consumed by the predecessor bounded panel are
frozen before selection. No predecessor scene may reappear. From the remaining
ordinary `train` corpus rows, one SHA-256 ranking with seed `20260804` selects
eight scenes per family and a second frozen hash allocates four to train and
four to evaluation. There is no content inspection, hand selection, screening,
refill, or role reassignment after outcomes are known.

This is a fresh development evaluation role. V4 and every sealed, `sealed_*`,
held-out, or production role remain inaccessible and ineligible.

## Frozen model protocol

The model protocol is byte-for-byte inherited in scientific behavior from
`go2_task_coupled_recurrent_dynamics_v1`:

- frozen DINOv2 ViT-S/14 context patches only;
- three context frames, pooled from `16 x 16` to `4 x 4` spatial cells;
- train-only channel PCA `384 -> 16` and train-only standardization;
- exact measured body-frame odometry and executed historical command tapes;
- exact candidate command tape queried after recurrent context formation;
- 3,604 trainable parameters per member;
- visual and exact-zero-vision arms with matched initialization;
- model seeds `2026080411`, `2026080412`, and `2026080413`;
- shared sampler seed `2026080414`;
- 800 updates, batch size eight, AdamW learning rate `3e-4`, weight decay
  `1e-4`, gradient clipping at `1.0`;
- standardized four-output physical residual MSE plus `0.25` strict-pair rank
  softplus loss;
- live task/action-only ridge control fit on the new train role;
- goal available to the scorer and control, not the recurrent model;
- no successor observation, successor feature, semantic label, depth target,
  reward, planner, rollout, encoder update, JEPA target, or EMA route.

The old task/action regret was an identity witness for the old role, not a
portable numerical constant. On the new role it is recomputed live using the
unchanged analytic control. Its method and the task-relative threshold are
frozen; equality to the old role's numeric regret is explicitly not required.

## Access order and integrity

Scene identities, role allocation, histories, plan, sources, DINO binding, and
authority must be frozen before generation. Generation may materialize both
development roles, but the model runner may initially open only the collection
result metadata and train receipt/context closure. It must:

1. open and rehash the 128 train state receipts and 384 train context PNGs;
2. fit all train-only statistics and six learned members;
3. durably write and reopen the exact checkpoint;
4. only then open and rehash the 128 evaluation receipts and 384 evaluation
   context PNGs;
5. open zero train or evaluation successor PNGs;
6. evaluate the durable checkpoint twice and require exact result equality.

Train/evaluation scene IDs and state IDs must be disjoint. The collector and
runner fail closed on count, binding, role, family, history, action, source,
runtime, or access-order mismatch. A failed collection or experiment consumes
the attempt and authorizes no retry or resume.

## Fixed reports and gates

The following predecessor gates are unchanged:

1. integrity, role disjointness, context-only custody, repeatability, and the
   privileged physical oracle must pass;
2. ensemble visual normalized rank regret must be `<= 0.13`;
3. visual minus live task/action regret must be `<= -0.02` and the paired
   family/scene-bootstrap 95% upper bound must be below zero;
4. visual minus matched no-vision regret must be `<= -0.01` and the paired
   family/scene-bootstrap 95% upper bound must be below zero;
5. visual must beat uniform-random expectation.

The same scorer, 10,000-resample paired family/scene bootstrap, seed, and gate
implementation are used. No threshold will be relaxed because the new role is
harder or because a point estimate is close.

In addition to the gate report, the result must report member and ensemble
regrets, oracle-equivalent selection, target progress, standardized physical
MSE, per-family values, task/no-vision comparisons, training traces, collection
cost, receipt counts, context/successor open counts, and source/checkpoint/result
bindings.

## Decision rule

- `PASS_SCENE_DIVERSITY_RECURRENT_REPLICATION_H1` requires every fixed gate.
  It supports only the claim that this frozen recurrent-DINO mechanism shows
  incremental visual H1 value under the more scene-diverse fixed-state recipe.
  It does not establish a world model useful for planning; a separately
  preregistered blind rollout/planner test would still be required.
- `STOP_SCENE_DIVERSITY_RECURRENT_REPLICATION_H1` is terminal for this
  recurrent-DINO recipe on the development route. It authorizes no additional
  scenes, seeds, updates, hyperparameter changes, threshold changes, or encoder
  swaps on the exposed roles. The next decision must be a materially different
  representation/data mechanism or a return to the repository's formal G2-G8
  path.
- Infrastructure or integrity failure yields
  `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`, still with no automatic retry.

Meaningful improvement that remains below a gate is recorded but does not
count as a pass and does not by itself authorize tweaking. In particular, the
predecessor visual-minus-no-vision point estimate must move by at least
`0.017486979` merely to reach the fixed `-0.01` relative threshold, and visual
regret must improve by `0.0236644345` to reach the absolute threshold.

## Resource bounds

The predecessor collection with the same 256 states, 2,304 branches, and 3,072
frames used 30.43 minutes, 117.35 MiB of PNG payload, and 15.19 GB peak selected
device memory at 72 lanes/scene. Four states/scene use 36 lanes; the prospective
estimate is approximately 37 minutes collection and 40 minutes end to end.
The one-shot collection wall ceiling is 7,200 seconds, the stored-RGB ceiling
is 512 MiB, and the exact stored-frame cap is 3,072. These are safety/resource
ceilings, not scientific stopping criteria.


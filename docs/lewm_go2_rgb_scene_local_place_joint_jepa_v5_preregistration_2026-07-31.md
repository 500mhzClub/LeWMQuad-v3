# RGB Scene-Local Place Joint-JEPA V5 Preregistration

Date: 2026-07-31 UTC

## Decision and narrow hypothesis

V4 is terminally closed at update 100 and 3,200 presentations with status
`FAIL_SCIENTIFIC_UPDATE100_CONTINUATION_GATE_TERMINAL`.  Its committed result
is `docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v4_scientific_result_2026-07-31.json`
at commit `8309cee66b15508b70b0d9e32ed3b920529e96ee`.

V4 did not fail because its new place key lacked an initial signal.  At update
zero, its scene-equal retrieval R@5 was `0.222972`, or `2.398687` times exact
chance, across all eight development scenes.  Training improved physical
margins from 39 to 70, strengthened local action discrimination, and more than
doubled the sampled positive-versus-negative place separation.  Nevertheless,
place R@5 fell by 43.35% to `1.358929` times chance, target rank fell from
`2.614277` to `2.147930`, and positive revisit energy became 5.59 times worse.

The source and schedule expose a specific mismatch.  V4 concatenated one row
from each of eight scene families and used all sixteen EMA targets in a
categorical contrastive loss.  For each query, fourteen of fifteen alternatives
were easy cross-scene or cross-family shortcuts, while checkpoint selection
ranks candidates only inside one scene.  V5 tests exactly one obvious repair:
train the same place representation only against scene-local, distinct-place
alternatives and require direct positive alignment to improve.

This is the last isolated place-head correction justified by this result.  It
is not permission for a V6 series.  A failure closes this mechanism and
requires a broader memory-integration mechanism audit.

## Frozen components

V5 preserves from V4:

- the shared V18 RGB/object-space encoder and `64 x 64 x 64` latent;
- the `Conv2d(64,16,1)`, exact GELU, adaptive `4 x 4` pool, and
  `Linear(256,64)` normalized place key;
- the zero-final-layer identity-residual place predictor;
- the complete inherited physical route, immediate local route, losses,
  gradient routing, AdamW optimizer, EMA coefficient and update order;
- model seed `20260732` and inherited projection seed `20260729`;
- the existing train and checkpoint-selection indexes, RGB normalization,
  scene-disjoint roles, runtime, and GPU;
- 16 physical, 8 local, and 8 place presentations per update;
- observations at updates 0, 100, and 400; and
- a hard maximum of 400 updates and 12,800 total presentations.

V5 starts from fresh initialization.  No V4 private restart tensor, failed
checkpoint, optimizer state, RNG state, or output is opened or reused.

## Exact place-schedule change

The frozen 3,200-row place train index is decoded exactly as before.  The
already validated anchor cell and yaw are immediately reduced to an opaque
SHA-256 place identity; raw cell or yaw values never enter a model batch,
objective, metric, or receipt.

A metadata-only deterministic preflight constructs exactly 800 four-row
groups, exactly 100 groups and 400 row presentations per family.  Every group:

- contains one scene only;
- contains four distinct opaque place identities; and
- contains the original registered anchor, positive, and same-yaw negative RGB
  references for each selected row.

The exact grouping, ordering, reuse, omission, and schedule hash are frozen by
the committed preflight receipt before any V5 RGB open or GPU use.  Group reuse
is allowed only where a family has fewer than 100 disjoint supported groups;
it is deterministic and is reported explicitly.  Within each update, the two
consecutive B4 groups remain separate competitor sets even though they share
one B8 autograd graph.

## Exact place objective

For query row `i` in one scene-local B4 group:

- `e_pos(i) = 1 - cosine(prediction_i, paired_positive_i)`;
- its competitors are its own registered EMA negative plus the other three
  paired EMA positives in the same B4 group;
- no target in the other B4 group and no cross-scene target is a competitor;
- `ranking(i) = mean_j relu(0.05 + e_pos(i) - e_neg(i,j))`.

The complete place loss is:

`mean(e_pos) + mean(ranking) + variance + 0.10 * covariance`.

Variance and covariance are the unchanged V4 B8 online-anchor terms: centered
standard deviation floor `0.05`, epsilon `1e-4`, and off-diagonal squared
covariance divided by 64.  The V4 temperature-0.10 categorical cross-entropy
and all cross-scene candidates are removed.  EMA targets remain stop-gradient,
and encoder, factorizer, predictor, physical, and local parameters are updated
together through the same single optimizer step.

## Update-100 continuation gate

V5 continues beyond update 100 only if every condition is true:

- structural, EMA-target, schedule, and place/local access integrity pass;
- place R@5 is at least `2.0` times exact chance;
- place R@5 retains at least 90% of its own update-zero chance multiple;
- at least six of eight checkpoint-selection scenes exceed exact chance;
- target place-key effective rank is at least `2.0` and at least 80% of its
  own update-zero value;
- mean positive revisit energy is strictly lower than at update zero;
- place negative-minus-positive bootstrap lower 95% is positive in aggregate,
  with at least six positive families; and
- at least 60 of 189 inherited physical margins pass.

The positive-energy condition prevents the exact V4 loophole in which the
reported separation widened only because positives and negatives both became
much worse.  Failure stops the attempt at update 100.

## Update-400 memory-entry gate

V5 earns a perception interface for a separately reviewed learned-memory
experiment only if every condition is true:

- all structural, target, schedule, split, and access integrity checks pass;
- scene-equal place R@5 is at least `2.0` times exact chance;
- at least six of eight scenes exceed exact chance;
- target place-key effective rank is at least `2.0` and at least 80% of its
  update-zero value;
- mean positive revisit energy is strictly lower than update zero;
- aggregate place separation has positive bootstrap lower 95% and at least six
  positive families;
- more than 72 of 189 physical margins pass; and
- all twelve inherited physical causal-control checks pass.

The earlier `3x` R@5 and absolute rank-4 requirements are deliberately demoted,
not retroactively waived.  A later causal memory may combine weak and aliased
single-frame cues over time; V5 must prove a useful, non-destructive
within-scene signal, while the memory experiment must separately prove ordered
history value.  Local immediate-predictor persistence, absolute energy gaps,
rough-depth, tail, and prior metrics remain diagnostic at this stage.

## Stop and custody rules

- One fresh V5 attempt only; retry and resume are false.
- Private immutable restart states are written at registered terminal
  observations but do not authorize opening, reuse, or promotion.
- Any source, schedule, accounting, target-gradient, access, split, or
  integrity violation terminalizes the attempt.
- A V5 failure authorizes no isolated place-head successor and no longer run.
- A V5 pass authorizes only preparation of one learned, stateful memory
  integration test with reset/reverse/shuffle causal controls.
- Probability calibration, navigation, G2, production, deployment, and any
  held-out or sealed access remain prohibited.  No eligible sealed role
  currently exists under the governing authority correction.

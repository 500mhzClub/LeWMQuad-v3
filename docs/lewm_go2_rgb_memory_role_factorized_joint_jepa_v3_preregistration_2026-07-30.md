# RGB Memory-Role Factorized Joint-JEPA V3 Preregistration

Date: 2026-07-30 UTC

## Decision and scope

V2 is terminally closed at 400 updates and 12,800 presentations with status
`FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL`; it may not be retried or resumed.
Its committed terminal result is
`docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_scientific_result_2026-07-30.json`.

V3 is one fresh, materially different learned-perception JEPA successor.  It
tests one narrow hypothesis: V2's memory key failed because global mean pooling,
a randomly initialized unconstrained predictor, and one paired negative allowed
the 64-dimensional place key to collapse to effective rank 1.41.  Preserving a
small spatial grid, anchoring the predictor at identity, and training against a
multi-candidate target set with an explicit anti-collapse term should produce a
retrieval key useful to a later learned memory.

This is not recurrent-memory training, controller training, navigation,
probability calibration, G2, held-out, sealed, production, or promotion work.
The V4 30-scene sealed benchmark remains unopened.

## Exact scientific delta from V2

Only the place/memory-key mechanism changes.

1. The shared V18 RGB/object-space latent remains `64 x 64 x 64`.  The place
   factorizer changes from global mean pooling to:

   - `Conv2d(64, 16, kernel_size=1)`;
   - exact GELU;
   - adaptive average pooling to `4 x 4`;
   - flattening to 256 values;
   - `Linear(256, 64)` and L2 normalization.

2. The place predictor remains a learned JEPA predictor, but becomes an
   identity-anchored residual:

   - `Linear(64, 128)`, exact GELU, `Linear(128, 64)`;
   - the final linear weight and bias are initialized to exact zero;
   - output is `normalize(place_key + residual)`.

3. The two registered place microbatches are concatenated into one eight-row
   place graph.  For eight predicted anchor keys, the candidate matrix is the
   eight paired EMA positive keys followed by the eight paired EMA negative
   keys.  Candidate order is exact and no label or identity enters the model.

4. The old single-pair `0.10` hinge is replaced by the following place loss:

   - `alignment = mean(1 - cosine(prediction, paired_positive))`;
   - `contrast = cross_entropy(prediction @ candidates.T / 0.10,
     arange(8))`;
   - for centered online anchor keys, `std = sqrt(var + 1e-4)` and
     `variance = mean(relu(0.05 - std))`;
   - `covariance = sum(off_diagonal(covariance_matrix)^2) / 64`;
   - `place_loss = alignment + contrast + variance + 0.10 * covariance`.

5. The EMA target remains frozen and stop-gradient, and is updated exactly once
   after the one joint optimizer step.  The new place modules use constructor
   seed `20260732`; the inherited V18 state and unchanged local modules retain
   their existing deterministic initialization.

The local role architecture, cyclic-wrong-action loss, data, and metrics remain
byte-for-byte scientific controls and are diagnostic only for this memory-entry
probe.  The already-verified inherited action-conditioned physical JEPA route
continues to supply immediate perception.  V3 does not change the local
mechanism and local failure cannot block a memory-only successor.  Before any
controller or navigation qualification, an action-conditioned predictor must
still robustly beat identity persistence unless the controller directly uses
the verified physical route.

## Frozen controls

V3 preserves V2's shared encoder and spatial trunk, inherited physical model
and losses, local route, EMA coefficient and update order, one AdamW optimizer
and parameter groups, gradient routing, physical/local/place data and exact row
ordering, scene-disjoint development split, batch size four, role schedule,
hardware/runtime bindings, RGB normalization, source geometry, authority
custody, and seed sources other than the new place modules.

Each update remains exactly:

- 16 physical presentations in four microbatches;
- 8 local presentations in two microbatches;
- 8 place presentations in two data microbatches combined into one place-loss
  graph;
- 32 presentations total.

The maximum is 400 updates and 12,800 presentations.  Observations occur only
at updates 0, 100, and 400.  No V2 tensor or failed state is opened or reused.

## Update-100 continuation gate

V3 writes a private, immutable restart state after the update-100 observation.
The state contains model, optimizer, EMA, accounting, CPU/GPU RNG, and exact
update position, with `resume_authorized=false`.  Reading or resuming it later
requires a separate committed authority.  Its existence is not checkpoint
publication or scientific promotion.

The same attempt continues from update 100 to 400 only if every condition is
true:

- structural, EMA-target, and access integrity pass;
- place R@5 is at least `1.50` times exact chance;
- target place-key effective rank is at least `2.0`;
- place negative-minus-positive bootstrap lower 95% bound is positive;
- at least six of eight place families have positive separation;
- at least 60 of 189 inherited physical margins pass.

Otherwise the attempt stops at update 100 as a valid scientific failure.  No
retry or identical extension is authorized.

## Terminal memory-entry gate

At update 400, memory integration is earned only if every condition is true:

- structural, EMA-target, and access integrity pass;
- mean scene-equal place R@5 is at least three times its mean exact chance;
- at least six of eight development scenes exceed their exact R@5 chance;
- target place-key effective rank is at least `4.0`;
- more than 72 of 189 inherited physical margins pass;
- all twelve inherited causal-control checks pass.

The fixed `0.10` place energy gap, fixed `0.05` local wrong-action gap,
absolute R@5 `0.40`, rank-retention ratio, raw persistence ratio, rough-depth,
tail, and prior metrics are diagnostic only.  This simplification does not
retroactively rescue V2, which achieved only 1.368 times chance, five of eight
scenes, and rank 1.406.

A private immutable restart state is also written after update 400 whether the
gate passes or fails.  Only a gate pass may publish a promoted perception
checkpoint.  A later continuation from either restart state requires a new
preregistration and one-shot authority and must not repeat completed updates.

## Stop rules and next branch

- One V3 attempt only; no retry or resume under this preregistration.
- Any integrity, accounting, source, data, schedule, target-gradient, or access
  violation terminalizes the attempt.
- Update-100 gate failure ends V3 without spending the remaining 9,600
  presentations.
- Update-400 gate failure ends V3 with no memory or navigation execution.
- Update-400 gate pass authorizes only preparation of a separately reviewed
  learned-memory integration experiment on scene-disjoint development data.
- Held-out and sealed material remains inaccessible until the repository's
  ordered development gates earn it.

# RGB ego-motion-aligned ray-consistency joint-JEPA V16 preregistration

Date: 2026-07-29

Status: preregistered source implementation and one bounded development
falsification only. No V16 reservation, data access, GPU work, training,
checkpoint, qualification, navigation, G2, held-out, sealed, production, or
promotion access has occurred.

## Governing result and decision

The science-identical V15 integrity replacement is complete and terminal in
commit `7a0dbc1f850bc8917bc45566425116fdef87ef42`. It reached update 1,400 and
passed all 24 inherited semantic checks, but failed its continuation gate with
94 physical margins instead of 99 and rough depth p95 `1.4730507612228394 m`
instead of strictly below `1.304 m`. The exact V14/V15 mechanism must not be
retried, resumed, or extended.

V16 tests one materially different perception hypothesis: measured motion
between the already paired RGB frames can impose metric agreement on the
learned unified ray-survival field. This directly targets the depth plateau
without another encoder-width, tokenization, decoder, or schedule revision.

## Sole scientific change

The V14 model architecture, online encoder, unified ray-survival evidence
head, semantic state, action-conditioned predictor, EMA target, parameter
partition, and inference API remain unchanged. V16 adds exactly one
training-only loss to the Camera gradient route.

For each current/next pair, independently produce the existing differentiable
`UNKNOWN/FREE/OCCUPIED` metric raster from each frame's unified ray hazards,
within-bin offsets, ground survival, and measured per-frame camera
calibration. The raster shape is `(B,3,64,64)` and channel order is exactly
`UNKNOWN`, `FREE`, `OCCUPIED`.

Let the stored realized motion be
`delta=(dx_forward,dy_left,dyaw)`, the next base pose expressed in the current
base frame. Warp the current raster into the next base frame with the existing
SE(2) convention:

```text
current_forward = dx + cos(dyaw) * next_forward - sin(dyaw) * next_left
current_left    = dy + sin(dyaw) * next_forward + cos(dyaw) * next_left
```

Use bilinear `grid_sample`, zero padding, and `align_corners=True` over exact
cell-center ranges:

```text
forward centers: [-0.95, 5.35] m
left centers:    [-3.15, 3.15] m
```

The shared-view mask is the conjunction of:

- the existing warp-overlap mask;
- all five current-frame ground supports being in-frustum for all four
  `2 x 2` source cells contributing to the warped `64 x 64` cell; and
- the equivalent unwarped next-frame validity.

Warped validity must be strictly greater than `0.999`; partially valid
bilinear footprints are excluded. Renormalize the warped class probabilities
over their three channels. For each valid cell define:

```text
known = FREE + OCCUPIED
q = OCCUPIED / clamp_min(known, float32_epsilon)
weight = stop_gradient(min(known_warped_current, known_next))
```

Clamp each `q` to `[float32_epsilon, 1-float32_epsilon]`. The sole new scalar
is the equally bidirectional stop-gradient Bernoulli KL in the aligned
next-frame lattice:

```text
M = 0.5 * weighted_mean(
      KL_Bernoulli(stop_gradient(q_warped_current) || q_next)
    + KL_Bernoulli(stop_gradient(q_next) || q_warped_current))
```

The weighted mean denominator is the detached weight sum. A batch with zero
weight returns an exact differentiable zero, never NaN. Both directions are
retained so gradients reach both frame encodings. V16 records the loss,
shared valid-cell count, positive-weight cell count, and weight sum.

The Camera objective changes from `C` to exactly:

```text
C_v16 = C + 0.1 * M
```

No other loss or coefficient changes. The existing joint navigation/JEPA
objective `N`, predictor forward, target stop-gradient, EMA update, route-wise
gradient accumulation, clipping, and optimizer step remain active in the
same update. V16 is therefore still joint JEPA training, not a separately
trained encoder or predictor.

Realized SE(2) is training-only geometric supervision. It is not passed into
the encoder, evidence head, semantic head, predictor, EMA target, evaluator,
or inference runtime. Inference remains RGB-only.

## Frozen inputs, initialization, and schedule

Reuse exactly the V14 first-1,000-update scientific identity:

- Raw V13 train role: 4,262 pairs from 72 development scenes;
- checkpoint-selection role: 495 pairs from 8 disjoint development scenes;
- each pair's existing `relative_se2_current_frame` field, validated as one
  finite float32 vector of shape `(3,)` and used only by `M`;
- the same RGB, labels, camera arrays, action fields, role boundaries, and
  source bindings;
- fresh N320 initialization, with no V14 or V15 checkpoint/state opened;
- constructor, schedule, execution, projection, and bootstrap seeds;
- the identical first 16,000-presentation schedule;
- float32 AdamW, three parameter groups, learning rates, betas, epsilon,
  weight decay, and route-wise clipping;
- four microbatches of four pairs per update;
- exactly 1,000 maximum optimizer/EMA updates and 16,000 presentations; and
- observations at updates `0`, `100`, `400`, and `1,000`.

There is one seed and one scientific attempt. There is no loss-weight search,
second topology, second seed, automatic retry, or automatic schedule
extension.

## Gates and stopping rule

Update 100 is informational. At update 400, continue only if structural
integrity and all twelve existing causal controls pass and all of the
following are true relative to the exact V14/V15 update-400 values:

- passed physical margins are at least `72`;
- total physical shortfall is strictly below `68.96964862816927`;
- rough depth p95 is strictly below `1.8582415819168085 m`.

Failure is terminal for this mechanism and publishes no continuation-eligible
checkpoint.

At update 1,000, the unchanged V14 final qualification conjunction remains
authoritative: all 24 inherited checks, at least one complete physical scope,
at least 112 margins, shortfall strictly below `33.05143763708337`, rough
depth p95 strictly below `0.9777327477931971 m`, rough ground balanced
accuracy strictly above `0.647134926562893`, rough pixel balanced accuracy
strictly above `0.8198594673963917`, and complete structural integrity.

If the full gate fails, a science-identical continuation may be proposed from
the update-1,000 recovery checkpoint only when all of these fixed
extension-eligibility conditions pass:

- structural integrity and all twelve causal controls pass;
- at least 23 of 24 inherited checks pass;
- at least 89 physical margins pass;
- total shortfall is strictly below `41.41604892978589`;
- rough depth p95 is strictly below `1.45 m`;
- rough ground balanced accuracy is strictly above `0.647134926562893`; and
- rough pixel balanced accuracy is strictly above `0.8198594673963917`.

Eligibility grants no automatic training or checkpoint access. Any extension
requires a separate exact authority, the identical mechanism and optimizer,
and a preregistered continuation schedule. If these conditions fail, V16 and
its checkpoint are terminal and inaccessible for further training.

## Full-state recovery checkpoints

After a passing update-400 gate, and after an eligible or fully qualifying
update-1,000 observation, publish one immutable full-state checkpoint and one
content-bound JSON binding. Each checkpoint contains:

- complete online and EMA-target model state and model-state manifest;
- optimizer parameter groups, moments, and exact step counters;
- joint-training accounting and next presentation cursor;
- Torch CPU and all CUDA RNG states;
- full schedule identity and consumed-prefix identity;
- source, preregistration, authority, configuration, and attempt identities;
- completed observation/trace bindings needed for exact continuation; and
- the consumed-input ledger and current access receipt needed for terminal
  custody reconciliation.

Publication is write-once and atomic: payload first, then its binding. A
payload without its exact binding is not resumable. Recovery starts at update
401 or 1,001 and may never replay the checkpointed update. Recovery is allowed
only after an infrastructure interruption and only under a separate exact
recovery authority. A failed scientific gate can never be resumed.

The checkpoint is development/recovery state, not a promotable model. It
cannot itself authorize calibration, G2, navigation, held-out access,
production, or deployment.

## Required source-only tests before execution

- zero motion plus identical evidence gives approximately zero `M`;
- a translated synthetic occupied feature follows the exact SE(2) convention;
- an aligned metric-depth change scores lower than an unaligned change;
- perturbing either frame gives finite nonzero gradients to that frame's
  hazards and within-bin offsets;
- excluded or fully masked cells contribute exact differentiable zero;
- malformed/nonfinite SE(2), wrong batch shape, or altered batch membership
  fails closed;
- checkpoint serialization is nonmutating and a CPU save/restore trajectory
  exactly matches uninterrupted next-step model, EMA, optimizer, accounting,
  and RNG state;
- changed schedule/source/cursor/accounting/EMA/optimizer identity, an orphan
  payload, or overwrite attempt is rejected; and
- source closure and filename-only custody checks show zero held-out, sealed,
  calibration, navigation, G2, or rejected-runtime dependency.

## Authority boundary

Implementation, focused tests, recursive source closure, independent source
review, narrow clean-export certification, and one-shot execution authority
must be frozen before reservation or scientific input access. The historical
V4 30-scene benchmark and every sealed or held-out role remain unopened.
Success at this probe qualifies only the next ordered development step; it
does not itself authorize navigation or held-out evaluation.

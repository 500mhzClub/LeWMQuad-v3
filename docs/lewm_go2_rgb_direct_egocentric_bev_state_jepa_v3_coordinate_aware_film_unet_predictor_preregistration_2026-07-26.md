# Direct BEV V3 coordinate-aware FiLM U-Net predictor preregistration

Date: 2026-07-26
Status: preregistered for implementation, independent source review, and
CPU-only synthetic tests; execution is not authorized.

## Frozen V2 evidence

V2 is permanently closed by
`docs/lewm_go2_rgb_direct_egocentric_bev_state_jepa_v2_integrity_terminal_audit_2026-07-26.json`:

- commit: `625a79dbfc85fbf32d0925b4668574828d433ca9`
- raw SHA-256: `93132058a0f94f652864e73e00cfb050c35f901e73d06277e13e3897825ef5a0`
- content SHA-256: `4ed33359075c9cda7ddac16854ccbfd902e0dfe900a38aa83cd94d5fb74f1340`
- byte count: `11802`
- status:
  `PASS_VALID_SCIENTIFIC_UPDATE_100_DIRECTIONAL_GATE_FAILURE_CLOSES_V2_NO_RETRY`
- classification:
  `VALID_SCIENTIFIC_GATE_FAILURE_STRONG_EARLY_DIRECT_BEV_PERCEPTION_LEARNING_ACTION_TRANSITION_REMAINS_AT_CHANCE_V2_PERMANENTLY_CLOSED`
- accounting: 100 updates, 1,600 presentations, 400 objective and backward
  calls, 100 optimizer and EMA updates, and two observations; no retry or
  qualification.

At update 100, V2 had learned useful direct RGB-to-BEV perception: G fell from
0.7755 to 0.2416, J from 0.6531 to 0.5396, raster balanced accuracy rose from
0.3141 to 0.7366, occupied recall from 0.1319 to 0.8788, raster NLL fell from
1.1749 to 0.4749, and correct-RGB scene wins rose from 6 to 8. The transition
remained action-blind: action macro balanced accuracy was 0.10243, action NLL
was 2.1972551, C was 0.998810, and no scene had a positive hardest-wrong
margin.

## One hypothesis

The V2 predictor's three local translation-equivariant convolutions receive
only a spatially constant action condition. Persistence can therefore reduce
J while the predictor lacks absolute position and sufficient spatial context
to learn coordinate-dependent egocentric transitions. V3 tests only whether a
small coordinate-aware, multiscale, action-modulated predictor makes the
already-useful state action-predictive.

## Frozen science

V3 preserves V2 exactly except for the predictor and its all-actions call
path. This includes the RGB perception encoder, global BEV decoder, three-logit
state head, N320 initialization, data and mappings, perception-weight seed and
shared-perception fresh initialization bytes, G + J + C objective, optimizer,
schedule, EMA,
update-zero gate, update-400 and update-1000 gates, and all accounting and
custody rules. Shared perception modules must be freshly constructed with the
same seed and draw order; no V2 checkpoint, tensor, trace, or runtime output
may be opened or reused.

"Optimizer preserved" means exact AdamW class, betas, epsilon, weight decay,
learning rates, grouping semantics, clipping, and float32 precision. The new
predictor necessarily replaces the old predictor parameter inventory; its
exact inventory is frozen below. Encoder parameters remain at `1e-4`, all
decoder/state/predictor parameters at `3e-4`, encoder+decoder/state joint
clipping at `1.0`, and predictor separate clipping at `1.0`.

The single V3 experiment ID is
`go2_rgb_direct_egocentric_bev_state_jepa_v3_coordinate_aware_film_unet_predictor`.
Its output root must be new, versioned, absent before reservation, and distinct
from every V1 and V2 output root.

## Sole model change

Replace only the V2 predictor with this exact residual FiLM U-Net. Every
convolution has kernel `3`, padding `1`, and bias; every GroupNorm has four
groups and affine parameters. `Block(I,O)` is exactly
`Conv(I,O) -> GN4 -> GELU -> Conv(O,O) -> GN4 -> GELU`.

| order | module | exact operation |
|---:|---|---|
| 1 | action embedding | `Embedding(9,64)` |
| 2 | enc64 | `Block(5,16)` |
| 3 | down32 | stride-2 `Conv(16,32) -> GN4 -> GELU` |
| 4 | enc32 | `Block(32,32)` |
| 5 | down16 | stride-2 `Conv(32,48) -> GN4 -> GELU` |
| 6 | enc16 | `Block(48,48)` |
| 7 | down8 | stride-2 `Conv(48,64) -> GN4 -> GELU` |
| 8 | bottleneck | `Block(64,64)` then FiLM-64 |
| 9 | dec16 | nearest 2x upsample, concat enc16, `Block(112,48)`, FiLM-48 |
| 10 | dec32 | nearest 2x upsample, concat enc32, `Block(80,32)`, FiLM-32 |
| 11 | dec64 | nearest 2x upsample, concat enc64, `Block(48,16)`, FiLM-16 |
| 12 | residual head | `Conv(16,3)`, weight and bias exact zero |

The four FiLM projections are constructed in stage order `64/48/32/16` and
are exact `Linear(64,2*C)` modules. For `gamma,beta = split(linear(action))`,
FiLM is exactly `x * (1 + gamma[:,:,None,None]) +
beta[:,:,None,None]`. Embedding, convolution, and linear modules use their
PyTorch default initialization under the frozen seed; GroupNorm uses exact
unit scale and zero bias; only the residual head is overwritten to zero.

The action indices are bound in order to
`(arc_left, arc_right, backward, forward_fast, forward_medium, forward_slow,
hold, yaw_left, yaw_right)`. No permutation is permitted. For state shape
`(B,3,H,W)`, the five encoder channels are exactly `(state logits, row,
column)`: `row = linspace(-1,1,H)` expanded along columns and `column =
linspace(-1,1,W)` expanded along rows, using the state's dtype and device.
Coordinates are constructed per call and are not a persistent buffer.

The predictor inventory is exactly 317,107 parameters in 79 tensors, making
the complete model 6,552,249 parameters in 277 tensors. Construction order is
the table order, with each FiLM projection immediately after its associated
block. The all-actions path encodes state+coordinates once, expands the shared
features in the frozen action order, and runs the shared decoder nine ways; it
must not encode the state nine times. Each candidate is
`current_state_logits + residual`, preserving exact update-zero persistence.

Fresh construction seeds only `torch.random.default_generator` with
`20260712`, restores the caller CPU RNG afterward, and makes zero CUDA, HIP,
MPS, XPU, or other accelerator RNG calls. Shared perception modules retain
V2's exact CPU draw order before the new predictor is constructed.

No pose, odometry, depth, flow target, metric motion, geometry label,
hand-coded primitive transform, analytical warp, `grid_sample`, attention,
auxiliary loss, or bypass is permitted.

## One capped falsification

There is one fresh attempt, with no scientific retry, resume, repair,
replacement, alternate seed,
width change, depth change, conditioning variant, or predictor variant. The
hard caps remain 1,000 updates, 16,000 presentations, and 60 GPU-active
minutes. Observation and failure remain stop-at-first-failure.

Evidence-only retry inside inherited terminal file sealing is retained and
may not execute science, reopen tensors, alter receipts, or create an attempt.

The exact V2 update-100 base conjuncts remain: all prior gates passed, all
registered values finite, state nonconstant, G strictly below update zero, J
strictly below update zero, action NLL strictly below `ln(9)`, action macro
balanced accuracy strictly above `1/9`, and correct-RGB wins in at least six
scenes. V3 additionally requires all five stronger conjuncts:

- action macro balanced accuracy `>= 0.13`;
- action NLL `<= 2.187`;
- hardest-wrong positive-scene count `>= 2`;
- aggregate raster balanced accuracy `>= 0.65`; and
- J `<= 0.60`.

Failure of any conjunct closes V3 at update 100. Passing permits only the same
attempt to continue to the unchanged update-400 and update-1000 gates and
caps. Passing V3 does not authorize navigation, G2, checkpoint qualification,
held-out or sealed access, production, promotion, or deployment.

## Present authority

Only implementation, independent source review, closure checks, and CPU-only
synthetic tests are authorized. Synthetic tests may instantiate Torch models
and synthetic tensors but may not open generated inputs, datasets, RGB,
labels, checkpoints, traces, runtime outputs, GPU state, held-out material, or
sealed material. Training, GPU use, reservation, execution, retry, and every
downstream use require a separately frozen source closure, independent review,
and explicit one-shot authorization.

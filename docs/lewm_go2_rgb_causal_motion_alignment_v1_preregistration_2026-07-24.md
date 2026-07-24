# RGB causal motion alignment V1 preregistration

Date: 2026-07-24

## Decision

Test exactly one materially different learned perception mechanism:
a small, motion-conditioned dense warp aligns the previous frame's
`16 x 16 x 192` visual tokens to the current token grid before the retained
temporal residual and multiresolution evidence decoder.

This is not another encoder architecture revision. The shared visual encoder
topology remains unchanged, although its weights remain trainable under the
existing supervised perception optimizer. This is also not JEPA training.

The predecessor pure-visual temporal V1 mechanism is terminal and must not be
retried, resumed, extended, or used for initialization. Its observed update
1,000 result was:

- `0/9` complete physical scopes;
- `111/189` passed margins;
- total shortfall `33.13261634065992`;
- rough pixel balanced accuracy `0.7403405148373643`;
- rough ground balanced accuracy `0.6217081280253147`; and
- rough depth p95 `1.0263007879257195 m`.

It was effectively indistinguishable from multiresolution V3. Its frozen
access-ledger parser also omitted the authorized development-render RGB root,
so its strict terminal receipt is contract-invalid. The sealed attempt and
frozen source remain untouched. Neither issue authorizes a rerun.

## Hypothesis

Same-grid subtraction assumes that visual tokens at row and column `(i, j)`
refer to the same scene content in both frames. Translation, yaw, roll, pitch,
and parallax violate that assumption. A learned, content-aware warp supplied
with causal nominal command motion and observed attitude change may align
useful visual evidence before fusion, particularly in rough motion.

The falsifiable claim is not that more temporal capacity helps. It is that
learned correspondence conditioned on deployment-available motion context
closes the fixed physical gate within the same 16,000-presentation budget.

## Sole scientific change

The only scientific change relative to temporal V1 is the learned alignment
of previous tokens before the unchanged temporal-difference residual.

For a warm pair:

1. run the shared encoder separately on previous and current RGB;
2. reshape both patch-token tensors to `B x 192 x 16 x 16`;
3. form the exact five-value causal condition:

   ```text
   [nominal_forward_m,
    nominal_left_m,
    nominal_yaw_rad,
    relative_roll_rad,
    relative_pitch_rad]
   ```

4. broadcast that condition over the token grid;
5. concatenate previous tokens, current tokens, and the condition;
6. predict a bounded two-dimensional offset at every token;
7. bilinearly sample the previous token grid at those offsets;
8. pass `current - aligned_previous` through the retained temporal residual;
9. add the residual to current tokens; and
10. feed the fused tokens to the unchanged multiresolution evidence head.

The exact alignment block is:

```text
concat channels: 192 previous + 192 current + 5 condition = 389
Conv2d(389, 32, kernel_size=1, bias=True)
GELU(approximate="none")
Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False)
GELU(approximate="none")
Conv2d(32, 2, kernel_size=1, bias=False), exact-zero initialization
offset_tokens = 2.0 * tanh(raw_offset)
grid_sample(mode="bilinear", padding_mode="border", align_corners=True)
```

Offsets are expressed in token units and converted exactly to the normalized
`align_corners=True` grid by multiplying each axis by `2 / 15`. Channel zero
is source-grid `x`/column displacement and channel one is source-grid `y`/row
displacement. Construct the identity grid with
`x = linspace(-1, 1, 16)` increasing left-to-right over columns and
`y = linspace(-1, 1, 16)` increasing top-to-bottom over rows, so
`identity_grid[row, column] = [x[column], y[row]]`. The sampling equation is:

```text
sample_grid = identity_grid + offset_tokens.permute(0, 2, 3, 1) * (2 / 15)
aligned_previous = grid_sample(previous_tokens, sample_grid, ...)
```

Thus a positive channel-zero offset samples farther right in the previous
token map, and a positive channel-one offset samples farther down. No flow
inversion, subtraction, axis swap, or post-warp mask is permitted.

The retained temporal residual is unchanged:

```text
delta = current_tokens - aligned_previous_tokens
Conv2d(192, 8, kernel_size=1, bias=False)
GroupNorm(4, 8, eps=1e-5, affine=True)
GELU(approximate="none")
Conv2d(8, 8, kernel_size=3, padding=1, groups=8, bias=False)
GELU(approximate="none")
Conv2d(8, 192, kernel_size=1, bias=False), exact-zero initialization
fused = current_tokens + history_valid * residual
```

The alignment block has exactly `12,832` parameters in four tensors. Together
with the retained `3,160`-parameter temporal residual, the changed
post-encoder mechanism has `15,992` parameters. The evidence head has
`368,681` trainable parameters in 35 tensors. The unchanged encoder has
`2,747,520` trainable parameters in 78 tensors. Total trainable capacity is
`3,116,201` parameters in 113 tensors. The exact `368,681` count is this
version's evidence-head ceiling; the smaller predecessor ceiling is
superseded only for this named alignment mechanism.

Alignment-local initialization uses a private CPU generator with seed
`20260726`; the caller CPU RNG is restored exactly. In construction order,
initialize the `389 -> 32` weight with
`xavier_uniform_(gain=1.0, generator=alignment_generator)`, set its bias to
exact zero, initialize the depthwise `3 x 3` weight with the same Xavier
operation and generator, and set the `32 -> 2` offset-projection weight to
exact zero. No framework-default random initialization may remain. The offset
projection and temporal output projection are exact zero at update zero.
Cold history and update-zero migration therefore reproduce the
multiresolution predecessor exactly.

## Causal input contract

The model may consume only:

- previous and current normalized egocentric RGB;
- previous and current `camera_basis_body_fru` already materialized in Raw
  V13 from the deployment-style attitude contract;
- the resulting nominal forward, left, and yaw delta;
- `history_valid`; and
- the existing target-frame camera calibration used by the evidence head.

The nominal table is computed once from train-role pairs only, exactly as the
existing Shared-V5 training runner does: sort the nine primitive names and
take the float32 component-wise `torch.quantile(..., 0.5, dim=0)` of train-role
`relative_se2_current_frame` for each primitive. A checkpoint-selection row
may select from that frozen table but may never contribute to it. Only the
runner sees a primitive string for this lookup; the model receives neither
the primitive ID nor the row's realized SE(2).

For a training row, the lookup key is that row's primitive: it is the issued
command on the exact frozen five-tick, `0.5 s` transition from the row's
current endpoint to its next endpoint. For a warm checkpoint-selection target,
the key is exactly
`predecessor_by_target[current_endpoint]["primitive"]`, where the unique
incoming pair's next endpoint is the evaluated current endpoint. An outgoing
pair for which the evaluated endpoint is the current side must never supply
the condition. Multiple predecessors, stream/scene/episode/env/reset
crossing, or any transition outside the frozen five-tick pair contract raises
before inference.

Roll and pitch for each frame are derived only from its yaw-aligned
`camera_basis_body_fru`. Reconstruct the rotation columns as
`[forward, -right, up]`, then compute:

```text
pitch = atan2(-forward_z, hypot(forward_x, forward_y))
roll = atan2(-right_z, up_z)
relative_angle = atan2(sin(current - previous), cos(current - previous))
```

The wrapped previous-to-current Euler roll and pitch coordinate differences
occupy condition elements four and five; they are not claimed to be an exact
SO(3) relative rotation. Validate each floating `3 x 3` basis in float64:
every element must be finite;
`basis @ basis.T` must equal identity with `rtol=0` and `atol=5e-5`;
`cross(right, forward)` must equal `up` with the same tolerances; and the
reconstructed `[forward, -right, up]` rotation must have determinant within
`5e-5` of `+1`. Derived angles must be finite, with wrapped differences in
`[-pi, pi]`. A malformed basis or invalid warm edge raises rather than being
silently downgraded.

A target with no incoming predecessor is exactly cold: previous RGB equals
current RGB, its condition is the exact float32 zero five-vector, and
`history_valid=False`. The alignment may be computed on those finite values,
but the final temporal residual is bypassed exactly. Missing history is the
only cold fallback; reset crossing, irregular timing, ambiguity, or malformed
data fails closed.

This is issued-command nominal motion, not measured executed displacement.
Deployment must later supply the same semantics from prior issued-command
history and IMU-derived camera bases. Promotion requires a reviewed runtime
hook that passes those histories before the post-encoder alignment block.

Forbidden model inputs include:

- the sample's realized `relative_se2_current_frame`;
- exact simulator pose, position, velocity, or world transform;
- requested target or evaluator feedback;
- scene geometry, labels, depth, or ground truth;
- calibration-role, G2, navigation, held-out, or prior-run outputs; and
- the failed temporal or multiresolution checkpoints.

No Raw V13 rebuild, sidecar rebuild, role refinement, or data reorder is
authorized.

## Training and initialization

Reuse exactly:

- Raw V13 train role: 4,262 pairs, 7,777 unique endpoints, 72 scenes;
- checkpoint-selection role: 495 pairs, 924 unique endpoints, 8 scenes;
- N320 initialization only;
- base seed `20260712`;
- decoder seed `20260724`;
- temporal seed `20260725`;
- schedule seed `20260713`;
- AdamW, float32, no autocast, betas `(0.9, 0.999)`, epsilon `1e-8`,
  weight decay `1e-4`;
- separate evidence-head and encoder parameter groups;
- identical learning rates with the frozen 8,000-update horizon;
- independent evidence-head and encoder clip norm `1.0`;
- microbatch size 4 and four microbatches per optimizer update;
- the existing Camera losses and weights;
- zero JEPA objective, JEPA backward, target EMA, calibration, or navigation;
  and
- checkpoints at updates 100, 400, and 1,000.

N320 migration copies exactly the encoder, pixel head, and ground head. It
copies no multiresolution decoder, temporal residual, alignment, JEPA, target,
predictor, or occupancy state. No rejected or prior probe checkpoint may be
opened.

Maximum training is 1,000 optimizer updates and 16,000 pair presentations.
There is one seed, one attempt, no resume, no schedule extension, no
learning-rate or threshold search, and no second alignment topology.

The input schedule remains
`.generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4/`
`schedule.json`: 607,373 bytes, file SHA-256
`08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270`
and canonical content SHA-256
`274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15`.
Before model construction, the runner must reproduce these exact cumulative
ordered-index prefix SHA-256 values:

- 1,600 presentations:
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
- 6,400 presentations:
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
  and
- 16,000 presentations:
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

## Evaluation

The primary checkpoint-selection population remains all 924 unique endpoints:
495 warm and 429 cold. The warm-only view remains informational and cannot
control a checkpoint.

The cyclic within-family wrong-RGB arm replaces the complete previous/current
RGB history while retaining target calibration, target supervision, target
history-valid mask, and target five-value motion condition. This isolates RGB
dependence; motion context alone cannot satisfy the unchanged physical
evaluator.

Updates 100 and 400 are integrity and informational checks only. Update 1,000
is terminal. PASS is the strict conjunction:

- at least `1/9` complete physical scopes;
- at least `98/189` passed margins;
- total shortfall `< 41.01776266878769`;
- rough pixel balanced accuracy `> 0.8198594673963917`;
- rough ground balanced accuracy `> 0.647134926562893`; and
- rough depth p95 `< 0.9777327477931971 m`.

Equality fails. A terminal FAIL rejects this alignment mechanism with no
retry. A PASS licenses only a separately preregistered perception
qualification run; it does not itself qualify a checkpoint or authorize JEPA,
G2, navigation, held-out, production, or deployment work.

Because the learned predictor can in principle ignore the five motion
channels, a PASS supports this integrated motion-conditioned alignment block;
it does not by itself prove that motion conditioning rather than visual
correspondence caused the gain.

## Receipt correction

Before execution, the versioned ledger validator must admit both:

- the existing five fixed runtime leaves and Raw V13 descendants; and
- only development RGB paths matching the anchored pattern
  `.generated/go2_render_selected_v04/scenes/scene_[0-9a-f]{16}/rgb/`
  `frame_[0-9]{6}_env_[0-9]{2}.png`.

The validator must still reject every prior output root, sealed path, symlink,
unexpected render path, role crossing, hash mismatch, incomplete read, and
unpaired ledger record. A focused synthetic full-ledger test must exercise an
accepted development RGB runtime load and terminal rehash before source
freeze. This is an integrity correction, not a scientific change.

For the real attempt, the complete finalized on-disk ledger—including every
runtime load, every terminal rehash, and its terminal record—must pass the
source-frozen corrected `parse_partial_access_ledger` before `integrity_pass`
or any scientifically admissible PASS/FAIL result may be emitted. A parser
failure consumes and terminalizes the attempt as contract-invalid and
authorizes no retry, even if observed metrics exist.

## Lean source gate

Before any GPU or generated-input access:

1. implement only the alignment model, five-value condition builder, minimal
   runner/evaluator plumbing, corrected ledger validator, and focused tests;
2. prove update-zero and cold-history identity, exact parameter counts,
   private RNG restoration, bounded finite offsets, finite gradients through
   both RGB frames and every alignment tensor, strict N320 migration, and
   wrong-RGB condition retention;
3. use one small fully synthetic alignment microfit only as a wiring check;
4. prove the runner never materializes per-sample realized SE(2) into the
   model batch;
5. freeze a recursive source manifest;
6. obtain a different-agent source-review PASS; and
7. obtain a distinct exact one-attempt execution authorization.

Do not build another general audit framework. Do not query a GPU, open a
generated input, reserve an output root, or start training during source
preparation.

## Authority

This preregistration authorizes source-only implementation and review of the
named mechanism. It does not authorize generated-input access, accelerator
access, experiment reservation or execution, checkpoint access, calibration,
JEPA training, G2, navigation, held-out access, production, promotion, or
deployment.

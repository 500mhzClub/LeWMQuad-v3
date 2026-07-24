# Go2 RGB causal temporal perception V1 preregistration

Date: 2026-07-24

Status: **PREREGISTERED SINGLE-MECHANISM DEVELOPMENT FALSIFICATION; SOURCE
IMPLEMENTATION AND ONE FIXED 16,000-PRESENTATION RUN AUTHORIZED BY THE USER;
NO QUALIFICATION, JEPA, G2, NAVIGATION, PRODUCTION, DEPLOYMENT, OR HELD-OUT
AUTHORITY**

Governing evidence:

- full handoff commit:
  `99370af8d895a8de30a7d6a3ea663e080b535af8`;
- authority correction:
  `docs/lewm_go2_heldout_maze_authority_correction_2026-07-24.md`;
- rejected multiresolution V3 terminal audit commit:
  `adc5eaf65b99eafa908561c10475ed7c3135b678`; and
- rejected multiresolution V3 independent review commit:
  `c024e8aedef2977cf5e17ef5f64aed892fc2585e`.

The full handoff's historical V4 held-out instructions are superseded by the
authority correction. No current or historical sealed material may be opened
or used to guide this work.

## Question

Does one frame of deployable temporal context fix enough of the rough-motion
and dense physical-evidence failure to justify a full perception qualification
run?

This is a supervised encoder/evidence-head test. It is not JEPA training and
does not test the action-conditioned JEPA predictor.

## Exactly one changed mechanism

Keep the rejected V3 progressive multiresolution decoder as the fixed
predecessor architecture and add one learned causal temporal residual before
that decoder.

For a target frame, the existing shared encoder produces previous and current
token grids with shape `(B,192,16,16)`. The new block is:

```text
delta = current_tokens - previous_tokens
h = GroupNorm(4,8,eps=1e-5,affine=true)(
      Conv1x1(192->8, stride=1, padding=0, no bias)(delta))
h = GELU(h)
h = GELU(depthwise Conv3x3(
      8->8, groups=8, stride=1, padding=1, padding_mode=zeros, no bias)(h))
residual = Conv1x1(8->192, stride=1, padding=0, no bias)(h)
fused = current_tokens + history_valid * residual
```

Both GELUs use exact approximation mode `none`.
The fused grid enters the unchanged progressive multiresolution decoder.
The final `8->192` projection is initialized to exactly zero, so update zero
is exactly the predecessor model for both warm and cold observations.
`history_valid=false` is an exact identity bypass at every later update.
Only raw tokens, never fused tokens, enter a fixed-lag history buffer; there is
no recurrent accumulation. Warm history is exactly one complete `0.5 s`
command block, or five `0.10 s` ticks, earlier in the same environment,
episode, and reset epoch. `history_valid=true` only when that exact timing-safe
token exists. Reset, stream change, missing history, or irregular timing clears
the buffer and fails cold. Runtime qualification must use this identical lag
and lifecycle.

The block receives no requested-primitive proxy, median commanded delta,
realized simulator SE(2), exact pose, camera-attitude delta, depth input,
ground truth, scene geometry, or evaluator feedback. Motion is inferred from
the learned visual difference. This avoids treating the raw requested
primitive as verified executed-command telemetry and leaves
deployment-equivalent executed-command conditioning to the later JEPA and
controller stages.

## Frozen topology and capacity

The temporal block uses an isolated CPU generator with seed `20260725` and
restores the caller CPU RNG exactly. The input and depthwise convolution
weights use Xavier-uniform initialization with gain `1.0`; GroupNorm weight is
one and bias is zero; the final projection weight is exactly zero. It has:

| Component | Parameters | Parameter tensors |
|---|---:|---:|
| `192->8` input projection | 1,536 | 1 |
| GroupNorm affine | 16 | 2 |
| width-8 depthwise `3x3` | 72 | 1 |
| zero-initialized `8->192` projection | 1,536 | 1 |
| **Temporal total** | **3,160** | **5** |

The existing evidence head has `352,689` parameters in 26 tensors. The
successor evidence head therefore has `355,849` parameters in 31 tensors,
which remains 2,144 parameters below the predecessor ceiling of `357,993`.
With the unchanged encoder, the exact trainable total is `3,103,369`
parameters in 109 tensors.

## Frozen inputs, outputs, initialization, and training

Unchanged:

- Raw V13 roles and bytes: 4,262 train pairs and 495
  checkpoint-selection pairs;
- RGB normalization, `112x112` input, shared encoder, token shape, progressive
  decoder, pixel/ray head, ground head, geometry, differentiable rasterizer,
  output schemas, physical metrics, thresholds, and scope definitions;
- N320 initialization as the only tensor initialization input;
- base seed `20260712`, multiresolution decoder seed `20260724`, and training
  schedule seed `20260713`;
- strict copy of only N320 encoder, pixel head, and ground head state;
- zero copied temporal entries, zero copied predecessor dense-decoder entries,
  one hard target sync, and zero rejected-checkpoint opens;
- V6 tail-depth loss and all other Camera loss terms and coefficients;
- AdamW settings and the exact predecessor learning-rate function;
- float32, four real `B=4` microbatches per update, effective batch 16,
  evidence-head then encoder optimizer groups, and independent norm-1 clips;
- trainable prefixes `encoder.` and `evidence_head.`; and
- frozen BEV decoder, predictor, occupancy head, target encoder, and target
  BEV decoder, with zero JEPA objective, JEPA backward, and EMA updates.

For each train pair, the current frame uses the exact cold-start bypass and
the next frame uses the current frame as its valid predecessor. Both retain
the existing supervised Camera losses.
The previous and current RGB batches use two separate shared online-encoder
calls in that order, each with the unchanged predecessor batch shape. This
matches deployment reuse of a stored previous raw-token grid and avoids a
batch-shape-dependent change to update-zero current tokens. The target encoder
is never called.

The failed V3 runtime checkpoint and every earlier rejected adaptation
checkpoint remain prohibited initialization or inspection inputs.

The unchanged camera basis, ground height, supervision, and raster geometry
are simulator-derived development oracles. A PASS only licenses a later
development qualification; it does not make this perception checkpoint
deployment-ready or relax G7's deployment-equivalence requirements.

## Pair-aware checkpoint evaluation

The authoritative compatibility population remains the same 924 unique
checkpoint-selection endpoints, eight families, nine scopes, and 189 physical
margins:

- 495 unique pair-next endpoints have a genuine predecessor and use the
  temporal block;
- 429 current-only endpoints have no predecessor and use the exact cold-start
  bypass; and
- where an endpoint occurs in both roles, its unique pair-next predecessor is
  used.

The side-specific counts were derived in a metadata-only audit of the bound
Raw V13 indexes, without opening RGB or supervision payloads:

- `pairs.jsonl`: 5,172 rows, 6,207,286 bytes,
  SHA-256 `5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d`;
- `endpoints.jsonl`: 9,460 rows, 9,108,028 bytes,
  SHA-256 `34e47ddcc40ad8c1f092c73193d16773cf4dedae05e7f4f684abb385cc2c0d01`;
  and
- the runner must independently reproduce 495 warm, 429 cold, 66 appearing in
  both roles, and zero ambiguous multiple-predecessor endpoints before any
  model construction.

This retains the existing physical gate while testing the mechanism wherever
the frozen pair data supplies causal history. The evaluator must additionally
report the 495 warm targets as a stratified, informational temporal view during
the same forward pass. Warm-only numbers cannot accept a checkpoint or replace
the 924-endpoint decision.

For wrong-RGB evaluation, preserve the existing cyclic current-endpoint
mapping within each family and replace the complete RGB history associated
with the mapped endpoint. Target geometry, supervision, and warm/cold status
remain those of the target. A mapped endpoint without a predecessor supplies
its mapped current RGB as both history images. At update zero, matched and
wrong-RGB outputs must exactly reproduce the predecessor model because the
temporal residual is zero.

No calibration role is opened or evaluated.

## One bounded falsification

Exactly one scientific attempt uses the existing first 16,000 ordered pair
indices:

| Update | Pair-index presentations | Decision |
|---:|---:|---|
| 100 | 1,600 | integrity and informational snapshot |
| 400 | 6,400 | integrity and informational snapshot |
| 1,000 | 16,000 | terminal scientific decision |

The exact cumulative schedule-prefix SHA-256 values are:

- 1,600 presentations:
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
- 6,400 presentations:
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
  and
- 16,000 presentations:
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

Updates 100 and 400 have no numeric early-stop gate. Stop early only for an
integrity or operational failure, which consumes and terminalizes that
version. This document does not preauthorize a repair or replacement. There
is no second seed, resume from learned state, schedule extension, threshold
search, observer rerun, or automatic second temporal mechanism.

The update-1,000 PASS is the strict conjunction on the 924-endpoint
compatibility population:

- at least `1/9` complete physical scopes;
- at least `98/189` passed margins;
- total shortfall `< 41.01776266878769`;
- rough-motion pixel balanced accuracy `> 0.8198594673963917`;
- rough-motion ground balanced accuracy `> 0.647134926562893`; and
- rough-motion depth p95 `< 0.9777327477931971 m`.

Equality fails. A PASS licenses only a separately bounded perception
qualification run. A scientific FAIL terminates only this pure visual
fixed-lag token-difference mechanism; it does not answer a separately
preregistered future question about deployment-equivalent odometry, IMU, or
executed-command conditioning.

## Lean source gate

Before the run:

1. implement the separately versioned model, pair-aware evaluator, runner, and
   focused tests;
2. prove shapes, parameter counts, RNG restoration, update-zero identity,
   cold-start identity, finite gradients through both RGB frames, strict N320
   migration allowlist, frozen-state preservation, and pair-aware population
   counts;
3. run one small fully synthetic microfit only as a wiring check;
4. obtain a different-agent source review; and
5. bind the exact source and runtime inputs used by the run.

The user has authorized the bounded experiment in principle, but execution is
conditional on committed frozen source, a different-agent source-review PASS,
an exact runtime authorization binding, and the runner's fail-closed
preflight. Source preparation alone cannot reserve the attempt.

Do not add a generic experiment framework, data rebuild, new audit subsystem,
or navigation-runner redesign.

## Relationship to the repository goal

Even a PASS does not establish a fully learned navigation stack. Perception
must then pass full development qualification. The separated
action-conditioned JEPA stage must be trained with its own optimizer/clipping
boundary, any encoder change must be requalified, and the mandatory matched
no-JEPA development arm must run before a JEPA causal claim. Only then may a
fresh G2 candidate and the ordered G3 through G7 gates be attempted. A new
externally custodied G8 role may be created only after the complete stack is
frozen.

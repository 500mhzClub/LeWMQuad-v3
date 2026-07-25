# RGB overlapping tokenization V1 preregistration

Date: 2026-07-25

## User architecture decision

The user explicitly approved:

> Approve RGB-only Overlapping Tokenization V1.

This selects the RGB-only genuine-JEPA path and authorizes source-only
implementation and review of exactly the mechanism below. It is not execution
authority.

## Decision

Test exactly one materially different encoder-front-end mechanism: replace
the shared visual encoder's non-overlapping `7 x 7`, stride-seven RGB patch
projection with an `11 x 11`, stride-seven, padding-two projection whose
adjacent receptive fields overlap by four pixels.

The frozen comparison baseline is
`SharedObservableCameraRayJepaV5MultiresV1`: the static multiresolution
perception model used by RGB multiresolution perception V3. The terminated
causal-temporal V1 and causal-motion-alignment V1 mechanisms are not retained,
reopened, or used for initialization. Returning to the common static baseline
does not add a second mechanism; the sole new mechanism relative to that
baseline is overlapping input tokenization.

This remains supervised perception adaptation of the shared visual encoder
and evidence head. It is not JEPA training.

The completed motion-alignment V1 attempt is terminal. Its committed audit
records:

- `0/9` complete physical scopes;
- `111/189` passed margins;
- total shortfall `33.05143763708337`;
- rough pixel balanced accuracy `0.741796837511955`;
- rough ground balanced accuracy `0.621981002078303`;
- rough depth p95 `1.0227776646614073 m`; and
- `FAIL_BOUNDED_FALSIFICATION_MECHANISM_TERMINATED` with
  `integrity_pass=true`.

That attempt and every earlier attempt remain immutable. No checkpoint or
runtime artifact from a rejected perception attempt may be opened.

## Hypothesis

The current patch projection divides a `112 x 112` RGB image into disjoint
`7 x 7` cells before any learned cross-token processing. A small image-space
translation can therefore move an edge or thin obstacle abruptly between
independent token inputs. A post-encoder mechanism cannot alter this
boundary-sensitive first-layer support after token formation.

Overlapping receptive fields allow adjacent tokens to observe the same local
edge while preserving the existing token centers, token count, transformer,
multiresolution decoder, and physical output contract. The falsifiable claim
is that this upstream spatial continuity closes the frozen physical gate
within the same 16,000-presentation budget.

This is materially different from the terminated post-encoder mechanisms
because it changes the raw-pixel dependency graph: away from image boundaries,
one pixel initially reaches exactly one token under the old projection and can
reach as many as four tokens under the new projection. It adds 41,472
trainable cross-boundary weights to the online encoder.

## Sole scientific change

For both the online encoder and its frozen EMA target, change only:

```text
before:
  Conv2d(3, 192, kernel_size=(7,7), stride=(7,7), padding=(0,0),
         dilation=(1,1), groups=1, bias=True, padding_mode="zeros")
after:
  Conv2d(3, 192, kernel_size=(11,11), stride=(7,7), padding=(2,2),
         dilation=(1,1), groups=1, bias=True, padding_mode="zeros")
```

Keep the encoder configuration's `patch_size` and effective token stride at
seven. Record kernel size and padding as separate architecture fields. Do not
set `patch_size=11`.

For input coordinate `i`, the old one-dimensional support is
`7i ... 7i+6`. The new window is `7i-2 ... 7i+8`. Initialize the new weight to
zero and copy the old weight exactly into:

```text
new_weight[:, :, 2:9, 2:9] = old_weight
new_bias = old_bias
```

The noncentral ring therefore contains exactly `41,472` zero scalars at
initialization. The central copy contains exactly `28,224` scalars and the
bias copy contains exactly `192` scalars.

Both projections produce:

```text
patch map:      B x 192 x 16 x 16
patch tokens:   B x 256 x 192
CLS + patches:  B x 257 x 192
```

The token center remains pixel `7i+3`, including both boundaries. The
positional embedding remains `1 x 257 x 192`. The initial projection is
algebraically identical for finite inputs. Source tests must require exact
central copies and exact outer zeros, plus a tight float32 CPU output
tolerance; cross-backend bitwise equality is not required because convolution
implementations may reorder reductions.

After the first backward pass the outer ring is allowed to learn. Its
gradients can also change the shared encoder-group clip scale from update one.
That training consequence is part of the selected mechanism, not a claim of
trajectory identity.

No temporal residual, motion alignment, optical flow, recurrence, attitude
condition, direct-BEV head, depth input, auxiliary loss, calibration search,
threshold adapter, data change, or decoder change is permitted.

## Exact topology and parameter budget

The online patch weight changes from `28,224` to `69,696` scalars, an increase
of `41,472`. Its bias and parameter-tensor count do not change.

The exact trainable partition is:

- shared online encoder: `2,788,992` parameters in 78 tensors;
- unchanged multiresolution evidence head: `352,689` parameters in 26
  tensors; and
- total trainable: `3,141,681` parameters in 104 tensors.

The frozen target encoder receives the same shape change, so total model
storage increases by `82,944` scalars. The complete model contains `7,049,460`
parameters in 232 parameter tensors. BEV decoder, target BEV decoder,
predictor, occupancy head, CLS token, positional embedding, transformer,
evidence decoder, pixel head, and ground head shapes remain unchanged.

The literal new identities are:

```text
module:
  lewm.models.shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1
class:
  SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1
model_family:
  shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1
architecture_schema:
  lewm_go2_shared_jepa_v5_multires_overlapping_tokenization_v1_architecture
```

The architecture contract must override the inherited schema and family and
bind kernel, stride, padding, dilation, groups, bias, padding mode,
center-copy slice, token geometry, and parameter counts. The
otherwise-identical Shared-V5 config is not sufficient evidence of the new
architecture.

## Initialization and migration

Construct the complete unchanged multiresolution base first under base seed
`20260712` and decoder seed `20260724`. Only after all existing modules have
been constructed may the two patch projections be replaced. Save and restore
the caller CPU RNG around replacement-module construction so no unrelated
initialization stream moves.

The sole tensor initialization input remains the exact N320 fit model. Direct
strict loading of its `7 x 7` patch weight into the new `11 x 11` parameter is
forbidden and would fail on shape. The versioned migration adapter must:

1. exact-copy the other 77 encoder state entries, including the
   `192`-scalar patch bias;
2. zero the expanded patch weight;
3. exact-copy the old `7 x 7` weight into `[2:9, 2:9]`;
4. exact-copy the existing six pixel-head and ground-head entries;
5. copy no multiresolution decoder, rejected probe, temporal, alignment, JEPA,
   predictor, occupancy, optimizer, or target state; and
6. hard-sync the target encoder from the migrated online encoder exactly once.

The migration receipt must distinguish 83 exact-copy entries from the one
transformed entry `encoder.patch_embed.weight`, while retaining 84
N320-derived entries in total. It must bind source and destination shapes,
center slice, central scalar count, exact-zero outer scalar count, both N320
hashes, caller-RNG restoration, target hard-sync count, and zero rejected
checkpoint opens.

The new target projection must be frozen and in evaluation mode. Start a
fresh optimizer; old optimizer moments are forbidden.

## Frozen baseline and loss binding

The exact executed static V3 science baseline is frozen at commit
`97824b29ce9f4789b18e7a0cb5bc36f2feac1704`:

- contract
  `lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py`,
  SHA-256
  `3553810c79686f642a30fdfd0d2ff6ae047a97ea65c1366cae4cb3231e44e669`;
  and
- model
  `lewm/models/shared_observable_camera_ray_jepa_v5_multires_v1.py`,
  SHA-256
  `a63da1137539953b2f40d184def1652ae05f63d7b434084b1a91787e1fc83d0b`.

The exact Camera objective is
`observable_camera_ray_v4_tail_depth_loss_v4` from
`lewm/models/shared_observable_camera_ray_jepa_v5_`
`protected_camera_adaptation_v4_tail_depth.py`, SHA-256
`6fc0a114386ee2fb0ae98704a970d38a7194db192283b904138015498fb02384`.
It retains the hierarchical first-hit, ground-clear-distance
state-balanced-BCE, derived-raster hierarchical-BCE, and all-cell raster-NLL
terms and coefficients; substitutes the fixed p95/CVaR tail-depth term in the
existing offset slot; and averages current and next frame totals with weights
`0.5` and `0.5`. The raster-NLL coefficient remains `0.25`, tail fraction
`0.05`, and depth-p95 ceiling `0.25 m`.

The new `one_science_delta` literal is
`overlapping_rgb_patch_tokenization_relative_to_static_multires_v3_only`.
Every field of the frozen static `science_contract()` must remain equal except
that literal, the new model/runtime identity, patch topology and architecture
contract, expanded migration receipt, and exact parameter counts named here.

## Preserved data and training contract

Reuse exactly:

- Raw V13 train role: 4,262 pairs, 7,777 unique endpoints, 72 scenes;
- checkpoint-selection role: 495 pairs, 924 unique endpoints, 8 scenes;
- static independent current-frame and next-frame `forward_frame` calls;
- N320 initialization only;
- base seed `20260712`;
- decoder seed `20260724`;
- schedule seed `20260713`;
- AdamW in float32 without autocast, betas `(0.9, 0.999)`, epsilon `1e-8`,
  and weight decay `1e-4`;
- separate evidence-head and encoder parameter groups;
- the frozen learning-rate schedule with the existing 8,000-update horizon;
- independent evidence-head and encoder clip norm `1.0`;
- microbatch size 4 and four microbatches per optimizer update;
- the exact bound Camera loss above and the frozen static V3 geometry,
  supervision, rasterization, wrong-RGB mapping, physical evaluator, and
  thresholds;
- frozen BEV decoder, target encoder, target BEV decoder, predictor, and
  occupancy head; and
- checkpoints at updates 100, 400, and 1,000.

There is zero JEPA objective, JEPA backward, target EMA update, calibration,
navigation, or held-out operation in this probe.

Maximum training is 1,000 optimizer updates and 16,000 pair presentations.
There is one seed, one attempt, no resume, no schedule extension, no
learning-rate change, no threshold search, no kernel/padding variant, no
second seed, and no replacement attempt.

The frozen input schedule remains
`.generated/go2_shared_observable_camera_ray_jepa_v5/matched_training_v4/`
`schedule.json`: 607,373 bytes, file SHA-256
`08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270`
and canonical content SHA-256
`274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15`.
Before model construction, the runner must reproduce these cumulative ordered
index-prefix SHA-256 values:

- 1,600 presentations:
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
- 6,400 presentations:
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
  and
- 16,000 presentations:
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

No Raw V13 rebuild, sidecar rebuild, role refinement, data reorder, or prior
runtime-root access is authorized.

## Fixed compute and operation cap

The exact attempt may use one discrete R9700 accelerator and no other
accelerator, multi-device, distributed, or observer compute. The terminal
maximum is:

- one optimizer construction;
- 1,000 optimizer steps;
- 16,000 pair-index presentations;
- 4,000 Camera objective evaluations;
- 4,000 backward calls;
- 1,000 evidence-head clip invocations;
- 1,000 encoder clip invocations;
- zero global clip invocations;
- three inline checkpoint-selection evaluations, at updates 100, 400, and
  1,000;
- three checkpoint snapshots and three immutable metric sidecars;
- zero observer evaluation reruns;
- zero JEPA objectives or JEPA backwards; and
- zero EMA updates after the one initialization hard sync.

The operation cap is conjunctive with the 16,000-presentation cap. A faster
device, unused wall-clock budget, promising curve, or incomplete update does
not authorize more operations.

## Evaluation and terminal decision

The primary checkpoint-selection population remains all 924 unique endpoints
and the same nine physical scopes and 189 margins. The existing cyclic
within-family wrong-RGB arm and all target calibration and supervision
semantics remain unchanged.

Updates 100 and 400 are integrity and informational checks only. Update 1,000
is terminal. PASS is the strict conjunction:

- at least `1/9` complete physical scopes;
- at least `98/189` passed margins;
- total shortfall `< 41.01776266878769`;
- rough pixel balanced accuracy `> 0.8198594673963917`;
- rough ground balanced accuracy `> 0.647134926562893`; and
- rough depth p95 `< 0.9777327477931971 m`.

Equality fails. Integrity failure terminates the attempt. A scientific FAIL
rejects Overlapping Tokenization V1 with no retry, repair, resume, second
seed, extension, or nearby kernel variant.

A PASS licenses only a separately preregistered perception-qualification
stage. It does not itself qualify a checkpoint or authorize JEPA, G2,
navigation, held-out, production, promotion, or deployment work.

## Attempt custody and admissibility

The launcher must validate the exact committed source manifest, independent
source review, and one-attempt execution authorization before any accelerator
query or scientific input open. A successful no-tensor hardware preflight
does not itself reserve or authorize execution.

Reservation requires the exact output root to be absent and atomically creates
one mode-`0700` attempt. Reservation consumes the sole attempt. Every exception
or integrity failure after reservation must publish the applicable normal,
contract-invalid-ledger, or pre-ledger failure receipt and terminalize the
attempt; none authorizes repair or retry.

Every runtime-input open attempt and accepted/rejected outcome must be
hash-chained and durably recorded. Before any scientifically admissible
`integrity_pass`, PASS, or FAIL is materialized, the complete finalized
on-disk ledger—including all runtime loads, terminal rehashes, and the terminal
`RUNTIME_INPUT_ACCESS_FINALIZED` record—must pass the source-frozen corrected
full parser. Parser failure or incomplete access evidence is a terminal
contract-invalid result with no retry, even when metrics exist.

Terminal publication must seal every file mode `0444` and every directory
mode `0555`, with no symlink anywhere in the output inventory. A provisional
checkpoint metric cannot claim integrity or control promotion before the
finalized-ledger parse.

## Preservation of the fully learned JEPA path

The overlap model keeps the JEPA tensor interface unchanged: online and target
encoders still emit 256 patch tokens of width 192, and the BEV decoder,
predictor, masks, losses, and EMA equations remain shape-compatible.

If and only if perception later qualifies, a separately reviewed JEPA loader
must instantiate this exact new model family and reject any `7 x 7` downcast
or old-family reconstruction. It must validate the model family and overlap
architecture-contract hash (`kernel=(11,11)`, `stride=(7,7)`,
`padding=(2,2)`, `dilation=(1,1)`, `groups=1`, `bias=true`,
`padding_mode=zeros`) before loading the qualified online encoder and
evidence-head state as the sole
perception source. It must bind the perception-qualification receipt and
state hash, load no stale target, old BEV, predictor, or optimizer state, and
hard-sync the same-shape EMA target before its first target forward.

The planned separated JEPA stage keeps qualified perception frozen while
initializing and training the BEV decoder and action-conditioned predictor
under their own preregistered seed, optimizer, and clipping contract. If a
future preregistration instead updates the encoder under predictive and
anti-collapse objectives, that invalidates the prior perception checkpoint
and requires complete physical selection and calibration requalification
before G2. Its pre-G2/G2 artifacts must carry the new family, architecture
hash, qualification provenance, and source/data manifests. Existing canonical
V5 loaders that instantiate or hardcode the old family cannot strict-load the
enlarged patch weight and must not be used unchanged.

The existing matched-training runner must not be reused unchanged because it
constructs the old model family, initializes directly from N320, and applies a
different joint optimization boundary. A future adapter may reuse its
data/loss/evaluation utilities only.

The later development stage must run the mandatory matched no-JEPA arm
required by the governing authority correction. It must share initialization,
data, architecture, optimizer schedule, presentation count, selection update,
and downstream protocol; may differ only in its preregistered predictive
objective or treatment; and may not select its own checkpoint. Any causal
attribution of generalization improvement to JEPA also depends on this matched
comparison.

The final deployed stack must bind an action-conditioned JEPA predictor or
latent rollout that causally affects candidate or action scores. A predictor
used only as a training loss, diagnostic, or discarded representation does
not satisfy the repository's fully learned JEPA goal.

No downstream loader or JEPA-training implementation is required before this
probe earns continuation.

## Lean source gate

Before any generated-input, checkpoint, or accelerator access:

1. implement one additive overlap model without editing the generic
   `VisionEncoder` or frozen predecessor models;
2. implement a static runner adapter with no temporal or motion inputs;
3. prove exact geometry, state-key inventory, parameter counts, center-copy,
   zero ring, caller-RNG restoration, target freeze, hard sync, and N320
   receipt structure using synthetic source tests;
4. prove finite nonzero outer-ring gradients, unchanged token/JEPA shapes, one
   EMA formula step, and same-family checkpoint roundtrip using accelerator-
   hidden synthetic tests;
5. prove all science constants except identity, patch topology, migration
   receipt, and parameter counts equal the frozen static contract;
6. exercise one synthetic runner prepare/forward/evaluate path and corrected
   terminal receipt boundary;
7. freeze a recursive source manifest;
8. obtain a different-agent source-review PASS; and
9. obtain a distinct exact one-attempt execution authorization.

Do not build a new general audit framework, generic checkpoint system,
compatibility matrix, or downstream JEPA runner before PASS.

## Authority

This preregistration and the user's architecture decision authorize only
source implementation, accelerator-hidden synthetic verification, source
closure, and independent review of the named mechanism.

They do not authorize opening generated inputs, RGB, datasets, N320 or any
other checkpoint/tensor, prior runtime outputs, rejected checkpoints,
protected material, or held-out material. They do not authorize querying an
accelerator, reserving or mutating an experiment output root, training,
execution, perception qualification, JEPA training, G2, navigation,
production, promotion, deployment, or held-out evaluation.

Execution requires a separately committed, exact one-attempt authorization
after the implementation and source review are frozen.

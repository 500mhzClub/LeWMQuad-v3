# Go2 RGB recurrent patch-memory temporal JEPA V1 preregistration

Date: 2026-07-31

Status: **PREREGISTERED; SOURCE IMPLEMENTATION AND EXECUTION ARE NOT YET AUTHORIZED**

## Purpose and claim boundary

This experiment asks one development-only question: after qualifying learned
single-frame spatial tokens, can one jointly trained, action-conditioned JEPA
use an ordered RGB history to predict the learned representation of the next
frame better than copying the present or ignoring history/action?

This is the temporal extension requested for the learned navigation stack. It
is not a navigation, planner, G2, production, sealed, or held-out-maze test.
A pass qualifies a short-horizon learned temporal/world-memory substrate for
later navigation integration. It does not itself establish maze completion or
held-out generalization.

## Predecessor evidence

The only learned predecessor is RGB single-frame multiblock masked-spatial
JEPA V1.

- Scientific-result commit:
  `6e1ae4496f36a983935aa9f8a377948dffd9a23b`
- Scientific-result path:
  `docs/lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_scientific_result_2026-07-31.json`
- Result file SHA-256:
  `92b2c23545b5f2b42f81b392359e7ad1c58e4773053d02df39a2df54509228ef`
- Result content SHA-256:
  `59e55e2ef718c670c1251062572ef4d6bb76f7ff51582fe599d7ebe534ebefba`
- Result byte count: `7685`
- Terminal success path:
  `.generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1/attempt_v1/success.json`
- Terminal success file SHA-256:
  `40e4d2e595609d4dff379634880caf122a9b1523ee555c3d7c3ae683d5e687b2`
- Terminal success content SHA-256:
  `f93162027bf4ea0f2c76c29708bd4342119c4ece010308b5d1164f591986f2fc`
- Selected checkpoint: `snapshots/update_1000.pt`
- Selected checkpoint SHA-256:
  `f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873`
- Selected checkpoint byte count: `52282877`
- Predecessor status: `PASS_PERCEPTION_QUALIFIED`

The predecessor checkpoint may be opened exactly once only under a later
reviewed one-shot authority. From `model_state_dict`, migration accepts exactly
the complete key sets with prefixes `encoder.`, `predictor_blocks.`,
`predictor_norm.`, and `predictor_output.`, plus the exact keys
`predictor_position` and `predictor_mask_token`. Predecessor
`target_encoder.*`, `ema_update_count`, optimizer state, RNG state, accounting,
and EMA lag are rejected from migration. The new target encoder is
hard-synchronized from the migrated online encoder and its EMA counter starts
at zero. The accepted keys must equal the corresponding new-model inventory;
missing or extra accepted-prefix keys are terminal.

The V18/V4 delay-line result is negative evidence, not a warm start. It reduced
training loss but predicted future latents much worse than persistence and
showed weak action/history sensitivity. No V18 model, checkpoint, object-space
lift, physical loss, delay line, or local Conv3d mechanism is inherited.

## Frozen development inputs

Only corrected-H6 development roles and the existing development-only place
retention panel are in scope.

- H6 train index:
  `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/train.jsonl`
  - SHA-256:
    `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77`
  - bytes: `10328000`
  - rows: `16000`
- H6 validation index:
  `.generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/val.jsonl`
  - SHA-256:
    `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`
  - bytes: `1317888`
  - rows: `2048`
- RGB root: `.generated/datagen_full/render_textured_v03`
- Place manifest:
  `.generated/go2_memory_role_place_triplet_index_v1/manifest.json`
  - SHA-256:
    `a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca`
  - bytes: `42308`
- Place checkpoint-selection index:
  `.generated/go2_memory_role_place_triplet_index_v1/checkpoint_selection.jsonl`
  - SHA-256:
    `a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0`
  - bytes: `473508`

Index validation necessarily parses the seven RGB path strings and six action
IDs in each H6 metadata row. Model-visible tensors are narrower:

- online context RGB: `rgb[0]`, `rgb[1]`, `rgb[2]`;
- model-visible actions: `actions[0]`, `actions[1]`, `actions[2]`;
- stop-gradient EMA target RGB: `rgb[3]`;
- `rgb[4:7]` is never opened or decoded;
- `actions[3:6]` is never converted to a model input tensor.

Train and validation scenes and RGB leaves must remain disjoint.

## Exact 4,000-sequence train schedule

The training schedule covers all 1,000 H6 train scenes and is family-equal.

For each family in lexicographic order:

1. group rows by `scene_id` and sort each scene's rows by original row index;
2. select the first row from every scene;
3. enumerate remaining rows round-robin by within-scene rank and then
   lexicographic `scene_id`;
4. take enough remaining rows to reach exactly 500 rows for that family.

Concatenate the eight 500-row selections, then sort rows by
`(SHA256("lewm-go2-temporal-patch-gru-v1/train/" + decimal_row_index),
row_index)`.

The canonical compact-JSON SHA-256 of the resulting 4,000 integer row indices
must be:
`853aad503738baed6bfbba18f3ac55c4715d7f164e71d4d0ef96c861befee7fc`.

Runtime assertions require 4,000 rows, 1,000 scenes, 500 rows per family, and
all nine action IDs at each of the three visible action positions.

Before source execution authority or GPU/output reservation, one separately
content-bound metadata-only authority must reserve the initially absent output
root `.generated/go2_rgb_recurrent_patch_memory_temporal_jepa_v1_metadata_preflight/attempt_v1`
and run exactly one preflight. It reconstructs the training schedule, sentinel,
and wrong-history donor mappings directly from the two bound H6 JSONL indices;
verifies their registered hashes, row/scene/family counts, action coverage,
HOLD eligibility, and train/validation disjointness; and writes only
`receipt.json`. The receipt schema records the authority/preregistration and
input identities, every registered computed identity/count, metadata-row and
path-string counts, zero RGB/checkpoint/model/GPU accesses, terminal status,
and a content hash computed with that field omitted. This preflight may parse
path and action strings but may not open RGB leaves, the predecessor
checkpoint, navigation data, G2, held-out, sealed, or calibration material.
The later training authority must bind a `PASS_METADATA_PREFLIGHT` receipt by
exact path, file SHA-256, content SHA-256, and byte count.

The 4,000 hashed rows are consumed exactly once in that order. Update `u`
uses schedule offsets `10*(u-1)` through `10*u-1`, divided into five
contiguous `B=2` microbatches. No row repeats and no alternate sampling or
shuffle is permitted.

## Model

The model is one jointly trained JEPA, not a frozen encoder plus a separately
trained probe.

### Warm-started spatial components

- online ViT encoder: `112x112`, `7x7` patches, `16x16 = 256` tokens,
  width `192`, depth `6`, heads `6`;
- same-frame predictor position table, mask token, two width-192 predictor
  blocks, predictor norm, and predictor output projection;
- target encoder: fresh hard-synchronized stop-gradient copy, EMA momentum
  `0.996`.

All online encoder and predictor parameters remain trainable.

### New learned temporal memory

- learned action embedding: `9 x 192`;
- learned time embedding: `3 x 192`;
- one shared single-layer GRU with input width `192`, hidden width `192`, and
  standard learned parameter initialization; its runtime hidden state is an
  exact-zero tensor at the start of every sequence;
- no fixed FIFO, local convolution, object-space lift, pooling, pose input,
  geometry transform, persistence initialization, or hand-written memory.

For a batch of three online context frames:

1. encode all frames through the jointly trainable full-frame spatial encoder;
2. reshape to `(B, 3, 256, 192)`;
3. add the corresponding learned time and outgoing-action embeddings;
4. run the shared GRU over the three time steps independently for each of the
   256 already globally contextualized patch streams;
5. retain the final `256 x 192` recurrent memory map.

### Future-token decoder

For each row, use the predecessor mask implementation unchanged with exact
ASCII namespace
`lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1|mask|20260801`,
role `train` or `val`, and the original H6 row index to select 64 future target
positions. Concatenate:

- the 256 recurrent memory tokens plus the warm-started predictor positions;
- 64 future queries formed from the warm-started mask token plus the gathered
  predictor positions.

Run the resulting 320 tokens through the warm-started two predictor blocks and
predict only the final 64 query tokens through the warm-started norm/output.
There is no direct current-token scatter or current-output bypass.

The detached EMA encoder sees the complete `rgb[3]`; only its 64 registered
target tokens enter the future JEPA loss. `rgb[3]` never enters the online
encoder.

A new descriptor-safe loader enforces the model-visible boundary. For factual
rows it may open only positions `rgb[0]` through `rgb[3]`; for wrong-history
donors it may open only donor positions `rgb[0]` and `rgb[1]`. It records
position-specific requested/opened counters and must prove zero opens for
factual or donor `rgb[4:7]` and donor `rgb[2:7]`. It must not reuse the V18 or
V27 evaluators, whose broader or incompatible access patterns are out of
scope.

## Objective and optimizer

There is exactly one training objective: normalized half-squared JEPA loss
between the 64 predicted future tokens and matching detached EMA `rgb[3]`
tokens. The spatial predictor is not a separately trained probe: its mask
token, positions, blocks, norm, and output participate directly in this future
JEPA loss together with the online encoder and GRU.

There is no RGB reconstruction, action classification, pose, geometry,
physical-label, route, success, collision, or policy loss.

One AdamW optimizer uses constant learning rates:

- online encoder: `3e-5`;
- warm-started spatial predictor: `1e-4`;
- new action/time embeddings and GRU: `3e-4`;
- betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`;
- one global gradient clip at `1.0`;
- one optimizer step and one target EMA step per update;
- exact float32, no scheduler, no AMP.

Initialization seed `20260731` applies only to new action/time embeddings and
GRU parameters. It does not alter the inherited mask namespace.

## Batch and presentation accounting

- five microbatches of `B=2` sequences per update;
- ten sequence rows per update;
- each sequence makes exactly four logical RGB-frame decoder requests:
  three online context frames and one EMA future-target frame;
- per update: 10 sequence presentations, 40 logical RGB-frame presentations,
  30 online encoder frame calls, 10 EMA target encoder frame calls, five
  microbatch graphs/backward calls, one global clip, one optimizer step, and
  one EMA step;
- maximum `400` updates, `4,000` unique H6 sequence rows, and exactly `16,000`
  logical RGB-frame presentations if update 400 is reached; repeated physical
  RGB leaf identities across different sequence rows remain separate logical
  presentations and decoder requests;
- update-400 branch totals: `12,000` online encoder frame calls, `4,000` EMA
  target encoder frame calls, `2,000` microbatch graphs/backward calls, and
  `400` clips/optimizer/EMA steps;
- no retry and no resume.

For any early terminal update `u`, exact training accounting is: `10*u`
sequence rows, `40*u` logical RGB-frame presentations, `30*u` online encoder
frame calls, `10*u` EMA target encoder frame calls, `5*u` microbatch graphs
and backward calls, and `u` clips, optimizer steps, and EMA steps. Validation
and place-panel accesses are reported separately and never counted as training
presentations.

Complete continuation checkpoints are emitted only at nonzero registered full
observations. They exist for reproducibility and do not authorize extending
this V1 beyond its cap.

## Frozen validation identities

The 256-row sentinel is selected from the ordered H6 validation rows as
follows. For each family, group rows by scene and sort each scene by row index.
If the family has at least 32 scenes, take the first row from the first 32
lexicographic scenes. Otherwise take the first row from every scene, then add
rows round-robin by within-scene rank and lexicographic scene until reaching
32. Combine the eight family panels and sort by
`(SHA256("lewm-go2-temporal-patch-gru-v1/sentinel/" + decimal_row_index),
row_index)`.

- sentinel rows: `256`;
- sentinel scenes: `144`;
- rows per family: `32`;
- canonical compact-JSON row-index SHA-256:
  `615287ba03169cfb390626d38163836d92ad1750fd5a74885e9105e56f5152ee`.

The same sentinel indices, future masks, wrong-history donors, and wrong-
action eligibility are used at updates 0, 50, and 100.

For every full-validation row `r`, the wrong-history donor is the candidate
`d` with the minimum tuple
`(SHA256("lewm-go2-temporal-patch-gru-v1/wrong-history/" + r.index + "/" +
d.index), d.index)` among rows with the same family and a different scene.
The canonical compact-JSON SHA-256 of donor indices in full-validation row
order is
`7bab828cc1170edc39b13e8277d3a739f97106eba4d88bed5631b27a5111823c`;
in sentinel order it is
`6d8978266e466ed191c978819d2aaa79e17773d32e4e17ac0a2542c0bb542dd4`.
An unavailable or repeated-scene donor is terminal.

HOLD is exact action ID `6`. The full wrong-action panel contains `1971`
eligible non-HOLD rows, with family row counts
`(248,233,247,246,248,249,248,252)` in lexicographic family order and scene
counts `(22,23,15,38,15,7,23,7)`. The sentinel contains `248` eligible rows,
with family row counts `(32,30,30,30,31,31,32,32)` and scene counts
`(22,22,15,30,15,7,23,7)`. Any mismatch is terminal.

## Evaluation controls

All control donors are deterministic, validation-only, same-family, and
different-scene where applicable. Scores are normalized half-squared token
energies on identical target positions.

1. `persistence`: matching EMA `rgb[2]` tokens copied as the prediction for
   EMA `rgb[3]`.
2. `current_only_reset`: use the identical encoder, embeddings, GRU, and
   decoder, set the GRU initial hidden tensor to exact zeros, supply only the
   factual full-frame `rgb[2]` token map with time-embedding index `2` and
   factual `actions[2]`, run exactly one GRU step, and decode normally.
3. `wrong_history`: replace only `rgb[0:2]` and their outgoing actions with a
   donor selected by the frozen mapping above; preserve the factual `rgb[2]`,
   `actions[2]`, and target.
4. `wrong_action`: preserve factual visual history but replace `actions[2]`
   with `(actions[2] + 1) mod 9`. Factual HOLD rows remain diagnostic but are
   excluded from the wrong-action qualification aggregate.

For every control, compute row energy, scene-equal then family-equal macro
means, real/control ratio, paired advantage, fixed 2,000-replicate bootstrap
lower 95% bound, and positive-family count.

Bootstrap seeds are `20260811` for persistence, `20260812` for current-only
reset, `20260813` for wrong history, and `20260814` for wrong action. Each of
2,000 draws independently resamples the observed scene advantages with
replacement inside each family, preserves that family's observed scene count,
averages scenes within family, then averages the eight family means. The lower
95% bound is sorted draw index `50` (zero based). No row-level resampling or
alternate seed is permitted.

Evaluation also records:

- recurrent-state effective rank, variance, and temporal change;
- predicted/target token effective rank and cross-sample variance;
- exact action and ordered-history prediction/state deltas;
- target isolation, gradients, optimizer, EMA, finite-state, and access
  accounting;
- the predecessor spatial control, raw-health, and place-retention panel.

For recurrent memory (`T=256`) and predicted/target tokens (`T=64`), health is
computed on an `N x T x 192` tensor. Accumulate the per-position sum and the
flattened second moment in float64; center each position across the `N` rows;
form the symmetrized `192 x 192` covariance with denominator `N*T-1`; clamp
its eigenvalues at zero; normalize them by their sum; and define effective
rank as `exp(-sum(p*log(max(p,1e-12))))`, or zero for zero total variance.
Cross-sample variance is the position-centered square sum divided by
`N*T*192`. The recurrent temporal-change diagnostic is mean squared change
between the GRU output after context steps 2 and 3. All populations use the
entire registered evaluation panel, never a fitted projection.

Open-loop H2-H4 prediction, navigation, and physical metrics are intentionally
excluded from V1.

## Qualification gates

At a full validation observation, all integrity/accounting checks must pass
and:

- real/persistence ratio `<= 0.95`;
- real/current-only-reset ratio `<= 0.95`;
- real/wrong-history ratio `<= 0.95`;
- real/wrong-action ratio `<= 0.98` on non-HOLD rows;
- paired bootstrap lower 95% bound is `> 0` for every required control;
- positive-family count is at least `6/8` for persistence, reset, and wrong
  history, and at least `5/8` for wrong action;
- recurrent-memory effective rank is at least `4`;
- prediction effective rank and cross-sample variance each retain at least
  `25%` of the matching EMA targets;
- each predecessor spatial control retains its exact qualifying gate: ratio
  `<=0.90`, bootstrap lower 95% bound `>0`, and at least `6/8` positive
  families;
- predecessor online and EMA raw-health measures each retain at least `75%`
  of temporal update zero;
- the predecessor scene-equal `retrieval.chance_multiple` and target place-key
  effective rank each retain at least `80%` of temporal update zero;
- every successful training-update receipt reports finite nonzero gradient
  tensors for the online encoder, GRU/action/time memory, and predictor,
  exactly zero target gradient tensors, exact optimizer membership, one clip,
  one optimizer step, and one EMA step. Every observation also requires finite
  model/state tensors, a frozen eval-mode target, and EMA count equal to the
  completed update count.

These are temporal/perception-development gates only. No route success, SPL,
collision, goal-selection, calibration, G2, navigation, or held-out metric is
required here.

## Observations, continuation, and selection

- update `0`: complete temporal validation on all 2,048 rows, the exact
  256-row sentinel slice from that same result, and the complete predecessor
  spatial/raw-health/place baseline;
- updates `50` and `100`: the same frozen 256-row sentinel only;
- updates `200` and `400`: complete 2,048-row temporal validation plus the
  complete predecessor spatial/raw-health/place panel;
- immutable complete checkpoints at updates `200` and `400` only.

The mandatory observation-survival predicate is: exact registered access and
training accounting through that update; all evaluated tensors, states, and
metrics finite; target encoder frozen, in eval mode, with zero gradients; EMA
count equal to completed updates; the latest training-update gradient and
optimizer receipt satisfying the exact gate above; recurrent-memory effective
rank at least `4`; and prediction effective rank and cross-sample variance each
at least `25%` of the matching EMA targets. At full observations it additionally
requires the complete predecessor spatial gates and the registered raw-health
and place-retention thresholds. At update zero, where training gradients and
relative retention do not yet exist, it instead requires the exact migrated
predecessor spatial gate, finite noncollapsed baseline health, zero EMA count,
and the remaining applicable clauses. Failure of this predicate at any
observation is terminal and cannot enter a continuation comparison.

“Sentinel passes all temporal thresholds” means the sentinel-computable subset:
all four registered control ratio, bootstrap, and family-breadth thresholds,
plus the recurrent-memory and prediction health thresholds in the mandatory
observation-survival predicate. It does not imply unavailable predecessor
spatial/raw-health/place measurements.

For continuation only, the exact causal-control set is
`{current_only_reset, wrong_history, wrong_action}`; persistence is a required
predictive baseline but is not included in early causal-trend aggregation.
`worst causal ratio` means the maximum lower-is-better primary ratio over that
set. `minimum causal family breadth` means the minimum positive-family count
over that set.

Continuation rules:

- update 50 compares sentinel to update-zero sentinel and continues only with
  the mandatory observation-survival predicate and either (a) a decrease of
  at least `0.01` in any causal-control ratio or (b) an increase in minimum
  causal family breadth;
- update 100 requires the mandatory observation-survival predicate, compares
  sentinel to update-50 sentinel, and continues only if (a) worst causal ratio
  decreases by at least `0.02`, (b) minimum causal family breadth increases,
  or (c) the sentinel passes all temporal thresholds;
- update 200 stops successfully if the complete gate passes;
- otherwise update 200 requires the mandatory observation-survival predicate,
  compares full validation to update-zero full validation, and continues only
  if either (a) every causal-control ratio is below `1.0`, or (b) worst causal
  ratio decreases by at least `0.02` **and** minimum causal family breadth
  increases;
- update 400 is terminal.

If update 200 qualifies, select update 200 and stop. Otherwise only update 400
can qualify and be selected. A nonqualifying terminal result is a scientific
failure of this V1; it does not authorize retry, resume, silent threshold
changes, or an architecture variant.

## Custody and authorization

- source implementation and synthetic tests occur without dataset payload,
  RGB, checkpoint, GPU, navigation, G2, held-out, or sealed access;
- execution requires a reviewed recursive source manifest, independent source
  review, exact narrow clean-export certification, and a content-bound one-
  shot authority;
- the predecessor checkpoint remains runtime-only and is never copied into
  the clean source root;
- only one visible AMD Radeon AI PRO R9700 is permitted;
- output root must be absent when authority is issued and before reservation;
- any access, integrity, numerical, or accounting exception consumes the
  attempt and produces a terminal no-retry/no-resume receipt;
- held-out/sealed, navigation, G2, calibration, production, promotion, retry,
  and resume remain closed.

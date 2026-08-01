# Existing-pool factual action-learning three-arm V1 preregistration — 2026-08-01

Date: 2026-08-01

Status: **source-and-contract preregistration only; not execution authority**.
This document freezes the scientific comparison and its interpretation before
runtime access. It does not authorize opening the H6 indices, RGB leaves,
packed frames, predecessor checkpoint, GPU runtime, or generated artifacts. It
does not authorize training, evaluation, checkpoint creation, retry, resume,
promotion, G2-G8 work, held-out access, or sealed access.

No source hashes or runtime input hashes are frozen here. A later exact plan,
independent source review, and explicit execution authority must bind the
complete committed source closure, interpreter, runtime inputs, output root,
and command before an attempt may be reserved.

Interpretation basis:
`docs/lewm_go2_world_model_existing_pool_interpretation_correction_2026-08-01.md`.

## 1. Registered question

Does the existing corrected-H6 development pool contain learnable **factual
requested-candidate-action information** for the frozen recurrent patch-memory
temporal-JEPA substrate?

Operationally, after matched training on the same rows and shared frozen visual
encodings, does the aligned requested candidate action improve next-frame
latent prediction on scene-disjoint factual validation rows relative to both:

1. a candidate-blind model with the candidate-action embedding removed; and
2. a model trained with a within-family, marginal-preserving derangement of the
   candidate action?

The experiment addresses factual requested-action learnability only. It is not
a counterfactual intervention experiment and cannot validate predictions for an
action that was not executed from the identical physical state.

## 2. Registered claims and non-claims

### 2.1 Maximum positive claim

If every registered gate passes, the maximum claim is:

> On the frozen corrected-H6 train/validation roles, aligned requested
> candidate actions improve factual scene-disjoint future-latent prediction for
> the frozen-encoder temporal-predictor protocol relative to matched
> candidate-blind and marginal-preserving deranged-action controls.

This is called `PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY`.

### 2.2 Claims this experiment can never make

No outcome may be represented as evidence that:

- an untaken action is predicted correctly from the same physical state;
- requested actions equal executed or unclipped commands;
- the complete approximately 3 TB pool was trained or exhausted;
- the encoder remains physically qualified after some other training protocol;
- the temporal architecture is generally sufficient for world modeling;
- a world-model score causally changes deployed planning or behavior;
- WM-A, WM-I, or formal G2-G8 has passed;
- a checkpoint is promoted, selected for deployment, or eligible for G2-G8;
- counterfactual data will never be needed; or
- a local failure proves that the existing corpus contains no action signal.

## 3. Frozen development roles

The only data roles are the existing corrected-H6 development indices:

- train: 16,000 rows from 1,000 scenes;
- validation: 2,048 rows from 150 scenes disjoint from train;
- each row contains seven causal RGB endpoints and six requested action IDs;
- model-visible context is `rgb[0]`, `rgb[1]`, `rgb[2]`;
- the factual target is `rgb[3]`;
- model-visible requested actions are `actions[0]`, `actions[1]`, `actions[2]`;
- `actions[2]` is the candidate action whose aligned contribution is tested;
  and
- RGB positions 4, 5, and 6 are forbidden to every phase of this experiment.

The train and validation roles remain development-only and non-held-out. No
test, held-out, sealed, navigation, G2-G8, counterfactual-smoke, or other role
may be substituted or joined.

Validation rows are never used for gradients, schedule construction,
hyperparameter selection, early stopping, arm selection, checkpoint selection,
or retry decisions.

## 4. Requested/executed-command boundary

The model conditions on the requested action IDs present in the corrected H6
role. The role does not provide a complete bound executed/clipped command tape.

This is intentional for the candidate input: requested commands are available
before a planner observes their future execution. Future executed commands must
not be joined as candidate-time inputs.

The result must nevertheless state that requested and executed commands may
differ because of clipping, controller limits, delay, contacts, or other
closed-loop effects. The experiment learns factual closed-loop response to a
requested action. It does not establish executed-command fidelity.

## 5. One fresh consumed attempt

The runtime contract permits at most one fresh attempt:

- attempt ID: `world_model_existing_pool_three_arm_v1/attempt_v1`;
- exact attempt root:
  `.generated/dev/world_model_existing_pool_three_arm_v1/attempt_v1`;
- maximum attempts: 1;
- the exact attempt root must be absent before reservation;
- reservation itself consumes the attempt;
- retry: false;
- resume: false;
- overwrite: false;
- refill: false;
- replacement attempt: not authorized by this preregistration; and
- a partial, failed, timed-out, or infrastructure-terminated attempt remains
  consumed.

The external supervisor, not the worker, creates the attempt root and an
immutable reservation receipt. The reservation binds the plan, review,
authority, committed source, runtime, inputs, output root, worker command,
checker command, and a fresh nonce. The worker must reproduce that exact
reservation binding and nonce in its terminal result.

The worker, packer, and checker receive an exact allowlisted environment built
from the bound plan rather than inheriting the supervisor's ambient process
environment. It fixes the visible GPU, Python isolation/hash settings, locale,
thread count, executable path, and noninteractive Git configuration; arbitrary
ambient variables, Python paths, preload paths, and Git-control variables are
absent. The exact Python target, virtual-environment configuration, and
`/usr/bin/git` executable are file-bound. Supervisor Git identity checks also
use a source-defined minimal environment and ignore ambient Git controls.

The hard total supervisor wall ceiling is 43,200 seconds and the GPU-phase
ceiling is 36,000 seconds. These are upper caps, not expected durations. The
external supervisor conservatively hard-terminates the entire worker process
at the smaller remaining cap, so pack time also counts against the 36,000
second worker ceiling; the worker additionally measures its narrower GPU phase
at update/evaluation boundaries. The worker receives no automatic extension.
A later authority may lower either ceiling but may not raise it without
revising this preregistration.

Before reservation, the external supervisor requires at least 16 GiB free on
the development output volume. Source-shape sizing estimates approximately
2.53 GiB for the pack and 3.25 GiB for all 24 snapshots before serialization
overhead; the larger floor prevents a known late one-shot failure mode. Failing
the floor does not consume the attempt.

## 6. Custody and access rules

The following are mandatory:

- never open or search sealed, held-out, test, G2-G8, or other protected roles;
- never open `sealed_test.json`, a `sealed/` directory, or any `sealed_*`
  directory;
- open only regular non-symlink runtime inputs explicitly bound by the later
  authority, where the two exact H6 index bindings transitively bind the
  permitted RGB leaf paths and the fresh packer identity-binds each opened
  leaf byte stream in its immutable pack manifest (individual RGB leaves are
  not prehashed into the authority);
- never recursively discover RGB leaves;
- obtain visible RGB leaf names only from the two exact bound H6 indices;
- restrict opened RGB leaves to positions 0 through 3;
- reject every request for positions 4 through 6 before opening a path;
- do not modify the existing H6 indices, source RGB pool, or predecessor
  checkpoint;
- do not use network access;
- write only beneath the one reserved attempt root; and
- keep the receipt-only checker from opening RGB, packs, checkpoints, tensors,
  or other runtime payloads.

The source closure must have no unlisted local Python import. The later plan
and review must enumerate the worker, supervisor, checker, model, metrics,
dataset, packer, and transitive local source required by the exact command.

Before RGB packing, a metadata-only support audit must verify the complete
ordered train and validation roles and fail closed unless:

- train and validation scenes are disjoint;
- each visible action position `a0`, `a1`, and `a2` supports all nine IDs;
- the visible ordered pairs `a0-a1` and `a1-a2` each support all 81 cells;
- the audit opens no RGB or protected material.

The exact `a0-a1-a2` triple inventory and its missing cells are reported but
are not a pass gate. Full 729-cell factorial history coverage is not required
to estimate the registered factual candidate-action contrasts; the exact
metadata preflight observes 722 cells and seven missing triples. Support in
future-only positions `a3` through `a5` cannot satisfy either action or
adjacent-pair gate. Triple coverage, entropy, and mutual-information values are
diagnostic, not pass gates.

## 7. Fresh bound frame pack

The worker creates one fresh pack inside the reserved attempt. A legacy pack,
mutable alias, pack outside the attempt, or pack with a stale schema/source
binding is rejected.

For each train and validation row, pack exactly positions 0, 1, 2, and 3 as
uint8 tensors after the reviewed transformation:

1. decode the exact 224 by 224 RGB PNG;
2. crop to the registered 224 by 168 region;
3. bilinearly resize to 112 by 112;
4. store uint8 RGB; and
5. reproduce float32 ImageNet normalization only when a batch is loaded.

The pack contains:

- frames with shape `(rows, 4, 112, 112, 3)` and dtype uint8;
- requested actions with shape `(rows, 3)` and dtype int64;
- row order, scene, family, source-index, and source-RGB identity receipts; and
- a sampled bit-exact verification against the reviewed decoder.

Verification selects exactly 24 rows per role using the frozen experiment seed
and reopens their four allowed leaves. Maximum pack access is therefore:

- 18,048 bound H6 rows;
- 72,192 initial allowed RGB leaf opens;
- 192 verification reopens;
- 72,384 maximum allowed RGB leaf opens;
- zero opens at positions 4, 5, or 6; and
- 2,716,729,344 packed frame bytes.

After packing, training and evaluation consume only the bound pack. They do not
reopen source RGB leaves.

## 8. Frozen shared predecessor substrate

Load the exact selected single-frame masked-spatial predecessor named by the
later plan. Construct the reviewed recurrent patch-memory temporal V1 template
using its existing migration boundary.

The template provides one shared visual substrate:

- online `encoder.*`;
- `target_encoder.*`, hard-synchronized from the migrated online encoder by the
  reviewed temporal constructor; and
- `ema_update_count`, which remains exactly zero.

For the complete attempt:

- online encoder parameters are float32, frozen, and in evaluation mode;
- target encoder parameters are float32, frozen, and in evaluation mode;
- neither inventory appears in an optimizer;
- neither inventory receives a gradient tensor;
- no target EMA update occurs;
- start and terminal tensor-inventory hashes must be identical; and
- every measurement records the same encoder and target identity.

This protocol deliberately isolates temporal/action/predictor learning and
prevents encoder drift from confounding the three-arm comparison. A pass says
nothing about jointly training the encoder.

## 9. Trainable arm inventory

Create three independently allocated arm cores from one identical initialized
template. Each arm contains exactly:

- `predictor_position`;
- `predictor_mask_token`;
- `predictor_blocks.*`;
- `predictor_norm.*`;
- `predictor_output.*`;
- `action_embedding.*`;
- `time_embedding.*`; and
- `temporal_gru.*`.

Before update 1:

- all three trainable tensor inventories must be exactly equal by name, shape,
  dtype, and value;
- their canonical initial state hashes must match;
- trainable parameter objects must be pairwise disjoint across arms; and
- no arm may share optimizer state with another arm.

Snapshots store the arm state and optimizer state plus the frozen-substrate and
predecessor bindings. They need not duplicate frozen substrate tensors.

## 10. Registered arms

The exact arm names are `conditioned`, `blind`, and `shuffled`.

### 10.1 `conditioned`

Training and evaluation use factual `actions[0:3]`. Action-embedding gains are
`[1, 1, 1]`.

### 10.2 `blind`

Training and evaluation preserve factual `actions[0]` and `actions[1]`, but
multiply the candidate-step action embedding by zero. The exact gains are
`[1, 1, 0]`.

The step-2 time embedding remains present. The candidate is not replaced with
HOLD, action ID 0, an average embedding, or a learned null token. This is a
candidate-action-blind control, not a history-action-blind model.

### 10.3 `shuffled`

Training preserves factual `actions[0]` and `actions[1]` and replaces only
`actions[2]` with the preregistered donor action. Action-embedding gains remain
`[1, 1, 1]`.

Evaluation switches this arm back to factual `actions[2]`. It is never
evaluated on a deranged target or deranged validation action.

## 11. Candidate-action derangement

Build one deterministic donor mapping over the 16,000 ordered train rows. For
each family independently, the mapping must satisfy all of the following:

- every source row receives exactly one donor row;
- every donor row is used exactly once;
- donor row index differs from source row index;
- donor scene differs from source scene for every row;
- donor `actions[2]` differs from factual source `actions[2]` for every row;
- the donor and factual `actions[2]` histograms are exactly equal; and
- no row crosses a family boundary.

The complete mapping and candidate-action vector are immutable and canonically
bound before training. Any unsatisfied constraint terminates the attempt as
`INCONCLUSIVE_CONTRACT_FAILURE` before the first optimizer step.

Validation is never used to construct a donor map.

## 12. Shared schedule and optimizer

All arms use the same deterministic row schedule, masks, microbatch boundaries,
target encodings, update count, and learning-rate fraction.

The frozen schedule is:

- seed: 20260731, included literally in the SHA-256 row-order namespace;
- updates: 700;
- sequence batch: 256;
- microbatch: 32;
- train-row pool: all 16,000 rows;
- each epoch orders all row IDs by
  `SHA256(namespace/epoch/row)`, where the namespace contains that seed;
- permutation tails carried into the next batch;
- no discarded permutation tail;
- warmup: 150 updates;
- cosine schedule horizon: 3,000 updates;
- observation updates: 0, 100, 200, 300, 400, 500, 600, and 700; and
- no early stopping, checkpoint selection, or metric-driven schedule change.

Each arm has an independent two-group AdamW optimizer:

| Group | Exact members | Base LR |
|---|---|---:|
| predictor | position, mask token, predictor blocks, norm, output | `1e-4` |
| temporal/action | action embedding, time embedding, temporal GRU | `3e-4` |

Both base rates are multiplied by 4.0 before applying the shared warmup/cosine
fraction. AdamW uses betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`,
and `amsgrad=false`. Global gradient norm is clipped to 1.0 separately over
each arm's trainable inventory.

The encoder and target are absent from every optimizer. There are exactly 700
optimizer steps per arm and zero EMA steps.

## 13. Shared encoding and exact objective

For each microbatch, normalize packed frames to float32 and encode once:

- context input: `(B,3,3,112,112)`;
- shared online history: `(B,3,256,192)`;
- future input: `(B,3,112,112)`;
- shared full target: `(B,256,192)`;
- registered per-row target indices: `(B,64)`; and
- gathered detached target: `(B,64,192)`.

The shared history and target require no gradient and are passed unchanged to
all three arms.

For each arm, the reviewed head computes:

- requested action and time conditioning: `(B,3,192)`;
- recurrent patch streams: `(B*256,3,192)`;
- recurrent memory: `(B,256,192)`;
- predictor input: `(B,320,192)`; and
- predicted target tokens: `(B,64,192)`.

The sole training objective is the mean normalized half-squared future-latent
energy:

`0.5 * (normalize(prediction) - normalize(detached_target))^2`, summed over
feature dimension and averaged over target tokens and rows.

No classification, contrastive, rank, action-ID, persistence, reconstruction,
retention, or auxiliary loss is permitted.

## 14. Training accounting

The exact training accounting is:

- 179,200 schedule-row presentations;
- 179,200 sequence presentations per arm;
- 537,600 total arm-head sequence presentations;
- 537,600 shared online context-frame encodings;
- 179,200 shared future target-frame encodings;
- 716,800 actual training frame encodings;
- 700 optimizer steps per arm;
- 2,100 optimizer steps across the three arms; and
- zero target EMA steps.

Shared encoding is a fairness and throughput requirement. Encoding the same RGB
separately per arm is a contract failure, not an equivalent implementation.

## 15. Factual validation protocol

At update 700 only, first evaluate a complete no-gradient train-fit panel over
all 16,000 ordered train rows. Encode each row once and reuse the frozen
history/target tensors across arms. Report paired per-row factual log-energy
advantages for conditioned versus blind and conditioned versus shuffled, then
average within each of the eight families and equally across families. This
panel supplies the registered training-fit point gates. It is not used for an
optimizer step, checkpoint choice, or schedule change. No train nine-way gate
is required.

At every observation update, evaluate all 2,048 ordered validation rows. Encode
the three context frames and factual future frame once per row and reuse those
encodings across arms and candidate queries.

For each arm:

1. compute factual-target energy using the arm's registered evaluation
   treatment;
2. report row, action, family, and scene summaries;
3. evaluate nine candidate IDs by preserving factual history actions and
   replacing only candidate `actions[2]` in the query;
4. compare every candidate prediction with the same factual target; and
5. retain complete finite vectors required for paired and clustered metrics in
   the runtime snapshot while publishing bounded JSON summaries.

For `blind`, the candidate gain remains zero for all nine queries, so exact
ties are expected and must be reported. The frozen nine-way decision uses the
lowest action ID on an exact tie, and separately reports tie count, tie rate,
unique-winner count, and unique-winner accuracy so that deterministic tie
breaking cannot be mistaken for learned action information.

Validation accounting across eight observations is:

- 16,384 validation-row panels per arm;
- 65,536 shared validation frame encodings;
- 442,368 nine-way arm/candidate row queries; and
- zero validation backward calls or optimizer steps.

The update-700 full-train fit panel adds:

- 16,000 train-fit rows;
- 64,000 shared frame encodings;
- 48,000 arm factual row queries; and
- zero backward calls or optimizer steps.

Across gradient training, the full-train fit panel, and all validation panels,
the attempt performs exactly 846,336 shared frame encodings, excluding pack
decoding/verification.

## 16. Registered metrics

All scalar and vector outputs must be finite. The primary energy is normalized
half-squared future-latent energy; lower is better.

The registered panel includes:

- conditioned, blind, and shuffled factual mean energy;
- paired per-row log-energy advantage for conditioned versus blind;
- paired per-row log-energy advantage for conditioned versus shuffled;
- family-equal and action-equal macro summaries;
- scene-clustered uncertainty for primary paired comparisons;
- nine-way candidate identification balanced accuracy;
- nine-way exact-tie and unique-winner diagnostics;
- hardest-wrong-action energy margin;
- persistence energy advantage;
- wrong-history energy advantage;
- prediction and target effective rank; and
- prediction-to-target effective-rank ratio.

Log-energy advantage is oriented so that values above zero favor the named
factual/conditioned prediction. Cross-arm, persistence, and wrong-history
confidence lower bounds are one-sided 95% bounds from their frozen
deterministic scene-clustered bootstraps. The nine-way bounds instead use the
separately specified Bayesian lower quantile below. Balanced chance for
nine-way action identification is exactly `1/9`.

For nine-way balanced accuracy and hardest-action margin, use a deterministic
positive-weight scene-cluster Bayesian bootstrap. At seed `20260803`, Python's
bound `random.Random` generator draws 52 random bits, maps them to the strictly
open unit interval as `(bits + 1) / (2**52 + 1)`, and transforms that value to
a strictly positive finite `Exp(1)` weight with `-log1p(-u)` for each scene
inside each family. The same scene-weight map is reused across all nine actions
and both statistics. Within each family/action cell, normalize
weights only across scenes that support that action and compute its weighted
scene-cell mean; then average families equally, average actions equally for
balanced accuracy, and take the minimum of the nine action-macro margins for
the hardest-action statistic. Run exactly 10,000 replicates and use sorted
index 500 for both lower quantiles.

Report the exact weighting/RNG algorithm identity, all 72 family/action
supporting-scene counts, and their minimum. Every registered family/action cell
must be supported by at least two validation scenes; lower support is a
contract failure. Every original scene receives a strictly positive weight, so
sparse cells remain defined without conditioning away bootstrap draws that
omit them. These are deterministic
Bayesian-bootstrap lower quantiles, not a guarantee of frequentist coverage.

No result may select a checkpoint or metric observation after seeing validation
values. The registered terminal decision uses the complete update-700 train-fit
panel, the update-700 clustered validation comparisons, and the validation
tail-trend requirements at updates 500, 600, and 700.

## 17. Gate precedence and localization

Apply these decisions in order; the first matching condition is terminal.

### 17.1 Contract integrity

Emit `INCONCLUSIVE_CONTRACT_FAILURE` if any attempt, custody, input, pack,
schedule, derangement, shared-encoding, parameter-partition, optimizer,
frozen-substrate, output-immutability, accounting, finiteness, or
validation-no-gradient requirement fails.

This status is not a scientific negative result.

### 17.2 Training fit

At update 700, require both complete-train, family-equal paired log-energy point
advantages to be above zero:

- conditioned versus blind; and
- conditioned versus shuffled.

Emit `LOCALIZE_TRAIN_FIT_FAILURE` if either point gate is not above zero. A
failure here precedes and blocks generalization interpretation. No train
nine-way, train bootstrap-confidence, or train tail-trend gate is required.

### 17.3 Scene-disjoint cross-arm generalization

After training fit passes, emit `LOCALIZE_GENERALIZATION_OR_CONFOUNDING` if
either of the following update-700 one-sided 95% lower bounds is not above
zero:

- conditioned versus blind paired log-energy advantage; or
- conditioned versus shuffled paired log-energy advantage.

In addition, the point advantage against both controls must be above zero at
each of updates 500, 600, and 700. A single late crossing is insufficient.

### 17.4 Candidate-action alignment

After both cross-arm comparisons pass, emit
`LOCALIZE_ACTION_ALIGNMENT_FAILURE` if either:

- conditioned nine-way balanced-accuracy lower bound is not above `1/9`; or
- conditioned hardest-wrong-action margin lower bound is not above zero.

### 17.5 Predictor usefulness and non-collapse

After action alignment passes, emit `LOCALIZE_PREDICTOR_NOT_USEFUL` if either
the conditioned persistence-advantage or wrong-history-advantage lower bound is
not above zero, or if the prediction-to-target effective-rank ratio is below
0.25 at two or more of updates 500, 600, and 700.

### 17.6 Pass

Emit `PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY` only if none of the prior
conditions applies.

The pass requires aligned candidate actions to beat both controls, action
identification and hardest-wrong alignment to exceed their references,
prediction to beat persistence and wrong history, and noncollapsed rank at no
fewer than two of the three tail observations.

## 18. Interpretation of localized outcomes

| Status | Maximum interpretation | Forbidden interpretation |
|---|---|---|
| `INCONCLUSIVE_CONTRACT_FAILURE` | Runtime or evidence contract did not complete | Model/data negative result |
| `LOCALIZE_TRAIN_FIT_FAILURE` | This frozen head/objective/schedule did not establish stable training fit | Existing pool has no action signal |
| `LOCALIZE_GENERALIZATION_OR_CONFOUNDING` | Training fit did not yield robust scene-disjoint advantage over both controls | Architecture or data uniquely at fault |
| `LOCALIZE_ACTION_ALIGNMENT_FAILURE` | Aggregate prediction advantage did not produce registered candidate discrimination | Untaken actions are invalid in general |
| `LOCALIZE_PREDICTOR_NOT_USEFUL` | Action comparison passed, but prediction remained weak versus simple/history controls or collapsed | Factual action signal is absent |
| `PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY` | Existing selected pool teaches factual requested-action prediction under this frozen protocol | Untaken-action causality or promotion |

The overlap and derangement audits accompany every scientific interpretation.
If observational support is narrow, a generalization failure may motivate a
targeted paired-data tranche, but it does not automatically authorize one.

## 19. Immutable output contract

The reserved attempt contains only fresh immutable outputs. Required receipts
include:

- `reservation.json`;
- fresh pack manifest and role artifacts;
- `overlap_audit.json`;
- `shuffle_audit.json`;
- eight JSON measurements and eight snapshots for each of `conditioned`,
  `blind`, and `shuffled` at updates 0 through 700 by 100;
- `result.json` with status `COMPLETE_PENDING_TERMINAL_REVIEW`;
- receipt-only `receipt_check.json`; and
- external `terminal_supervision.json`.

There are exactly 24 measurement receipts and 24 snapshot bindings. Every
artifact is exclusive-create and may not overwrite a pre-existing path. The
terminal result directly binds every pre-result worker artifact by path, byte
count, and digest. Its exact worker inventory therefore has 57 paths: the pack
manifest and six role artifacts, two audit receipts, 24 measurements, and 24
snapshots. The supervisor-created reservation is separately bound under
`result.attempt`; the result, receipt check, and terminal supervision record
are then bound in order by the external supervisor, since no receipt can bind
itself or a file that does not yet exist.

The result's top-level shape is frozen to the checker contract. Rich treatment,
schedule, frozen-substrate, citable-status, and scientific-decision details are
nested under `joint_decision`; reservation binding, nonce, and status are
nested under `attempt`; device, wall time, and output inventory are nested under
`runtime`; detailed pack-role accounting remains under `accounting`; and
`input_bindings` contains bindings only.

The receipt checker may open only `result.json`, the two JSON audit receipts,
and the 24 JSON measurement receipts. Pack payloads, RGB, checkpoints, tensor
snapshots, and runtime data remain identity-only to the checker.

## 20. Terminal review and next decision

No worker or checker output is self-promoting. The result remains development
only and non-citable until an independent terminal review checks exact bindings,
accounting, custody, gate implementation, and interpretation.

After a valid terminal result:

- on `PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY`, the next data question is
  a separately preregistered small scene-disjoint counterfactual evaluation
  role; do not generate another large training corpus by default;
- on a localized scientific failure, follow the registered localization before
  proposing objective, representation, optimization, or data changes; and
- on contract failure, do not retry or resume this attempt. Any replacement
  requires an explicit new preregistration, independent review, and authority.

Nothing in this document changes the meanings or custody of WM-A, WM-I, or
formal G2-G8.

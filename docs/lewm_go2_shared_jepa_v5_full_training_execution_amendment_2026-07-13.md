# Shared JEPA V5 full-training execution amendment

Date: 2026-07-13

Status: **frozen before trainer implementation, learned output, or role-payload access**

## Purpose and authority

This preregistration closes the execution-design gap between a qualified
two-seed V4 observable-camera-ray fit and the reviewed staged Shared JEPA V5
G2/G3 lifecycle. It fixes the only allowed V4 migration, initialization,
training, matched ablation, checkpoint selection, probability calibration,
development gates, immutable publication, failure handling, and one-shot G2
boundary before any full-training implementation or result exists.

This document is source- and metadata-derived. It authorizes no implementation,
dataset construction or use, model/checkpoint open, RGB or label open, GPU
execution, training, selection, calibration, G2/G3 contact, held-out access,
runtime use, hardware use, navigation evaluation, production use, or promotion.
Each later boundary requires its own reviewed source and exact artifact binding.

## Frozen source and planning lineage

The final reviewed model/output/loss boundary is:

- `lewm/models/shared_observable_camera_ray_jepa_v5.py`, SHA-256
  `b438295d7ec5cb0897cc953a229f461da7fca16322c4c936555d37833a36e4b9`;
- `lewm/tests/test_shared_observable_camera_ray_jepa_v5.py`, SHA-256
  `848aa8be369b89c973a4da916f9c7abeff47eca12aceb4304cf612ed4d53227b`;
- output/loss correction record, SHA-256
  `83dcd8f8702656c25f4584295827d0c82cf1db113abe2de4a417e7b528abff1f`.

The final reviewed staged lifecycle boundary is:

| Artifact | SHA-256 |
|---|---|
| `scripts/go2_shared_jepa_v5_one_shot.py` | `62a19f3028e9152120af990528752431b996f56b4bc9b62db32eba47ae235a1f` |
| `scripts/go2_shared_jepa_v5_launcher.py` | `7f273649fa6c8b4256c552359927fc20bb59d1bfbd5b47194a3f5a941c5b8958` |
| `scripts/run_go2_shared_jepa_v5_gate.py` | `37402f0f75a7a4f475539e269e77aeae072ce80b0af0bcb4147e2ec1b33ff57a` |
| `scripts/finalize_go2_shared_jepa_v5_gate.py` | `f0426201f5344d0eb1d43e183e4755ac8fd7aecdc9af6e5b7c19076af3f5dc34` |
| `scripts/publish_go2_shared_jepa_v5_checkpoint.py` | `4e045365dadb28bd37cdbb49808bef7528d4e5cb0c3e77ff5aae678559174fab` |

The staged lifecycle independent review is SHA-256
`bcb587c5bd7ea08063cbbf1c8d5a4a99b29c24fdfc490469aae4bff6dbe98abc`.
All six production-stage authority identities remain unset. The lifecycle may
consume a separately qualified shared checkpoint; it may not manufacture one.

The governing design records are bound as follows:

| Record | SHA-256 |
|---|---|
| V5 joint-training gap audit | `b4bc71e6cc2728fdbc5c1a3822d4be130b9c2ccac3bb8cf2a9baece6bc497f6a` |
| Shared JEPA G2/G3 plan | `54ad8c08546c46c8989a84e497b54b83366526f8f5ed6faed6364880fa1a702a` |
| Navigation-work readiness goal | `1095252d67f2b450861e97a6083c2866ee3158382f339049e1766b3369dd8a12` |
| Development raw-supervision preregistration | `07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb` |
| Builder source-inventory amendment | `39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3` |
| V4 fit-ladder gate | `49887b8b39ba16e490f6171ac0efe239456e1d27081312a71800ca33c247f874` |

Any change to the reviewed V5 model or staged lifecycle bytes requires a new
dated amendment. A later dataset binding may fill currently unknown builder
manifest/audit file hashes, but may not change a scientific value in this
document.

## Exact role universe

The parent paired-navigation identities are fixed before role payload access:

- manifest path:
  `.generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/dataset_manifest.json`;
- manifest file SHA-256:
  `ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180`;
- row-index file SHA-256:
  `187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac`;
- role-assignment SHA-256:
  `016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02`;
- untouched-G2 scene-set commitment:
  `0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`;
- source-index file SHA-256:
  `11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c`.

There are exactly eight rendered navigation/stress families, in this literal
order:

1. `large_enclosed_maze`;
2. `local_composite_motifs`;
3. `loop_alias_stress`;
4. `medium_enclosed_maze`;
5. `open_obstacle_field`;
6. `rough_local_dynamics`;
7. `small_enclosed_maze`;
8. `visual_sensor_stress`.

Each family contributes nine train scenes and exactly one scene to each of
checkpoint selection, probability calibration, and untouched G2. No role may
be reassigned, balanced, backfilled, subsampled, or filtered by a learned
result.

| Role | Scenes | Paired transitions | Endpoint instances | Unique endpoints |
|---|---:|---:|---:|---:|
| train | 72 | 4,262 | 8,524 | 7,777 |
| checkpoint_selection | 8 | 495 | 990 | 924 |
| probability_calibration | 8 | 415 | 830 | 759 |
| development total | 88 | 5,172 | 10,344 | 9,460 |
| g2_evaluation | 8 | 469 | forbidden before one-shot G2 | forbidden before one-shot G2 |

The allowed development sidecar file SHA-256 values are:

- train: `6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6`;
- checkpoint selection:
  `4ed434d04afc94b7b82050f5e9fafc900cc03c33a2d847f9784410f8f76f65de`;
- probability calibration:
  `3e5c10e6c15969eb30fbf38bbdb7b47d5fafe25bf14c5547f07ac609b79d91ae`.

The metadata plan must reproduce content SHA-256
`8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3`,
ordered-pair SHA-256
`76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea`,
and ordered-endpoint SHA-256
`8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698`.

The full-training source may be implemented only after a successor metadata
plan, raw-supervision builder, and independent auditor pass review and publish
immutable development-only manifest/audit identities. Those future identities
must be bound before exact execution. The raw artifact must contain exactly the
development rows above and no G2 supervision.

## V4 two-seed migration rule

Shared V5 construction is forbidden until the final reviewed V4 full-panel
ladder reports a canonical pass for every rung `N=5,16,32,320` for seed
`20260710`, then every rung for seed `20260711`, and its immutable two-seed
finalizer reports a pass without a threshold change. Both complete source,
checkpoint, verifier, stage, seed, and two-seed chains must be reopened and
rehash exactly.

The migration source is fixed now: it is the seed-`20260710`, `N=320` final fit
checkpoint. Seed `20260711` is a replication requirement only. It may not be
selected because it has a lower loss, substituted after a primary-source
failure, averaged, ensembled, or used to initialize a second arm. If the
primary checkpoint is unavailable, structurally invalid, or incompatible with
the reviewed V5 migration method, the V5 attempt fails closed even when the
second seed passed.

The trainer must load the primary checkpoint into the exact reviewed
`ObservableCameraRayEvidenceV4Model`, instantiate
`SharedObservableCameraRayJepaV5` with its frozen production config, and invoke
only `migrate_from_fit_model`. It must prove:

- the V4 encoder state copied exactly into the one online V5 encoder;
- every compatible evidence-head entry copied exactly and no extra key did;
- source and pixel-ray shapes are unchanged;
- the returned `V4HeadMigrationReceiptV5` hashes the source fit state, migrated
  encoder, migrated evidence head, and migrated key count; and
- the EMA target encoder and target BEV decoder were hard-synchronized exactly
  once from the post-migration online state before training.

## One deterministic initialization and two matched arms

The only V5 initialization seed is `20260712`. Before GPU or role-payload
access, the reviewed implementation constructs one FP32 production-config V5
model on CPU under that seed, performs the exact primary V4 migration, and
serializes the complete post-migration training state in sorted-key order.

Two independent model instances, `promoted_jepa` and `matched_no_jepa`, must be
loaded from that same serialized state. Their complete tensor-state SHA-256
values must be identical before optimizer construction. Their CPU and GPU RNG
states, optimizer hyperparameters, initial zero optimizer state, data schedule,
microbatch boundaries, camera inputs, actions, labels, counterfactuals, update
budget, checkpoint steps, and evaluation order are identical. The arms run
sequentially on the same R9700; they never run concurrently.

The promoted arm uses the reviewed complete joint loss. The ablation uses only
the complete current/next V4 package as its backward loss. It still executes
the same model forward and diagnostic controls, and its EMA target is updated
once after every optimizer step, but no JEPA prediction, equivariance, action
contrast, variance, anti-collapse, or persistence term contributes gradient.
The ablation cannot select, replace, initialize, average with, or promote a
checkpoint.

## Exact train-only schedule

Only the 4,262 train-role pairs may influence parameters. The schedule is
constructed on CPU before the first train payload open:

1. take the exact ordered train-pair identity list from the independently
   accepted raw-supervision manifest;
2. create one CPU `torch.Generator` with seed `20260713`;
3. concatenate complete `torch.randperm(4262, generator=g)` cycles;
4. take the first `128,000` indices;
5. partition them, without dropping or padding, into 8,000 consecutive
   effective batches of 16 pairs; and
6. partition each effective batch into four consecutive microbatches of four
   pairs.

The resulting 128,000 pair presentations are about 30.03 complete train-role
passes. The implementation must freeze the ordered train identities, all
128,000 schedule indices, per-update pair identities, and canonical schedule
SHA-256 before learned output. Both arms consume those exact bytes.

Primitive vocabulary is the literal sorted train-role primitive set. Each
row's action is its one-hot primitive. The commanded SE(2) delta table is the
componentwise train-role median for each primitive, computed once before model
construction and hashed. The wrong-action control is the one-position cyclic
roll of the one-hot action dimension, and its wrong commanded delta is matrix
multiplication by that same frozen table. The wrong-commanded-delta control
uses the reviewed V5 sign-reversal path. There is no augmentation, stochastic
crop, role balancing, class balancing, replay buffer, curriculum, early stop,
or result-dependent data-order change.

## Optimizer and device contract

Both arms use:

- AdamW over every trainable online parameter;
- betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, no AMSGrad;
- 8,000 optimizer updates;
- microbatch four pairs, accumulation four, effective batch 16 pairs;
- each microbatch loss divided by exactly four before backward;
- global gradient-norm clip `1.0`, once after the fourth backward and before
  each optimizer step;
- one optimizer step and then exactly one
  `update_ema_target_after_optimizer_step` call per update;
- FP32 parameters, activations, loss, optimizer state, and EMA; no autocast,
  AMP, quantization, compilation, gradient checkpointing, parameter averaging,
  or distributed execution; and
- checkpoints after updates `1000,2000,3000,4000,5000,6000,7000,8000`.

The one-indexed learning rate is fixed. For updates `u=1..400`, it is linear
from `1e-6` to `1e-4` inclusive:

```text
lr(u) = 1e-6 + (1e-4 - 1e-6) * (u - 1) / 399
```

For `u=401..8000`, it is cosine from the update-400 value to `1e-5`:

```text
lr(u) = 1e-5 + 0.5 * (1e-4 - 1e-5)
        * (1 + cos(pi * (u - 400) / 7600))
```

The trainer uses at most four spawned CPU loader/decode workers, each with
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`; persistent workers and prefetch factor two are fixed.
CPU metadata, audit, decode, hashing, and publication work may overlap only
where the access ledger remains ordered and complete.

Neural work is allowed only on ROCm GPU0 whose exact discovered name is
`AMD Radeon AI PRO R9700` and whose reported memory is at least 32 GiB.
`HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, and device `cuda:0` are
required; `HSA_OVERRIDE_GFX_VERSION` must be absent. Any device named Raphael,
any GPU1/iGPU visibility or use, fallback to CPU/iGPU, multi-GPU execution, or
wrong device identity is a structural failure before role-payload access.

Before the exact reservation, the reviewed implementation must pass a
synthetic-shape GPU0 smoke using the exact model, microbatch, four accumulated
backwards, optimizer step, gradient clip, and EMA update but no repository
role payload. An OOM or unsupported deterministic operation requires a new
pre-output amendment; it does not permit an in-run batch change.

## Mandatory joint objective

Every promoted microbatch must call the reviewed `forward_training_pair` with
current/next RGB, action, realized and commanded deltas, both camera
calibrations, both ground heights, the prediction mask, and all required
counterfactuals. It must then call only `combine_joint_losses` with complete raw
V4 supervision for both endpoints.

The frozen arithmetic is:

```text
frame_v4 = 0.25 * ordered_first_hit_nll
         + 0.25 * target_bin_offset_smooth_l1
         + 0.25 * ground_clear_distance_state_balanced_bce
         + 0.25 * derived_raster_hierarchical_bce

pair_v4  = 0.5 * current_frame_v4 + 0.5 * next_frame_v4
joint     = established_jepa_total + 1.0 * pair_v4
```

The established JEPA package retains the production config exactly: prediction
weight `1.0`, equivariance `0.25`, action contrast `1.0`, variance `0.1`, target
standard-deviation target `0.5`, EMA momentum `0.996`, and the reviewed
persistence/counterfactual controls. Hierarchical-raster-only supervision,
single-endpoint V4 loss, omitted counterfactuals, changed ground visibility,
changed component weight, dynamic loss normalization, or a caller-created
partial loss package is structurally ineligible.

## Checkpoint-selection-only eligibility and ranking

Update zero is evaluated and recorded as a nonselectable migration baseline.
The eight saved promoted checkpoints are evaluated in increasing update order
on all 495 checkpoint-selection pairs. Physical metrics use each of the 924
unique endpoints exactly once; JEPA metrics use all 495 pairs. No calibration,
G2, held-out, train-role metric, or ablation result may rank checkpoints.

At each candidate, the correct-RGB arm and a deterministic within-family cyclic
wrong-RGB arm are evaluated with calibration fixed. Geometry and labels remain
attached to the original endpoint. A candidate is eligible only when its
aggregate and each of the eight family reports pass all nonempty checks:

| Physical check | Required value |
|---|---:|
| pixel first-hit balanced accuracy | `>= 0.95` |
| depth median / p95 error | `<= 0.10 m / 0.25 m` |
| ground clear balanced accuracy | `>= 0.95` |
| every nonempty distance-group balanced accuracy | `>= 0.92` |
| derived-raster NLL | `<= 0.15` |
| derived-raster balanced accuracy | `>= 0.95` |
| each present UNKNOWN/FREE/OCCUPIED recall | `>= 0.95` |

The correct-minus-wrong controls must also pass for aggregate and every family:
pixel balanced-accuracy drop `>=0.12`, depth median/p95 increase
`>=0.12/0.20 m`, ground balanced-accuracy drop `>=0.12`, raster-NLL increase
`>=0.12`, and raster balanced-accuracy drop `>=0.12`.

Mandatory JEPA health is evaluated from raw accumulators, not reported means.
Aggregate and every family must have:

- finite loss/control values and a positive prediction-valid-cell count;
- target cross-sample standard-deviation mean `>=0.05`;
- target cross-sample effective rank `>=4.0`;
- warped-persistence target change `>1e-4`;
- prediction-to-warped-persistence ratio `<1.0`;
- wrong-action advantage over target change `>=0.10`;
- wrong-commanded-delta advantage over target change `>0.0`; and
- both wrong-action and wrong-commanded-delta prediction sensitivity `>0.0`.

No best failing checkpoint exists. If one or more candidates are eligible,
rank them by the following lexicographic tuple, maximized without rounding:

```text
(
  minimum normalized physical-gate margin over aggregate and families,
  minimum normalized JEPA-health margin over aggregate and families,
  mean normalized physical-gate margin over aggregate and families,
  mean normalized JEPA-health margin over aggregate and families,
  -aggregate complete V4 loss,
  -aggregate prediction-to-persistence ratio,
  -update_number,
)
```

For a lower bound `x>=t`, normalized margin is `(x-t)/max(abs(t),1e-12)`.
For an upper bound `x<=t`, it is `(t-x)/max(abs(t),1e-12)`. Strict checks use
their literal difference from the boundary. Nonfinite or empty required values
make the checkpoint ineligible. The last key selects the earlier update on an
exact tie. The selected update and complete training/deployment state hashes
are frozen before the ablation opens checkpoint-selection payload.

## Matched no-JEPA evaluation

The ablation completes the same 8,000-update schedule and writes checkpoints at
the same eight steps. It does not run a selection sweep. After the promoted
update is immutable, the ablation is evaluated once at that exact update on
the same ordered checkpoint-selection pairs and controls. Its checkpoint cannot
be substituted even if every physical metric is better.

The report carries promoted-minus-ablation raw per-scene, aggregate, and
per-family physical metrics, plus both arms' JEPA-health diagnostics. A causal
claim that JEPA improved development generalization is allowed only when the
promoted arm has a strictly better aggregate normalized physical margin, has a
positive physical-margin delta in at least five of eight families, and does not
reduce planner-admitted FREE precision after the fixed calibration below.
Failure of this reporting rule is a negative ablation result, not permission to
promote the no-JEPA arm or revisit checkpoint selection.

## Fixed calibration and threshold algorithm

After the selected update is frozen, each arm independently opens the 415
probability-calibration pairs. Vector calibration uses the derived three-class
physical-raster logits for all 759 unique endpoints exactly once, all valid
cells, native class priors, and no subsampling, balancing, backfill, replacement,
or train/selection reuse. All three classes must be present.

For each arm, start three log scales and three biases at zero. Apply positive
diagonal vector scaling as:

```text
scale_c = exp(clamp(log_scale_c, -3, 3))
centered_bias_c = bias_c - mean(bias)
calibrated_logit_c = scale_c * source_logit_c + centered_bias_c
```

Fit unweighted multiclass NLL on CPU float32 with full-batch
`torch.optim.LBFGS`, learning rate `0.5`, maximum iterations `80`, and
`line_search_fn="strong_wolfe"`. Then clamp log scales to `[-3,3]`, center the
biases, and recompute NLL. Missing classes, nonfinite parameters, or calibrated
NLL above uncalibrated NLL by more than `1e-6` is terminal failure.

Use the exact global threshold grid:

- FREE probability minimum: `(0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99)`;
- OCCUPIED probability maximum: `(0.01,0.02,0.05,0.10,0.20,0.35)`;
- UNKNOWN probability maximum: `(0.01,0.02,0.05,0.10,0.20,0.35)`;
- OCCUPIED detection minimum:
  `(0.01,0.02,0.05,0.10,0.20,0.35,0.50)`.

Reject overlapping admission/detection tuples. A tuple passes only with at
least one admitted FREE cell, admitted-FREE precision `>=0.99`, obstacle
exclusion recall within 2 m `>=0.95`, and obstacle detection recall within 2 m
`>=0.95`. Select from passing tuples by maximizing, in order, useful FREE
recall, admitted-FREE precision, obstacle detection recall, OCCUPIED detection
minimum, and negative FREE threshold. If no tuple passes, calibration fails;
the best merely compatible tuple is not promotable.

The promoted global calibrator and one global threshold tuple are fixed before
any G2 contact. Per-family calibration or thresholds are forbidden. The
ablation uses the identical algorithm but its own fitted six-vector and global
tuple solely for matched reporting.

## Aggregate and per-family development gate

Before a shared checkpoint is qualified for one-shot G2, the selected promoted
state must satisfy all selection eligibility above and, on the calibrated role,
both aggregate and each family must satisfy:

- admitted observable-physical FREE precision `>=0.99`;
- directly observable physical-obstacle detection recall within 2 m `>=0.95`;
- observable physical-obstacle exclusion recall within 2 m `>=0.95`;
- useful observable-physical FREE recall `>=0.90`;
- calibrated raster NLL no greater than uncalibrated raster NLL plus `1e-6`;
- every JEPA-health check from checkpoint selection; and
- no structural, role, provenance, access-ledger, source, state, or finiteness
  failure.

There is one global all-check pass. An average cannot hide a failed family, an
empty admitted-FREE set, an absent class, or a failed JEPA control. Calibration
data cannot choose another checkpoint or update any model weight.

## Immutable namespace, training record, and access ledger

The sole output namespace is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v1`

One reviewed isolated operation must reserve the canonical attempt before GPU,
model, RGB, label, or role-payload access. It retains the claimed-directory
descriptor and identity through failure or final publication. All writes are
exclusive, descriptor-relative, no-follow, fsynced, and immutable. A successful
tree has this fixed logical inventory:

```text
reservation.json
source_review.json
input_bindings.json
schedule.json
initialization.json
arms/promoted_jepa/training_trace.jsonl
arms/promoted_jepa/checkpoints/update_{1000..8000}.pt
arms/promoted_jepa/checkpoint_metrics.json
arms/matched_no_jepa/training_trace.jsonl
arms/matched_no_jepa/checkpoints/update_{1000..8000}.pt
arms/matched_no_jepa/matched_update_metrics.json
selection.json
calibration/promoted_jepa.json
calibration/matched_no_jepa.json
ablation_comparison.json
qualified_checkpoint.pt
access_ledger.json
training_record.json
completed.json
```

`completed.json` is last and binds every relative path, byte count, file
SHA-256, semantic/content SHA-256, state SHA-256, and directory inventory.
`qualified_checkpoint.pt` contains only the selected promoted deployment state
plus the exact provenance/calibration/threshold contract required by checkpoint
schema V5; it is development-only and has `runtime_ready=false`.

The append-only, content-chained access ledger records sequence number, stage,
arm, role, operation, canonical relative path, expected and observed file hash,
byte count, process/worker identity, and preceding-event hash. The final record
must prove:

- parameter-gradient inputs came only from train;
- promoted selection opened only checkpoint selection at update zero and the
  eight fixed candidate updates;
- ablation selection opened checkpoint selection only at the already selected
  promoted update;
- calibration opened only probability calibration after selection was frozen;
- G2 sidecar/payload/RGB, G3, held-out, sealed, runtime, navigation result,
  physical executor/reset, hardware input, and production-promotion opens were
  exactly zero; and
- every input/source opened during the run rehashed identically before
  completion.

An independent verifier must reopen raw checkpoints and role inputs, recompute
migration, schedule, selected metrics, calibration, thresholds, comparison,
deployment-state filtering, output inventory, and ledger chain. Trainer
self-reporting cannot qualify the checkpoint.

## Failure and escalation rules

Any wrong source/hash, role/count, seed, schedule, arm state, device, update,
checkpoint, loss, EMA cadence, calibration, output, ledger, or access boundary
is a structural invalidation. Nonfinite loss/gradient/state, OOM, process death,
or durability failure is an execution failure. Either consumes the reserved
attempt and writes one sanitized terminal failure receipt when possible.

A structurally valid run with no eligible checkpoint, a failed calibrated
development gate, or a failed family is a numeric failure. It publishes the
immutable diagnostic record but no qualified checkpoint and no G2 license.

No failure permits retry, resume from an unbound checkpoint, seed substitution,
extra updates, smaller batch, different accumulation, new checkpoint cadence,
different ranking, threshold relaxation, per-family threshold, calibration
refit, role backfill, loss reweight, ablation promotion, or G2 contact. Any
future attempt requires a new dated first-principles diagnosis, one explicitly
named intervention, new namespace/revision, implementation review, and
different-agent preregistration review before output. G2 and held-out remain
closed during such development.

## One-shot G2 boundary

Only a completed training record and qualified checkpoint that pass independent
reconstruction may be bound into the already reviewed staged lifecycle. Binding
proceeds one revision at a time and starts with the G2 runner authority only;
the G2 finalizer, candidate publisher, G3, and full-promotion identities remain
unset until their exact predecessor exists.

The G2 attempt is role-global and keyed by the paired-manifest identity,
untouched-G2 set commitment
`0c9d5cfb6fdeec9be17a1afa8aed13fb62848a06594782c98933e1db8a2e1402`,
and immutable evaluation-protocol generation. It reserves before the first G2
byte open and consumes exactly the eight G2 scenes and 469 parent transitions
once. Checkpoint/source renaming or substitution cannot create a second
attempt.

After G2 contact there is no retraining, checkpoint substitution, calibration
refit, threshold or schedule change, ablation promotion, or second G2 run. A G2
failure is terminal for this candidate. A pass licenses only the existing
staged G2 finalizer and then candidate publication; it is not G3, held-out,
runtime, navigation, hardware, production, or promotion authority.

## Required implementation review

Before any exact command, a trainer implementation, independent verifier, and
immutable publisher must be additive, source-hash frozen, and reviewed by an
agent other than the implementation author. The review must use synthetic and
temporary fixtures only, prove the complete source closure and single-operation
authority, reproduce all schedule/loss/ranking/calibration arithmetic, falsify
role and path substitution, verify GPU0-only fail-closed discovery, and confirm
the canonical output root and all role payloads remained unopened.

Until those checks pass and all prerequisite dataset and V4 two-seed identities
exist, every training, calibration, checkpoint-use, G2, G3, held-out, runtime,
hardware, navigation, production, and promotion license remains false.

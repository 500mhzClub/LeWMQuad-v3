# Shared JEPA V5 full-training execution amendment V2

Date: 2026-07-13

Status: **frozen before implementation, payload access, GPU execution, or
learned output; different-agent review required**

## Purpose and authority

This additive V2 closes the three findings in the frozen independent review of
the Shared JEPA V5 full-training execution amendment V1. It does not rewrite
the V1 scientific design. The exact V1 bytes remain the base contract, and this
document supersedes only:

1. the GPU-smoke/exact-attempt reservation ordering;
2. the authority treatment of the live navigation-readiness status; and
3. the interpretation and claim boundary for the selection-role ablation.

No other V1 value, threshold, role, count, seed, schedule, optimizer, loss,
selection rule, calibration rule, development gate, failure rule, or one-shot
G2 boundary changes.

This document authorizes no implementation, repository payload open, model or
checkpoint open, GPU execution, training, selection, calibration, causal claim,
G2/G3 contact, held-out access, runtime use, navigation evaluation, hardware
use, production use, or promotion. Each boundary remains closed until the
required additive implementation, exact input binding, and different-agent
review exist.

## Frozen V1 and review evidence

| Artifact | SHA-256 |
|---|---|
| V1 amendment | `b21d01d062543cc7b7f3f5281f66ac40df76726c678a9364f7a4e451b035a4a7` |
| V1 author handoff | `fa0a497fad2f17a5d0919e1160b6040cbe13740315cfc180418d99dbf494d6bc` |
| V1 independent test | `b2959ea11cff80091a9f94c61dde14750726332001326c0fa30bd186418c6b38` |
| V1 independent review | `2cd1bf56edd213041496c67238dcf540f2f4a1b72e9abae529e327b4e22c125c` |
| V1 BLOCK record | `c3debd1ee4394e8916b8bfeb7d9237c44f3152e0fd36c27cdf84819c3e356273` |

V1 remains BLOCK evidence and has no standalone execution authority. An
implementation must consume the exact V1 scientific body together with this
V2 override and a future V2 PASS record. It may not choose between V1 and V2
clauses.

## Stable authority correction

`docs/lewm_go2_navigation_work_readiness_goal_2026-07-13.md` is a live status
record. It is informational only and is deleted from the authoritative parent
closure established by the V1 governing-design table. No past, current, or
future hash of that path may:

- gate implementation or execution;
- enter an attempt identity, source-closure identity, or qualification result;
- be required to reproduce after a legitimate status update; or
- override a frozen scientific or lifecycle clause.

An execution record may include the path and its observation time only in a
field explicitly named `non_authoritative_status_context`. Such a field is
excluded from source-closure comparison and cannot affect PASS, BLOCK,
selection, calibration, retry, G2, or promotion decisions.

The stable training-design authority is the exact V1 amendment plus this exact
V2 amendment and their different-agent review records. The other frozen V1
source and design parents remain authoritative at their V1 hashes. A future
implementation manifest must bind that stable closure and must not reintroduce
the live readiness document as a hashed parent.

## Science retained without change

The following V1 decisions are incorporated literally and are not reopened by
V2:

- both V4 seeds must pass every `N=5,16,32,320` rung, while only seed
  `20260710`, `N=320` may migrate;
- one CPU FP32 post-migration initialization at seed `20260712` is copied
  byte-identically into promoted-JEPA and matched-no-JEPA arms;
- train role only supplies gradients: 72 scenes, 4,262 pairs, 8,524 endpoint
  uses, and 7,777 unique endpoints across the eight fixed families;
- CPU schedule seed `20260713`, 128,000 pair presentations, 8,000 updates,
  effective batch 16, microbatch four, and accumulation four;
- AdamW, the fixed warmup/cosine learning-rate formula, FP32, clip `1.0`, and
  exactly one EMA update after every optimizer step;
- the promoted loss is the complete established JEPA package plus complete
  current and next V4 supervision, with all four equal V4 components;
- the no-JEPA arm changes only backward-loss membership and cannot select,
  replace, initialize, average with, or promote a checkpoint;
- checkpoint-selection uses only its 8 scenes, 495 pairs, and 924 unique
  endpoints, with the fixed physical/JEPA gates and unrounded lexicographic
  ranking at updates `1000..8000`;
- calibration uses only its 8 scenes, 415 pairs, and 759 unique endpoints,
  with the fixed six-parameter vector scaling and conservative global threshold
  grid;
- aggregate and every-family development gates remain mandatory;
- all exact neural work remains restricted to GPU0 named exactly
  `AMD Radeon AI PRO R9700` with at least 32 GiB; Raphael/GPU1/iGPU, fallback,
  multi-GPU, mixed precision, and adaptive batch changes remain forbidden;
- the immutable output, complete actual-open ledger, independent
  reconstruction, no-retry failure semantics, and role-global one-shot G2
  boundary remain mandatory; and
- all six production-stage authority identities remain unset.

## Two distinct operations

V2 separates capacity validation from the exact training attempt. They use
different namespaces, reservations, process lifetimes, ledgers, and receipts.
Neither operation may inherit a live Python object, CUDA/ROCm context, tensor,
allocator cache, RNG state, optimizer state, descriptor, worker, or temporary
file from the other.

The only allowed order is:

1. V2 design receives different-agent PASS;
2. additive preflight/exact-executor source receives different-agent PASS;
3. the payload-free preflight namespace is reserved;
4. the synthetic R9700 smoke runs once and publishes an immutable receipt;
5. a different agent verifies and freezes that receipt;
6. a future exact-execution binding freezes the receipt hash, all prerequisite
   V4/dataset hashes, runtime closure, and implementation closure;
7. a fresh standard-library-only operation reserves the exact attempt; and
8. only after that exact reservation may a fresh process perform any exact
   Torch/GPU, model, checkpoint, RGB, label, or role-payload open.

The preflight is not part of the exact attempt, cannot consume or create the
exact namespace, and cannot be cited as learned evidence. The exact attempt
cannot silently rerun the preflight.

## Immutable payload-free GPU preflight

The sole preflight namespace is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v2_preflight`

An isolated, reviewed preflight reserver must create that directory with
exclusive no-follow semantics and retain its directory descriptor before the
first preflight GPU-runtime access. A completed preflight has exactly:

```text
reservation.json
source_closure.json
access_ledger.json
gpu_smoke_receipt.json
completed.json
```

`completed.json` is written last and binds the ordered inventory, byte counts,
file hashes, semantic hashes, directory identity, and preceding receipt hashes.
All writes are exclusive, descriptor-relative, fsynced, and immutable. An
existing namespace, alias, partial tree, or terminal failure cannot be reused,
overwritten, resumed, or treated as a pass.

The smoke may open only the independently reviewed implementation/source
closure and generate in-memory deterministic synthetic tensors. It must not
open or derive from any repository dataset, row index, role sidecar, source
scene, frame, RGB, ray label, manifest, render plan/summary, V4 fit checkpoint,
V5 checkpoint, learned tensor state, calibration payload, G2/G3 input, held-out
input, runtime/navigation result, physical executor/reset input, or production
artifact.

The smoke instantiates the exact reviewed production-config V5 class from a
fresh deterministic synthetic initialization. It uses the exact source/pixel
shapes, microbatch of four, four accumulated backwards, complete promoted
joint-loss path, loss division by four, global clip `1.0`, AdamW step, and one
post-step EMA update. Synthetic current/next RGB, actions, realized/commanded
deltas, calibrations, ground heights, masks, counterfactuals, and complete
four-component current/next V4 labels must exercise every required input and
loss branch. Synthetic values test shape, memory, finiteness, backward, and
device support only; they are not accuracy or migration evidence.

The preflight device contract is exactly the V1 device contract:

- `HIP_VISIBLE_DEVICES=0` and `ROCR_VISIBLE_DEVICES=0`;
- `HSA_OVERRIDE_GFX_VERSION` absent;
- `cuda:0` reports exactly `AMD Radeon AI PRO R9700` and at least 32 GiB; and
- no Raphael, GPU1/iGPU, CPU fallback, or second accelerator is visible or
  used.

`gpu_smoke_receipt.json` must bind at least:

- V1, V2, implementation, model-source, and test-source hashes;
- the full environment, Python, Torch, ROCm, kernel, driver, and device
  identity/memory observations;
- production config, all tensor shapes/dtypes/devices, seed, optimizer fields,
  accumulation count, gradient clip, EMA count, and loss-component inventory;
- finite per-component losses, total loss, gradient norm before/after clipping,
  optimizer-step count, EMA-step count, peak allocated/reserved memory, and
  terminal synchronization result;
- the ordered content-chained access ledger and proof of zero forbidden opens;
  and
- terminal `PASS` or sanitized `FAIL`, with no adaptive retry.

OOM, unsupported deterministic operation, wrong device identity, nonfinite
state, missing loss branch, wrong cadence, forbidden open, or durability
failure writes one terminal preflight failure when possible. It creates no
exact attempt and requires a new dated amendment before another capacity
configuration is tried.

The smoke process must terminate after publication, close every descriptor,
destroy its objects and workers, synchronize and release the GPU context, and
return success only after an independent verifier reconstructs the immutable
receipt. A receipt PASS licenses only creation of a future exact-execution
binding. It licenses no payload or training access.

## Exact-attempt reservation boundary

The sole exact namespace is revised to:

`.generated/go2_shared_observable_camera_ray_jepa_v5/full_training_v2`

The future reviewed exact-execution manifest must contain the independently
accepted preflight receipt SHA-256 plus every V1-required source, runtime,
dataset, raw-supervision, V4 ladder, migration, and implementation identity.
Supplying that already reviewed manifest to the reserver does not authorize an
open of any bound payload.

A fresh standard-library-only reserver must claim the exact namespace and
write/fsync `reservation.json` while holding its directory descriptor. This
must occur before the exact process performs any of the following:

- Torch import or GPU-runtime/device initialization;
- model-source import, model construction, or tensor-state construction;
- V4/V5 model or checkpoint open;
- train, checkpoint-selection, probability-calibration, RGB, label, source,
  or other repository payload open; or
- worker creation, decode, inference, backward, optimization, EMA, selection,
  or calibration.

Only frozen authority/implementation-manifest bytes needed by the
standard-library reserver may be read before reservation. The first subsequent
exact-run input open is the frozen preflight receipt, whose observed hash must
equal the execution-manifest binding. The exact ledger records every later
open. A missing, aliased, mutable, failing, or mismatched receipt consumes the
reserved exact attempt and fails before Torch or payload access.

The exact process must be newly spawned after reservation. It cannot share the
preflight process or inherit its environment mutations, file descriptors,
workers, RNG state, GPU context, model, tensors, optimizer, allocator, or cache.
It rediscovers and revalidates the R9700 under the same fail-closed device rule.

The V1 exact output inventory is retained under `full_training_v2`, with these
two semantic changes:

- add `preflight_receipt_binding.json`; and
- replace `ablation_comparison.json` with
  `selection_role_ablation_diagnostic.json`.

The exact reserver, ledger, immutable completion, terminal failure, no retry,
and independent reconstruction rules otherwise remain literal V1 rules.

## Selection-role ablation is diagnostic only

The matched no-JEPA arm still trains for the same 8,000 updates and is evaluated
once at the immutable promoted update on the same ordered 495
checkpoint-selection pairs. This preserves the matched engineering diagnostic
and V1 controls. It does not create an independent generalization population.

`selection_role_ablation_diagnostic.json` may report raw per-scene,
per-family, and aggregate values for both arms and exact promoted-minus-ablation
deltas. It must carry:

```text
population_role = "checkpoint_selection"
interpretation = "matched_development_diagnostic_only"
causal_generalization_claim_authorized = false
qualification_or_selection_effect = "none"
```

No selection-role ablation result, sign, margin, family count, calibration
result, or JEPA-health result may:

- support wording that JEPA caused or improved generalization;
- qualify or disqualify the promoted checkpoint;
- change checkpoint ranking, calibration, thresholds, training, or G2 access;
- promote or substitute the no-JEPA arm; or
- trigger a retry, new seed, schedule change, or post-result intervention.

The V1 sentence permitting a causal development-generalization claim from the
selection-role comparison is deleted. A selection-role difference is
descriptive evidence about that reused development population only.

## Optional future untouched comparison

V2 authorizes no untouched two-arm evaluation. Any later causal-generalization
comparison requires a separate dated preregistration and different-agent PASS
before the first byte, metadata row, count, metric, or result from its untouched
population is opened. That preregistration must freeze a population used by
neither training, checkpoint selection, nor probability calibration; both
already fixed arm checkpoints; both already fixed calibrators and global
thresholds; evaluation order; metrics; missing-class behavior; namespace;
single-attempt ledger; and failure semantics.

The existing one-shot G2 population is not implicitly available for this
purpose. Reusing G2 would require an explicit additive lifecycle amendment that
preserves its role-global one-shot reservation and fixes whether both arms are
part of the one attempt before any G2 contact. Without that reviewed amendment,
G2 remains promoted-arm only and closed to the ablation.

If a future untouched comparison is authorized, the narrow claim predicate is
fixed now. Using unrounded raw accumulators and the V1 normalized physical-gate
margin definition, let:

```text
M_arm       = minimum aggregate normalized physical-gate margin
M_arm,f     = minimum normalized physical-gate margin in family f
P_arm       = admitted_FREE_true_FREE_count / admitted_FREE_count
delta_M     = M_promoted - M_ablation
delta_M_f   = M_promoted,f - M_ablation,f
delta_P     = P_promoted - P_ablation
```

Both admitted-FREE denominators must be positive. Any empty required class,
nonfinite value, structural/access failure, or failed arm makes the predicate
false. The future report may state only that the matched JEPA arm improved the
preregistered untouched criterion when all are true:

```text
delta_M > 0.0
count_f(delta_M_f > 0.0) >= 5 of the same 8 frozen families
delta_P >= 0.0
```

Thus the exact precision comparison is
`P_promoted >= P_ablation`, evaluated directly without rounding under each
arm's pre-contact frozen global calibration and threshold tuple. Comparing
either arm only with the absolute `0.99` gate is not a substitute for this
between-arm inequality.

Passing this predicate would support only the stated matched-run claim on the
separately frozen untouched population. It would not establish universal,
population-level, navigation, runtime, G2/G3, safety, production, or promotion
causality. A future preregistration may add stricter statistical requirements;
it may not weaken these conditions after access.

## Failure and publication consequences

The V1 numeric, structural, provenance, access, and no-retry rules remain in
force. These are additional structural failures:

- preflight and exact namespaces, processes, reservations, or ledgers overlap;
- preflight opens a repository payload or emits learned evidence;
- exact access precedes exact `reservation.json` durability;
- the exact process reuses any preflight live state;
- a live readiness-status hash becomes authoritative;
- selection-role ablation affects qualification or supports a causal claim; or
- an untouched comparison occurs without its own pre-contact preregistration.

No such failure permits a second smoke, exact retry, batch adjustment, device
fallback, checkpoint substitution, threshold refit, causal reinterpretation,
or G2 contact. A new dated amendment is required.

## Required different-agent review

Before additive implementation, a reviewer other than
`/root/raw_plan_v2_qa` must rehash V1, its handoff, its independent test/review/
BLOCK record, and this V2. The review must verify that:

1. the smoke has its own immutable namespace and payload-free receipt;
2. the exact reservation precedes every exact Torch/GPU, model, checkpoint,
   RGB, label, and role-payload open;
3. no process or live state crosses from preflight into the exact attempt;
4. the live readiness record is informational and absent from authority hashes;
5. the selection-role panel is diagnostic and has no qualification effect;
6. any future untouched claim requires separate pre-contact preregistration and
   the exact `delta_M`, family-count, and `delta_P >= 0` comparisons; and
7. every retained V1 scientific and one-shot G2 rule is unchanged.

A PASS licenses only additive preflight/reserver/trainer/verifier/publisher
implementation against V1 plus V2. It does not license preflight execution,
dataset use, V4 execution, model/checkpoint access, training, selection,
calibration, untouched comparison, causal claim, G2/G3, held-out, runtime,
navigation, hardware, production, or promotion.

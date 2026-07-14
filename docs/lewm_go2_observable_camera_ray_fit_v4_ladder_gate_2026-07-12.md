# Observable camera-ray V4 development fit ladder gate

Status: **amended, non-authorizing contract**. The original document and gate
were frozen before V4 model fit output. The reviewed V2 partition amendment
supersedes only the N1/N4 partition and exposure clauses described below. This
contract authorizes neither RGB access nor training, checkpoint loading,
heldout/G2/runtime access, or promotion.

## Exact inputs

- dataset manifest file/content: `2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85` / `9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812`;
- audit receipt file/content: `2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c` / `a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76`;
- RGB receipt content: `d763d7ae294e4e5a9e5f2352672913bc06411388d92abe1fb0f5090dfc41d5c3`;
- fit rungs: `N=5,16,32,320`, using nested registered-family round-robin subsets;
- seeds, in order: `20260710`, then `20260711` only after the complete first-seed ladder passes.

Every larger rung must bind the canonical passing gate for the immediately
preceding rung by caller file SHA and content SHA. Seed `20260711` N5 must also
bind the complete passing seed-`20260710` ladder gate. Direct N16/N32/N320 or
second-seed bypass is structurally invalid; N4 is no longer a ladder rung.

The exact subset content hashes are frozen per rung in the gate module. A
caller-generated frame-key list with the right length is insufficient. The
namespaced rank separator is the two literal ASCII bytes backslash and zero
(`5c30`), matching the preregistration command; it is not a NUL byte.

Exact target partitions are independently frozen by file/content SHA-256
`4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a` /
`8dd54d178e3c00a8622d89e4e371a115e1391f34588f667c20cd95b970fc68d2`.
The CPU-only reproducer and amendment file SHA-256 values are respectively
`4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed`
and `1e65f8884b1b8e0ad2219ddad54f79f9fabae514bfcaa048b29c8113b076ac1f`.
Each rung binds its exact ordered keys, family counts, target signature,
ordered per-frame target hashes, and ordered target-byte hash directly from
that source boundary. No value is inferred from reported result metrics.

## One-attempt artifact chain

Before GPU discovery, RGB decode, model construction, or training, the trainer
atomically creates the sole canonical directory
`attempts/seed_<seed>/n<fit-size>/` and writes immutable `reservation.json`.
The directory is never removed. A normal failure removes partial model/result
files and writes a sanitized `failed.json`; an uncatchable interruption leaves
the reservation alone and still consumes the attempt. A reviewed amendment is
required before any retry.

A successful attempt contains exactly `reservation.json`, `checkpoint.pt`,
`result.json`, and `completed.json`. The completion receipt is written last and
binds the other three files by caller SHA-256 and semantic content SHA-256.
Stage finalization consumes all four canonical caller hashes plus the actual
caller-hashed trainer authorization and independent review record. It validates
checkpoint schema, metadata, state keys/shapes/dtypes, finiteness,
deterministic buffers, and its semantic state manifest on CPU. It does not run
inference from reported aggregates alone.

A separately licensed verifier must first create the sole canonical
`metric_verifications/seed_<seed>_n<N>.json`. It reloads the exact checkpoint,
selected train targets, matched RGB, and cyclic wrong RGB, reruns inference,
and reconstructs every loss, confusion matrix, depth quantile/evidence hash,
raster NLL, family metric, and gate decision. Stage finalization re-executes
the 180-file CPU-only target reproducer before checkpoint import or inference
and must reproduce the frozen ordered target-byte commitments. It then requires
the verifier receipt byte-for-byte; a self-consistent metric
mapping is insufficient. The verifier authorization is currently all false.

Gate outputs have one derived filename under the canonical gate root:
`stage_seed_<seed>_n<N>.json`, `seed_<seed>.json`, and `two_seed.json`. There is
no caller-selectable output alias.

## Frozen exposure

| N | Steps | Train batch | Eval batch |
|---:|---:|---:|---:|
| 5 | 1000 | 1 | 1 |
| 16 | 1200 | 1 | 1 |
| 32 | 1600 | 1 | 1 |
| 320 | 3200 | 1 | 1 |

All rungs use AdamW, learning rate `1e-4`, weight decay `1e-4`, gradient clip
`1.0`, FP32, no autocast, one frozen attempt, and the 3,105,513-parameter
`ObservableCameraRayEvidenceV4Model`. The four registered losses each have
weight `0.25`. Schedule hashes for both seeds are frozen in the gate module;
the algorithm is CPU `torch.Generator.manual_seed`, concatenated `randperm(N)`
cycles, then the first `steps * batch_size` indices.

Exactly N selected train RGBs may be hash-opened and decoded once, in at most
six spawn workers with one native thread each, then rehashed
once before publication. Nonselected RGB, heldout, G2, runtime, and GPU1 opens
must remain zero.

CPU handles authorization, audits, bounded RGB decoding, checkpoint structure
validation, and finalization. Neural fitting and evaluation are pinned only to
visible GPU0, the R9700. The Raphael iGPU/GPU1 is hard-rejected and is never a
fallback.

## Numeric thresholds

| N | Pixel hit BA | Depth median/p95 m | Ground BA | Distance/family BA | Raster NLL | Raster BA | Present class recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.99 | 0.06 / 0.15 | 0.99 | 0.97 / 0.97 | 0.06 | 0.99 | 0.97 |
| 16 | 0.97 | 0.08 / 0.20 | 0.97 | 0.94 / 0.94 | 0.10 | 0.97 | 0.95 |
| 32 | 0.95 | 0.10 / 0.25 | 0.95 | 0.92 / 0.92 | 0.15 | 0.95 | 0.92 |
| 320 | 0.95 | 0.10 / 0.25 | 0.95 | 0.92 / 0.92 | 0.15 | 0.95 | 0.95 |

Distance thresholds apply to every nonempty registered distance group. Family
thresholds apply to every selected family. Raster recalls apply only to classes
present in that subset. Exact target class counts are bound in code. N5 has all
three raster classes present, so all three recalls are gated.

For every rung, wrong-RGB evaluation keeps each target's camera calibration fixed and
cyclically changes only RGB. The matched arm must beat the wrong-RGB arm by the
following preregistered minimums:

| N | Pixel BA drop | Depth median/p95 increase m | Ground BA drop | Raster NLL increase | Raster BA drop |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.08 | 0.08 / 0.12 | 0.08 | 0.08 | 0.08 |
| 16 | 0.10 | 0.10 / 0.15 | 0.10 | 0.10 | 0.10 |
| 32 | 0.11 | 0.11 / 0.18 | 0.11 | 0.11 | 0.11 |
| 320 | 0.12 | 0.12 / 0.20 | 0.12 | 0.12 | 0.12 |

The result also carries the exact identity/cyclic wrong-RGB index mappings,
finite loss components whose weighted sum reproduces total loss,
matched/wrong target-partition equality, raster NLL sum/count consistency, and
content-bound sorted-depth quantile evidence. Confusion-derived recalls and
balanced accuracies are recomputed rather than trusted.

## Failure semantics

Malformed provenance, counts, configuration, exposure, seed order, attempts,
access ledger, checkpoint receipt, target partitions, or prerequisite gates
are **structural invalidations** and raise. A structurally valid result below a
numeric threshold produces an immutable failed stage gate and a diagnostic
checkpoint may exist, but checkpoint use and the next rung remain unauthorized.
Thresholds must not be relaxed after seeing output; a failure requires a newly
reviewed exposure or architecture contract.

A passing stage may authorize only execution of the immediate next fresh-fit
rung. It never authorizes loading or deploying the diagnostic checkpoint;
checkpoint-use, heldout, G2, runtime, and promotion licenses remain false.
Before granting that narrow execution permission, the filesystem validator
reopens every canonical stage dependency and its reservation/result/checkpoint/
completion/metric files, re-runs the full finalizer, and compares the gate bytes
exactly. A seed gate repeats this process for all four stage chains.

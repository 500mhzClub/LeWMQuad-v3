# Camera V4 gate-aligned ladder V1 preregistration and handoff

Date: 2026-07-14

Status: **runtime implemented; source review and exact execution pending**

## Purpose

Replace the repeated fixed-N5 V14/V15/V16 wrapper succession with one practical,
parameterized development runner. The dataset, target partitions, model, five-term
gate-aligned objective, thresholds, and held-out boundary do not change.

## Terminal V16 boundary

V16 consumed its only attempt. Training, matched/wrong evaluation, and checkpoint
serialization occurred transiently, then the retained publisher rejected a
four-field native-thread receipt against a six-field schema. No numeric loss,
metric, checkpoint hash/bytes, result, completion, metric receipt, or gate was
printed or persisted. This records the qualitative full-compute fact; it does not
claim that nothing was computed.

The terminal reservation file/content hashes are
`1769d282f528c6c64b1fb67ad229c6ebf2dbc55ae61b1a53451a76538a69bf1c` /
`00fdc565b3791579ca4c6bbc090eac8db2d87b3e54d37e647b46dc9780a28e15`.
The failure file/content hashes are
`06ae522dc0748d6d0857e5d8cfd22d96fbc78e5e1463c30c8928670d2c22dd51` /
`c861eca6b88abe469ab73b29f0499f2a6e549d16c6f4ad266aaa9eb3dc8f49d5`.
The empty lock hash is
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The exact GPU-visibility receipt file/content hashes are
`cbda43a1b251d48eb400e263bd6e81645d02d44c9630a513b368de821c87545a` /
`06c72c6275bbb9101753774189b0987e12cfcf4e57cbcbe1329299f12b6df2ec`
(`3817` bytes).
V16 retry remains forbidden.

Row 0 below is a new preregistered infrastructure successor in a fresh shared
ladder namespace, not a V16 retry. It observes no V16 model state and retains the
same row-0 science.

## Frozen topology and compute

The runner derives the only next row; no seed, N, or output path is caller chosen.

| Row | Seed | N | Updates | Batch | Exposures | Schedule SHA-256 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 20260710 | 5 | 4000 | 5 | 20000 | `fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380` |
| 1 | 20260710 | 16 | 4000 | 5 | 20000 | `06f3ab002349bb8726d1abd7ae5350de711938b67a8ea7e7da7ae66145f9248e` |
| 2 | 20260710 | 32 | 4000 | 5 | 20000 | `5d93d4d4f4697635170a3739557ccdddc7da0e0bc9e874438802cf65298627fc` |
| 3 | 20260710 | 320 | 4000 | 5 | 20000 | `4084f8d5c14989cb76df4f01e4de46b0b6a88537ba607ccc4152795304bc3bd6` |
| 4 | 20260711 | 5 | 4000 | 5 | 20000 | `829d366eb9dcefdaad66596413939da455209909009e29177271ff5ed9c76c2e` |
| 5 | 20260711 | 16 | 4000 | 5 | 20000 | `57d5cd679ab7eb99654430a166a53985c21ffcf261faa70c8df0357ac7dc80f3` |
| 6 | 20260711 | 32 | 4000 | 5 | 20000 | `405632e5e6c8e26590debfa5139090ca89c4ac262930ca286ce82e9b9db1f10c` |
| 7 | 20260711 | 320 | 4000 | 5 | 20000 | `2b5475b725f1ae3c956adaef0a72153b0fdafd6d1ba36d827ed23792ac6a0b9a` |

This is fixed-compute scaling, not equal exposure per example. N320 receives
62.5 epochs. A failure means failure under this frozen budget, not proof of
asymptotic incapacity.

Every row constructs a fresh model and optimizer. The replication seed remains
constant across N within a seed for controlled comparisons. The runner records
the actual initial-state hash and derives a unique initialization identity from
that hash plus the canonical row/attempt identity. No predecessor checkpoint may
be deserialized or used as model input. Integrity-only byte rehashing of a prior
completed bundle is still required before progression. Only row 3, seed
20260710/N320, may later be considered by a separately reviewed Shared-V5
training contract.

## Frozen runtime direction

- GPU0 must expose exactly the R9700; Raphael and every other visible device are
  forbidden.
- All six native thread variables, including VECLIB and BLIS, equal one.
- Torch intra/inter-op setup occurs once before other Torch work.
- RGB decoding is serial and hash-verified.
- Training uses 4000 AdamW updates, B5, FP32, LR/weight decay `1e-4`, clip `1.0`,
  fresh initialization, and the five-term V15 gate-aligned objective.
- Evaluation is batch-one matched RGB plus cyclic wrong RGB.
- A fresh isolated child strictly reloads the checkpoint and reconstructs both
  controls and the frozen numeric gate.
- Warning evidence is fully allowlist-validated, then stored as a compact count
  histogram rather than tens of thousands of repeated strings.
- Any post-reservation failure is terminal and stops the ladder. There is no
  retry, threshold change, data change, or warm start.

The sole root is
`.generated/go2_observable_camera_ray_fit_v4/gate_aligned_ladder_v1`.
One different-agent source review will cover the complete runner/test/proof
closure and the fixed serialized eight-row authority. There are no per-rung
source copies or review files.

## Implemented runtime

The same runner now owns the complete thin lifecycle for every row. It performs
stdlib-only environment and review checks before delayed Torch imports, validates
the exact frozen data closure, derives the only legal row, and materializes the
selected target partition before reservation. It does not decode RGB, construct a
trained output, or calculate an outcome before the row is exclusively reserved.

After reservation it serially decodes only the selected hash-bound RGBs, trains
the fresh model, evaluates matched and cyclic-wrong RGB batch-one controls,
rehashes the complete input closure, and serializes only the final-update
development checkpoint. An isolated `python -I -B` child then opens that
checkpoint exactly once, reconstructs a fresh strict-loaded model, independently
recomputes both controls with a separately expressed five-term loss, and rebuilds
the unchanged retained V4 threshold gate. The parent writes the metric receipt
and gate only after the child succeeds.

The gate contract is frozen by threshold-contract SHA-256
`408b10d8dc4f3734acb8ba17e974da4a84108a9c964d9b10787e7df59b165c60`.
The retained gate source is frozen at
`aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad`.
The source review also binds every directly reused runtime source and the exact
dataset, audit, trainer-authorization, trainer-review, subset, target-partition,
model, objective, optimizer, and evaluation contracts.

Every completed row is reopened and its actual reservation, checkpoint, result,
completion, metric receipt, and gate bytes are rehashed and cross-bound before a
later row can start. Rows 1--3 bind their predecessor; row 4 binds all four first-
seed gates; rows 5--7 bind their predecessor. Same-seed rows must reproduce the
same initial state bytes, while the initialization identity remains unique because
it also binds the one-shot row attempt. `predecessor_checkpoint_opens=0` records
that no predecessor is loaded for scientific/model use; its bytes are read only
to verify their frozen hashes.
After all eight rows pass, one `ladder_gate.json` binds the reviewed source,
frozen topology, threshold contract, and exact eight row-gate byte identities.
It grants no downstream checkpoint-use or benchmark authority.

An infrastructure, validation, device, or nonfinite failure after reservation
removes only the runner-owned partial leaves, writes `failed.json`, and permanently
stops the ladder. A valid numeric miss preserves the full measured bundle and a
`failed_numeric_gate` gate, then permanently stops the ladder. Neither case is
retryable. Numeric values are not printed before they are immutably persisted.

## Execution boundary

Exact execution remains blocked until the complete runner/test/proof closure is
committed and one different agent publishes the canonical source-review JSON with
`runtime_complete=true`. After that, each invocation runs exactly one row:

```text
env -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  -u HSA_VISIBLE_DEVICES -u HSA_OVERRIDE_GFX_VERSION \
  -u NVIDIA_VISIBLE_DEVICES -u ONEAPI_DEVICE_SELECTOR -u ZE_AFFINITY_MASK \
  HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  /home/andrewknowles/TinyQuadJEPA/bin/python -I -B \
  scripts/run_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1.py \
  --next --source-review-sha256 REVIEW_FILE_SHA256
```

Only row 3 (seed 20260710/N320) is later eligible to be named by a separately
reviewed Shared-V5 training successor. This ladder itself authorizes no held-out,
G2, navigation, production, promotion, or checkpoint migration work.

# V4 N5 full-panel V2 implementation handoff

Date: 2026-07-13

Status: **additive source successor complete; exact execution remains blocked
pending a different-agent V2 review**

Implementation author: `/root/v4_execution_successor_review`

This is an author handoff, not a review or execution authorization. The author
must not create the canonical V2 PASS review and must not run the exact attempt.

## Governing BLOCK evidence

V2 preserves and rehashes the complete frozen V1 candidate and its independent
BLOCK before every authority issuance/use:

- V1 review MD:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review_2026-07-13.md`,
  SHA-256
  `11479b03ff9eac24dd5541d38faeda480739c8d17de7b2b658759e306ace2d5e`;
- V1 BLOCK JSON:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review_block_2026-07-13.json`,
  file/content SHA-256
  `ccd8d97988d2ce165722703fbfcf813758ee42a5408e02d26bf7db38d8ea506e` /
  `99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7`;
- frozen V1 adversarial tests:
  `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review.py`,
  SHA-256
  `387147a8dd6fe1a20184284a05c18df73419ca91c21054eb378e79a8194d5b3b`;
- V1 author handoff:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_implementation_handoff_2026-07-13.md`,
  SHA-256
  `8f4735a3ecd20a8c19bd729fdaf71ceb60a3a884de717423e8f84ef6ef2745f7`.

The five frozen V1 source hashes remain exactly those recorded by the BLOCK:

- policy `875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88`;
- launcher `3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8`;
- trainer `48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca`;
- verifier `00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0`;
- finalizer `1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b`.

## Additive V2 source closure

- authority/policy:
  `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py`,
  SHA-256
  `096b597b0e84a6822fd8fcdd8221da27e95757aaa2c05ca148afad6e23ad60d2`;
- isolated launcher:
  `scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py`,
  SHA-256
  `03311bb48da80b912c2576844adf5cd488c1b9a0818268d2252902d860436591`;
- authority/reservation trainer boundary:
  `scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py`,
  SHA-256
  `357369b652c489ab99937c06afaed0ec4cf66aa1f46017f74f5dac46da93d3aa`;
- independent inference verifier:
  `scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py`,
  SHA-256
  `cab757839c3d784cb5760f30c2bde6163311bfbf87df1620c9c0f77ff69b624b`;
- fail-closed finalizer:
  `scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py`,
  SHA-256
  `a5dc625b8b270913df56d8b5044c263ba3fdbd1ef6cb3e6f62e084a5335ee323`.

Test sources:

- helper:
  `lewm/tests/n5_full_panel_v2_test_support.py`, SHA-256
  `95892d289798580e0911eab1be43e8e899125ee8484eb2fa4e3afd5af2ed0557`;
- adversarial V2 suite:
  `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py`,
  SHA-256
  `e249dce8af66d8e6709f1823f433ba76a56be8a54129f0620e20efa61d9ed8dd`.

## Authority closure

`VerifiedAuthorityV2` cannot be normally constructed, copied, deep-copied,
dataclass-replaced, serialized, or reconstructed into a recognized receipt.
The receipt contains immutable primitive/byte fields and no issuer, registry,
lock, state record, marker, or mutable collection reference. The explicit test
capability has empty slots and likewise contains no issuer/registry reference.

Exact issued-object records, immutable issuance digests, lifecycle states, test
scope roots, and locks live only in the policy API's closure storage. Records
are frozen values; a state transition replaces a record atomically under an
`RLock`. Validation requires the exact retained object identity and original
digest, correct purpose and target path, correct lifecycle state, and a fresh
rehash of the full canonical source review/static closure.

Production purposes bind to a closed path map:

- `exact_run` -> the one canonical N5 attempt;
- `metric_verification` -> the canonical metric receipt;
- `finalization` -> the canonical gate receipt.

The production trainer exposes no caller-controlled attempt path. Temporary
paths require a separate exact `TestAuthorityCapabilityV2`, whose root and
records are closure-held, whose purpose is permanently `test_exact_run`, and
which rejects the repository and canonical output. Object-level mutation of a
test receipt cannot promote it: both its test issuer and production validator
reject the changed issuance digest/unknown identity.

`run_exact()` atomically consumes `issued -> active` before entering retained
science. Reservation consumes `active -> claiming -> claimed`; every replay,
cross-issuer receipt, alternate path, concurrent duplicate consumer, clone, or
mutated receipt is rejected.

## Crash recovery and durability

Reservation is serialized by a private regular advisory-lock file. `flock`
releases automatically on process death. Under that lock V2 scans both the
legacy fixed staging name and the V2 unique staging namespace. Every V2 staging
directory is a random private `0700` sibling and contains a fully fsynced
reservation plus a content-hashed recovery manifest before it is claimable.

The reviewed recovery table is:

| Classification | Required action |
| --- | --- |
| legacy/incomplete | inventory, remove without claim, fsync parent, create a unique private staging directory |
| complete | rehash source review, reservation, and manifest; append recovery evidence; resume exactly one |
| foreign | inventory metadata/content safely, remove without claim, fsync parent |
| mutated | inventory, remove without claim, fsync parent; never resume it |
| multiple complete equivalents | choose the lexical first, remove the other validated equivalents, record every action |

The canonical `reservation.json` retains the `preclaim_recovery` ledger. A
crash during recovery refresh can leave only an incomplete/mutated staging
state, which the same table safely removes on the next process. No staging
state can silently become an attempt.

Before rename, V2 captures the staging `(st_dev, st_ino)`. This exact identity
prevents a losing concurrent caller from terminalizing the winning caller's
attempt. After `os.rename(staging, attempt)`, the next statement is
`_fsync_directory(seed_root)`; no failure injection or protected transition is
between them. An exception can write a terminal receipt only when the
canonical directory has the caller's captured staging identity.

Every caught post-claim failure removes partial success artifacts, fsyncs the
attempt, exclusively writes and fsyncs `failed.json`, then fsyncs both the
attempt and seed parent. Successful publication writes completion last and
also fsyncs the attempt and seed parent.

## Retained scientific and downstream contracts

V2 delegates numerical/data work to the exact frozen V1 trainer after V2
authority consumption and replaces only V1's reserve/failure/publication/RGB
worker callbacks. The retained V1 hashes are part of the V2 source review.

Therefore the experiment remains exactly seed `20260710`, fresh N=5 model,
AdamW, 400 updates, batch size five, 2,000 exposures, LR/weight decay `1e-4`,
float32, no autocast, clip `1.0`, four `0.25` losses, final update only, and
matched plus cyclic wrong-RGB batch-one evaluation. Its schedule SHA-256 remains
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

Selected RGB is still rehashed after training and exact-input revalidation but
before checkpoint serialization. The verifier still reloads and rehashes every
checkpoint tensor, constructs a fresh model, reopens the five selected RGBs,
recomputes both controls and all metrics, and calls the unchanged N5 gate. V2
wrappers retain GPU0 R9700 validation, Raphael rejection, five-worker maximum,
one native thread per process, and zero access to held-out, G2, selection,
calibration, runtime, hardware, or production roles.

## CPU verification

All commands hid accelerator devices, disabled external pytest plugins, and
capped OMP/MKL/OpenBLAS/NumExpr threads to one. Three suites ran concurrently:

```text
V2 adversarial/recovery/durability suite: 20 passed in 1.12s
retained V1 focused suite:               17 passed in 0.73s
retained predecessor closure:            31 passed in 2.04s
py_compile for all five V2 sources:       passed
```

The V2 suite covers direct construction, object shells, copy/deepcopy,
dataclass replacement, serialization, mutation, test promotion, cross issuer,
source-review mutation, atomic concurrent consumption, one-path replay,
concurrent claim ownership, legacy/incomplete/complete/foreign/mutated and
multi-complete recovery, private staging cleanup, preclaim failure, postrename
fsync adjacency, terminal parent durability, retained schedule, and import
safety.

Reproduction command:

```bash
env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
  PYTHONPATH=/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /usr/bin/python3 -m pytest -q \
  lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py
```

The retained 48-test command is copied verbatim from the V1 handoff and must be
rerun by the reviewer.

## Required independent review

An agent other than `/root/v4_execution_successor_review` must independently:

1. rehash this handoff, all five V2 sources, both V2 test sources, all frozen V1
   sources, and the V1 review/BLOCK/exploit evidence;
2. rerun the V2 suite and the retained 48-test command with threads capped to
   one and accelerator devices hidden;
3. inspect the closure-held lifecycle, canonical production path surface,
   staging recovery table, exact inode ownership check, and rename/fsync source
   ordering adversarially;
4. confirm the canonical V1 and V2 PASS review files and the entire canonical
   output root are still absent;
5. only if every check passes, exclusively create
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_independent_review_2026-07-13.json`
   from `expected_source_review_core(...)`, add its canonical
   `content_sha256`, and record its file SHA-256 in a separate reviewer report.

The required schema is
`lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_source_review_v1`.
Until that exact different-agent PASS review exists and revalidates, the V2
launcher fails closed.

No retry, N16, second seed, V5, G2, held-out, selection, calibration change,
runtime, hardware, navigation, production, or promotion authority is granted.
At handoff time no exact manifest/target/RGB/checkpoint/model/accelerator input
was opened and no canonical output directory was created.

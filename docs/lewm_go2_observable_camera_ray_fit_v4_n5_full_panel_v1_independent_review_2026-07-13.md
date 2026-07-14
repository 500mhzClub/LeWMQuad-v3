# V4 N5 full-panel V1 independent review

Date: 2026-07-13

Verdict: **BLOCK**

Reviewer: `/root/downstream_integration_review`

The frozen V1 sources implement the preregistered scientific experiment,
selected-RGB revalidation, independent checkpoint inference, unchanged numeric
gate, and fail-closed downstream licenses. They do not provide an unforgeable,
single-use execution authority or a crash-durable one-attempt reservation.
Therefore the exact attempt is not authorized.

The canonical PASS review JSON
`docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review_2026-07-13.json`
was deliberately not created.

## Frozen artifacts

- policy:
  `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`
  - SHA-256:
    `875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88`
- launcher:
  `scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`
  - SHA-256:
    `3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8`
- trainer:
  `scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`
  - SHA-256:
    `48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca`
- verifier:
  `scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`
  - SHA-256:
    `00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0`
- finalizer:
  `scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py`
  - SHA-256:
    `1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b`
- implementation handoff:
  `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_implementation_handoff_2026-07-13.md`
  - SHA-256:
    `8f4735a3ecd20a8c19bd729fdaf71ceb60a3a884de717423e8f84ef6ef2745f7`
- independent adversarial tests:
  `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_independent_review.py`
  - SHA-256:
    `387147a8dd6fe1a20184284a05c18df73419ca91c21054eb378e79a8194d5b3b`

All five candidate source hashes and the handoff hash exactly matched the review
request. The candidate source, candidate tests, handoff, trigger amendment, and
preregistration were not edited.

## Blocking findings

### 1. The verified-authority capability is forgeable and copyable

`VerifiedAuthority` is a public dataclass and `_AUTHORITY_MARKER` is an
importable module global. `require_verified_authority()` checks only
`value._marker is _AUTHORITY_MARKER` at policy lines 538-541. It has no exact
issued-object registry, original digest, copy denial, source-review revalidation,
or consumption state.

The independent probe directly constructed `VerifiedAuthority(...,
_marker=policy._AUTHORITY_MARKER)` without calling `verify_authority()`.
`require_verified_authority()` accepted it. A `dataclasses.replace()` clone of a
legitimately verified object was also accepted.

This bypasses the claimed ordering in launcher lines 58-73. In particular,
trainer `_run_training()` and `_reserve_attempt()` rely on the same marker-only
check. Calling those importable functions does not rerun the static trigger or
canonical different-agent review preflight.

Both independent authority tests fail because the forged and cloned objects are
accepted.

### 2. One authority can reserve multiple noncanonical attempts

The experiment fixes one canonical output path, but `_reserve_attempt()` accepts
an arbitrary `attempt_path` and does not compare it to
`policy.CANONICAL_ATTEMPT_PATH`. The authority object has no consumed state.

The independent probe used one valid test authority to create complete
reservations under two distinct temporary output roots. The second reservation
succeeded. This violates both the exact-path contract and the single-attempt
license independently of the global-marker forgery.

The replay test fails because no one-use or canonical-path rejection occurs.

### 3. A stale pre-claim staging directory strands the sole attempt

`_reserve_attempt()` uses the fixed sibling
`.n5.reservation-staging` and calls `os.mkdir(staging)` at trainer line 158.
There is no recovery record, instance identity, inspection rule, resume path, or
authorized stale-staging cleanup. The `os.mkdir` also occurs before the guarded
`try` block.

An uncatchable interruption before `os.rename()` can therefore leave staging
while the canonical attempt remains absent. Every later authorized call fails
at `os.mkdir`; `verify_authority(require_unclaimed_output=True)` still considers
the attempt unclaimed, and no canonical reservation or terminal receipt records
what happened.

The independent probe created that exact interrupted state. The next reservation
raised `FileExistsError`, left staging in place, and produced no canonical
attempt. The crash-recovery test fails.

### 4. A post-rename terminal failure does not durably commit the claim

After `os.rename(staging, attempt_path)`, trainer line 175 marks `claimed=True`.
The injected post-claim failure at lines 176-177 occurs before the only
`_fsync_directory(seed_root)` at line 178. The exception path writes and fsyncs
`failed.json` inside the renamed attempt directory, but `_terminate_failure()`
never fsyncs `seed_root`.

Consequently the terminal failed attempt's parent-directory entry is not made
durable on this path. A power loss may erase the rename even though the process
reported a terminal claimed failure. The independent fsync trace contains only
the staging and attempt directories, not the seed root.

The durability test fails because the seed-root fsync is absent.

## Scientific and downstream checks that passed

- The preregistration, structural-trigger amendment, and exact terminal
  invalidation are hash-bound before the ordinary launcher imports the trainer.
- Frozen numerical, model, dataset-contract, target-partition, metric, and gate
  source hashes match the policy map.
- The experiment is exactly fresh seed `20260710`, `N=5`, AdamW, `400` updates,
  all five frames per update, 2,000 exposures, learning rate and weight decay
  `1e-4`, float32, no autocast, global clip `1.0`, and four losses at `0.25`.
- The deterministic schedule hash is
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`;
  every one of the 400 batches is a permutation of all five frames.
- Only the final update is serialized. No best-step, restart, warm-start,
  selection, or calibration surface exists in the successor CLI.
- Matched and cyclic wrong-RGB controls run at evaluation batch size one. The
  aggregate total is rebuilt from the four aggregate components and checked
  against the unchanged `1e-9` structural tolerance.
- All five selected RGB files are rehashed after training/input revalidation and
  before checkpoint serialization. The verifier separately hash-decodes all
  five again.
- The launcher and verifier expose only GPU0 through `HIP_VISIBLE_DEVICES=0`,
  remove `HSA_OVERRIDE_GFX_VERSION`, and set all four native thread variables to
  one. The frozen runtime validator requires exactly one visible device named
  R9700 with at least 16 GiB and rejects Raphael.
- Publication uses exclusive file creation, fsyncs each file, writes completion
  last, and cleans caught post-reservation partial success artifacts before a
  terminal failure receipt. The four blockers above concern authority,
  replay/path scope, pre-rename crash recovery, and the immediate post-rename
  durability gap.
- The verifier exclusively reloads the checkpoint, rehashes every state tensor,
  reconstructs a fresh model, reruns both controls, requires exact result
  equality, and invokes the frozen metric and N5 threshold functions.
- The finalizer recomputes the metric decision and keeps retry, N16, second seed,
  V5, G2, held-out, selection, calibration change, runtime, hardware,
  production, and promotion authority false. A pass can authorize only design
  review for a later-rung successor.

## Verification

All test processes hid HIP, CUDA, and ROCr devices, disabled external pytest
plugins, and capped OMP/OpenBLAS/MKL/NumExpr threads to one. Candidate and
predecessor suites ran concurrently as separate CPU processes.

```text
candidate focused suite:
  17 passed in 0.67s

predecessor closure suite:
  31 passed in 1.97s

independent adversarial suite:
  1 passed, 5 failed in 0.68s
  - exact source/schedule/RGB publication ordering: passed
  - importable-marker authority forgery rejection: failed
  - authority clone rejection: failed
  - one-authority/one-attempt replay rejection: failed
  - stale pre-claim staging recovery: failed
  - post-rename seed-root durability: failed

py_compile: passed
```

The canonical PASS review file and the entire canonical `n5_full_panel_v1`
output root remained absent. No dataset manifest, train RGB, target shard,
checkpoint, model output, accelerator runtime, GPU, G2, held-out, sealed,
selection, calibration, runtime, hardware, navigation, or production input was
opened during this review.

## Required additive successor closure

Preserve all V1 files and this BLOCK evidence. A reviewed additive V2 must:

1. make authority an exact live object issued by the canonical preflight, retain
   its immutable original digest in an issuer-owned registry, reject direct
   construction, clone/replace, mutation, cross-issuer transfer, and replay,
   and revalidate the canonical review/source closure before protected work;
2. consume that exact authority atomically for one canonical attempt path only;
   production reservation code must reject every alternate path, while any
   test-path injection requires a separate explicitly test-only capability;
3. define and test recovery for an interrupted fixed staging directory before
   another attempt is claimed, including an auditable rule for incomplete,
   complete, foreign, and mutated staging content; no stale staging state may
   silently strand or create a second attempt;
4. fsync the seed-root directory immediately after the atomic rename and before
   any fallible post-claim operation, then preserve parent-directory durability
   on every caught terminal-failure path;
5. pass all six independent tests plus the 48 candidate/predecessor tests, and
   receive another different-agent review before any canonical PASS JSON or
   exact execution is created.

The experiment itself must remain unchanged. No exact attempt, retry, later
rung, second seed, V5, G2, held-out, runtime, hardware, navigation, production,
or promotion authority is granted by this BLOCK review.

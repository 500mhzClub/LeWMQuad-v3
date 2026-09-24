# V4 N5 full-panel V3 implementation handoff

Date: 2026-07-13

Implementation author: `/root/v4_full_panel_v3_author`

Status: **AUTHOR COMPLETE, DIFFERENT-AGENT REVIEW REQUIRED**

V3 is an additive execution-lifecycle successor. V1 and V2 source, tests,
handoffs, and BLOCK evidence were not edited. No canonical PASS review JSON or
experiment output was created, and no exact data, RGB, model, checkpoint,
accelerator, protected role, G2, held-out, runtime, hardware, or production
input was opened.

## Parent BLOCK bindings

V3 binds the V2 independent review Markdown at file SHA-256
`24953fc64da151a6ff1f4ad89e5465e1caae300223556702e0f5c8430d47ee04`
and the V2 BLOCK JSON at file SHA-256
`ddca89e467e4cc30e52bacf57b28c040465e712843fde465f472f3cc8b38fc73`
and canonical content SHA-256
`c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a`.
It also binds the V2 author handoff and independent exploit test at
`3056b00f7b5f224c0507f07505c005f4f5ea2171fb97e6f78585cf7f0460bb61`
and
`a53c5e5d351784ff2a4824231998194e15040597897411c91e7727ec73a95e69`.

The retained V1 BLOCK content remains
`99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7`.
Every retained V1/V2 and frozen numerical source is rehashed in static
preflight.

## Additive V3 artifacts

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py` | `b0f5929aadfaeb9a10f2211db21297c7c01d10305e094a249e5ad8f27b8f46d3` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py` | `8a8bec79bbbfdd2554e0625afc3d423ea9ec8e56baf1134f70d334efe357af66` |
| `lewm/tests/n5_full_panel_v3_synthetic_execution.py` | `83af899f8479f6a3e98530da5af2c58b2b0fd25b48e29954ef77db08e5bf5c91` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v3.py` | `730513d7607b02539b58cde883600a28e6d0e3592333a16d5df67ac3e092beee` |

## Authority-free execution boundary

V3 removes the failed abstraction rather than hiding another Python registry:

- there is no authority class, capability class, issuer, lifecycle record,
  token factory, registry, transition API, or caller-held execution object;
- the sole production operation accepts only the different-agent review file
  SHA-256 and an RGB worker count in `[1,5]`;
- the review path, output root, attempt, metric receipt, and gate are constants
  and cannot be injected by a caller;
- the launcher respawns the reviewed file under `python -I -B`, removes Python
  path/startup injection, selects only GPU0, removes the ROCm override, and
  caps every native thread family to one;
- the isolated child owns static/source preflight, the sole atomic claim,
  frozen training, independent metric recomputation, and finalization before
  returning one summary;
- small local identity tokens exist only while adapting byte-frozen V1
  numerical functions. They are never returned or stored in V3 module state,
  and every retained-module binding is restored in `finally`.

Ordinary imports expose no mutable lifecycle record and no reusable
capability. The sole-attempt fact is the durable filesystem claim, not an
in-process mapping.

## Reservation and durability

The V2 filesystem corrections are retained in the authority-free executor:

- a private `0600` flock serializes claimers;
- staging directories have random names, mode `0700`, and are fsynced;
- legacy, V2, incomplete, foreign, mutated, complete, and duplicate staging
  states are inventoried under the lock;
- complete staging is rehashed against the current canonical review and
  refreshed before resume;
- `(st_dev, st_ino)` identifies the directory owned by this claimant;
- `_fsync_directory(seed_root)` immediately follows
  `os.rename(staging, attempt)`;
- only the claimant with the recorded inode may terminalize a post-claim
  failure;
- terminal and successful publication fsync the attempt and parent;
- selected RGB and frozen inputs/sources are rehashed before publication.

Production has no failure-injection or alternative-path parameter. Lifecycle
tests use the separate
`lewm.tests.n5_full_panel_v3_synthetic_execution` operation. Its constructor
rejects the repository, anything inside it, any ancestor containing it, and
the canonical output namespace. It cannot become production-eligible.

## Frozen science and downstream checks

The scientific contract is unchanged: seed `20260710`, fresh N=5 model,
AdamW, 400 updates, batch size five, 2,000 exposures, learning rate and weight
decay `1e-4`, float32, no autocast, clip `1.0`, four equal losses, final update
only, matched and cyclic wrong-RGB batch-one evaluation, GPU0 R9700, and one
native thread per process. The schedule commitment remains
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

The end-to-end operation reuses the frozen numerical trainer only after
preflight, then independently reloads and rehashes the checkpoint, constructs
a fresh model, reopens five RGBs, recomputes both controls and the frozen N5
decision, writes the metric receipt exclusively, validates it again, and
writes the gate exclusively. Checkpoints remain development-only.

## CPU verification

All commands disabled external pytest plugins, capped OMP/MKL/OpenBLAS/NumExpr
threads to one, and hid HIP/CUDA/ROCr devices.

```text
V3 adversarial/source/recovery/durability suite: 19 passed in 0.93s
retained V1 focused and predecessor closure:     48 passed in 1.88s
V2 independent BLOCK reproducer:                 1 passed, 3 failed in 0.04s
py_compile for both production V3 sources:       passed
git diff --check for all V3 artifacts:           passed
```

The three V2 reproducer failures are required parent evidence: closure cells
still expose `production_records` and `test_scopes`, consumed state can be
reset, and reconstructed objects can be registered. V3 contains none of those
objects or APIs.

The V3 suite covers closure/default/global/referent traversal, production path
surface, non-isolated rejection, source-parent rehash, synthetic confinement,
copy/deepcopy/reconstruction/replay, concurrent claim ownership, complete
resume, cross-root transfer rejection, incomplete/foreign/mutated recovery,
preclaim cleanup, postclaim terminal durability, immediate rename/parent-fsync
ordering, import safety, frozen science, and source-review contract shape.

## Required different-agent review

A reviewer other than `/root/v4_full_panel_v3_author` must:

1. rehash this handoff, all four V3 artifacts, every retained V1/V2 source,
   both BLOCK records, both independent exploit tests, and frozen numerical
   dependencies;
2. rerun the V3 suite, retained 48-test closure, V2 BLOCK reproducer, and
   `py_compile` with CPU threads capped and accelerators hidden;
3. traverse V3 functions/classes/defaults/closures/globals/referents for any
   mutable lifecycle record or reusable capability and inspect restoration;
4. falsify path injection, source-review mutation, replay, concurrent claim,
   staging transfer/recovery, inode ownership, and rename/fsync durability;
5. confirm the canonical V3 PASS JSON and output root remain absent;
6. only after every check passes, create the canonical different-agent review
   JSON from `expected_source_review_core(...)` and a separate reviewer report.

Until that review passes, the canonical review file is absent and exact
execution fails closed. This handoff grants no exact run, retry, N16, second
seed, V5, G2, held-out, selection, calibration, runtime, hardware, navigation,
production, or promotion authority.

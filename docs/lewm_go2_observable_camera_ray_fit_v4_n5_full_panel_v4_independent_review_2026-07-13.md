# V4 N5 full-panel V4 independent review

Date: 2026-07-13

Reviewer: `/root/coordinator_v2_qa`

Verdict: **BLOCK**

V4 closes the eight frozen V3 findings, preserves the numerical experiment,
and keeps its claimed directory descriptor live through the normal training,
verification, and finalization sequence. Three independent checks remain
inconsistent with the documented end-to-end boundary.

## Blocking findings

### 1. Verification and finalization failures are not terminalized

The retained trainer is wired to `_publish_success` before it returns from
`_run_frozen_training`. The subsequent verification/finalization block in
`execute_exact` has a `finally` clause that closes the claim descriptor, but it
has no exception handler and no call to `_terminate_failure`.

Consequently, a verifier failure can leave the sole attempt claimed with
`checkpoint.pt`, `result.json`, and `completed.json` but no metric or gate. A
finalizer failure can additionally leave the metric receipt. The attempt is
single-use and declares retry false, so this is not a recoverable intermediate
state. The independent AST check fails because the post-training lifecycle has
no terminal failure branch.

Relevant implementation locations:

- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py:1083`
- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py:1298`
- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py:1315`

### 2. The repository root is not opened component-by-component

`read_regular_bytes_at` opens the supplied root as one absolute pathname with
`O_NOFOLLOW`, then begins descriptor-relative traversal below it. `O_NOFOLLOW`
protects only the final component of that absolute open; it does not bind the
root's ancestors.

The independent temporary-path check moved a root ancestor, installed a
symlink to the moved tree, performed the actual absolute root open, and restored
the original entry before the call returned. V4 accepted the substitution and
made two payload reads. This contradicts the handoff statement that the
repository root itself is acquired through a component-wise no-follow walk.

Relevant implementation location:

- `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py:380`

### 3. Claimed-path ancestry can become aliased during publication

The retained claim descriptor and device/inode identity remain valid. However,
`_assert_owned_claim` checks the canonical name with one absolute `os.stat(...,
follow_symlinks=False)`, which does not reject symlinks in ancestor components.

The independent temporary synthetic check moved the output ancestor and
installed a symlink back to the same tree after the claim. Publication was
accepted and `completed.json` was written. The production check uses the same
absolute-stat pattern, so the claimed directory remains identity-bound but is
not proven to remain at an unaliased canonical path.

Relevant implementation location:

- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py:118`

## Frozen candidate

The candidate and author artifacts were rehashed and were not edited.

| Artifact | SHA-256 |
|---|---|
| V4 policy | `ff291b94b1546ae9ccf0b85de5f96b87edce4ad5b7992ca16bbbf13dcd1d4485` |
| V4 executor | `19cbdc5692911b31b3b44883b0cfefcc81daa4afc16250b89c1317dd9b66afe4` |
| V4 synthetic lifecycle | `01e49c303d0e2c8e76e7ecbdbd2d0cf159948a5f36a4dc6248d0e014d9c69fb5` |
| V4 author tests | `299fd18b88a869916a916adc4e8848235e955447e9a1f245aeaeec6e7ee69688` |
| V4 author handoff | `4e0aa7e2efa266feb774a4b095cbddca105cfd046aac7a0da7f942f1b2b6925e` |
| independent V4 tests | `2942b23215f506fa9893013d377f5bb4ce4b2327083a1806be4746bfdae56e9f` |

The frozen V1, V2, and V3 BLOCK content identities remain
`99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7`,
`c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a`,
and `d84152d611631364e4c52114a753c36fdabd1cf69d5508d4cb25b5b93dd67f2f`.

## Passing evidence

Independent checks confirmed:

- ordinary import exposes no production-defined lifecycle function, stage
  writer, reservation class, or stage value;
- all eight V3 mismatch cases pass in V4: no constructible reservation or
  writer surface, replacement claim rejection, canonical review-leaf alias
  rejection, and source-parent replacement rejection;
- sub-root source components use descriptor-relative
  `O_DIRECTORY | O_NOFOLLOW`, leaves are no-follow singly-linked regular files,
  and opened descriptors receive stable before/after fingerprint checks;
- the claimed directory descriptor is opened before rename, retained through
  normal publication, and closed after finalization;
- recovery, concurrency, source/input/RGB rehash, retained-module restoration,
  and exact metric/gate recomputation ordering remain present;
- the isolated launcher retains `-I -B`, GPU0-only visibility, no ROCm
  override, and one native thread per family; and
- the frozen experiment remains 400 updates, 2,000 exposures, full N=5 panels,
  schedule SHA-256
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
  and CPU-smoke total loss `0.265`.

The frozen V3 reproducer remains `7 passed, 8 failed`, preserving its BLOCK
evidence. Those eight V3 failures have passing V4 counterparts and are not the
reason for this verdict.

## Verification

All pytest commands disabled external plugins, fixed OMP, MKL, OpenBLAS, and
NumExpr threads to one, and hid HIP, CUDA, ROCr, and HSA devices. The CPU smoke
used the project interpreter because isolated system Python does not contain
Torch; it performed no optimization or accelerator work.

```text
independent V4 conformance:  14 passed, 3 failed in 0.73s
focused V3 closures in V4:    8 passed in 0.07s
V4 author suite:             23 passed in 0.78s
combined retained closure:   90 passed in 2.29s
frozen V3 reproducer:         7 passed, 8 failed in 0.70s
V4 isolated CPU smoke:       PASS
py_compile:                  PASS
git diff --check:            no whitespace errors
```

The canonical V4 PASS review JSON and canonical output root remain absent.
Exact optimization was not run. No dataset, RGB, checkpoint, protected role,
G2, held-out, selection, calibration, runtime, hardware, navigation,
production, or promotion payload was opened.

## Required successor

An additive successor must preserve V1-V4 and all frozen evidence while:

1. acquiring the repository root from the filesystem root through a
   descriptor-relative no-follow walk and retaining/rechecking the complete
   root-component identities through each source/review read;
2. validating the claimed attempt against retained no-follow parent-component
   descriptors, not an absolute stat that can traverse a new ancestor alias;
3. catching verification and finalization failures while the claim descriptor
   is live, removing or invalidating partial success/metric/gate artifacts, and
   durably publishing one terminal failure receipt; and
4. adding passing counterparts for all three independent failures before a new
   different-agent review.

Until then, source closure is not approved and exact attempt, retry, later
rung, V5, G2, held-out, runtime, hardware, navigation, production, and
promotion authority remain false.

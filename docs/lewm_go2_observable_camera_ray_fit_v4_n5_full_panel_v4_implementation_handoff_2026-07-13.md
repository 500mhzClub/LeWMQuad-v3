# V4 N5 full-panel V4 implementation handoff

Date: 2026-07-13

Implementation author: `/root/raw_plan_v2_qa`

Status: **AUTHOR COMPLETE, DIFFERENT-AGENT REVIEW REQUIRED**

V4 is an additive source-closure successor to the frozen V3 BLOCK. V1, V2,
and V3 source, tests, handoffs, reviews, and BLOCK evidence were not edited.
No canonical V4 PASS review JSON or experiment output was created. Exact
optimization was not run, and no dataset, RGB, model, checkpoint, accelerator,
protected role, G2, held-out, runtime, hardware, navigation, production, or
promotion payload was opened.

## Parent V3 BLOCK bindings

V4 binds the V3 implementation handoff at file SHA-256
`c97b3f761955fb6d73469c53632c27388626ae75b010c317fe64b860f76bf8db`,
the independent review at
`d28eadce56668b0cf793806bb98e7c793eb9d874b7ca818d4d9b3c3205fe53e7`,
and the machine-readable BLOCK at file SHA-256
`d1f859aea2a80f090c3ee09df5194f5b4bcfca22865f323de543f3b216b3e168`
and canonical content SHA-256
`d84152d611631364e4c52114a753c36fdabd1cf69d5508d4cb25b5b93dd67f2f`.
It also binds all five frozen V3 implementation/test artifacts, including the
independent adversarial test at
`b7d3669135f22311e13c840e04c4ec2ed583365fc77f7fce6c5c0ecc4e512395`.

The V3 review established four blocking defects:

1. importable reservation and writer surfaces accepted caller-created stage
   values and could publish completion, metric, and gate artifacts;
2. the claimed directory identity was not retained after reservation, so a
   replacement pathname could receive later publication;
3. resolving both review paths allowed a symlink to masquerade as the
   canonical review entry; and
4. leaf-only no-follow source reads did not retain parent-directory identity.

The frozen V1, V2, and V3 BLOCK content identities remain
`99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7`,
`c4d93bbac0c849a2add12bb0ab69609cef0c58a6e203a02d6b806b3c7a41fd8a`,
and
`d84152d611631364e4c52114a753c36fdabd1cf69d5508d4cb25b5b93dd67f2f`.
Static preflight rehashes all retained parents and frozen numerical sources.

## Additive V4 artifacts

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py` | `ff291b94b1546ae9ccf0b85de5f96b87edce4ad5b7992ca16bbbf13dcd1d4485` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py` | `19cbdc5692911b31b3b44883b0cfefcc81daa4afc16250b89c1317dd9b66afe4` |
| `lewm/tests/n5_full_panel_v4_synthetic_execution.py` | `01e49c303d0e2c8e76e7ecbdbd2d0cf159948a5f36a4dc6248d0e014d9c69fb5` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v4.py` | `299fd18b88a869916a916adc4e8848235e955447e9a1f245aeaeec6e7ee69688` |

## Single-operation execution boundary

The production executor defines no module-level function or class. Ordinary
import performs no lifecycle operation and exposes no V4 reservation,
completion, failure, metric, gate, stage writer, `main`, or `execute_exact`
surface. All lifecycle definitions and all stage values exist only inside the
canonical file's `__main__` branch, which synchronously dispatches the complete
operation. They cannot be imported and called with caller-created mappings.

The non-isolated entry respawns that fixed file under `python -I -B`, removes
Python path/startup injection, exposes only GPU0, removes the ROCm override,
and caps each native thread family to one. The isolated child owns static and
different-agent source preflight, the one durable claim, frozen training,
independent checkpoint recomputation, finalization, and terminal publication.
There is no production failure injection or alternative output path.

## Descriptor-bound claim and publication

V4 opens the complete staging directory before its atomic rename and retains
that one descriptor plus its `(st_dev, st_ino)` identity in the internal
reservation through training, failure terminalization, success publication,
verification, and finalization. Immediately after rename it fsyncs the seed
parent and verifies that the canonical pathname still names the open
directory. The descriptor is closed only in the end-to-end operation's final
cleanup.

Every later attempt read or write is descriptor-relative, no-follow, and
guarded by an identity check between the retained descriptor and canonical
directory entry. Claim files must be singly-linked regular files. Replacing or
moving the canonical attempt directory therefore fails before any artifact is
written into the replacement.

The existing private flock, random `0700` staging directories, `0600` files,
complete-staging recovery, single-use claim, immediate rename/parent-fsync,
terminal durability, and post-training input/source/RGB rehash remain in
place. Synthetic crash and concurrency testing remains in the separate,
production-ineligible V4 synthetic operation.

## Component-safe source review

The policy opens the repository root and walks every review/source parent one
component at a time with descriptor-relative `O_DIRECTORY|O_NOFOLLOW` opens.
It retains every descriptor until the leaf read finishes, requires a
singly-linked regular leaf opened with `O_NOFOLLOW`, and compares complete
stable fingerprints before and after the read for the leaf and every retained
parent. The fingerprint includes device, inode, mode, link count, owner,
group, size, modification time, and change time.

The supplied source-review path must lexically equal the canonical repository
entry; neither side is resolved through an alias. Symlink leaves, symlink
parents, hard-linked leaves, FIFOs, component changes, and parent identity
replacement therefore fail closed.

## Frozen science and downstream checks

The scientific contract is unchanged: seed `20260710`, fresh N=5 model,
AdamW, 400 updates, batch size five, 2,000 exposures, learning rate and weight
decay `1e-4`, float32, no autocast, clip `1.0`, four equal losses, final update
only, matched and cyclic wrong-RGB batch-one evaluation, GPU0 R9700, and one
native thread per process. Every update remains a full five-frame panel. The
schedule commitment remains
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

The frozen trainer is still followed by source/input/RGB rehash, independent
checkpoint reload into a fresh model, independent control recomputation,
exclusive metric receipt creation, validation, and exclusive gate creation.
The selected RGB decode work uses the reviewed process pool and inherits the
one-thread environment. Checkpoints remain development-only.

## CPU verification

All commands disabled external pytest plugins, capped OMP, MKL, OpenBLAS, and
NumExpr threads to one, and hid HIP, CUDA, ROCr, and HSA devices.

```text
V4 author/adversarial/source/recovery suite:       23 passed in 0.76s
V4 + V3 author + retained numerical closure:      90 passed in 2.32s
frozen V3 independent BLOCK reproducer:            7 passed, 8 failed in 0.68s
V4 isolated CPU contract smoke:                    PASS
py_compile for all four additive V4 artifacts:     PASS
git diff --check for all four additive artifacts:  PASS
```

The eight expected V3 failures cover the importable stage-evidence class,
reservation copying/reconstruction/mutation, completion writer, metric and
gate writers, claimed-directory replacement, canonical review alias, and
source-parent replacement. The V4 author suite contains passing counterparts
for all eight cases, plus regular canonical-review acceptance, no-follow leaf
type checks, static-parent rehash, synthetic confinement/recovery/concurrency,
claim durability, frozen science, GPU0 launcher, and review-contract shape.

The CPU smoke reproduced 400 updates, 2,000 exposures, full panels, schedule
SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
and total frozen synthetic loss `0.265`. It did not perform optimization or
open experiment inputs.

## Required different-agent review

A reviewer other than `/root/raw_plan_v2_qa` must:

1. rehash this handoff, all four V4 artifacts, every retained V1-V3 artifact,
   all three BLOCK records, independent exploit tests, and frozen numerical
   dependencies;
2. rerun the V4 suite, 90-test retained closure, frozen V3 BLOCK reproducer,
   CPU contract smoke, `py_compile`, and `git diff --check` with CPU threads
   capped and accelerators hidden;
3. independently inspect the import namespace and AST to prove that no stage
   class, writer, reservation, transition, or partial execution operation is
   importable and that all published values originate inside one script entry;
4. falsify review/source symlink, hardlink, FIFO, parent-swap, and component
   replacement attacks, including before/during/after descriptor walks;
5. falsify claim-path replacement and directory transfer at reservation,
   training failure, success publication, metric verification, and gate
   finalization boundaries, and confirm descriptor closure;
6. confirm the canonical V4 review JSON and canonical output root remain
   absent; and
7. only after every check passes, create the canonical different-agent review
   JSON from `expected_source_review_core(...)` and a separate reviewer report.

Until that review passes, exact execution fails closed. This handoff grants no
exact attempt, retry, N16, second seed, V5, G2, held-out, selection,
calibration, runtime, hardware, navigation, production, or promotion
authority.

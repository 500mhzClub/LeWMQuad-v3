# V4 N5 full-panel V6 lifecycle-recovery implementation handoff

Date: 2026-07-13

Implementation author: `/root/camera_v5_independent`

Status: **AUTHOR COMPLETE, DIFFERENT-AGENT REVIEW REQUIRED**

V6 is an additive infrastructure replacement for the consumed, terminal V5
attempt. It is not a scientific or numeric retry. V1 through V5 source,
reviews, receipts, and evidence were not edited. No canonical V6 PASS review or
V6 experiment output was created. Exact optimization was not run. No V5 numeric
result or checkpoint survived or was inspected, and no dataset, RGB, model,
checkpoint, protected-role, G2, held-out, runtime, hardware, navigation,
production, or promotion payload was opened. All author verification was
CPU-only with accelerators hidden.

## Frozen recovery authority

The pre-implementation V6 lifecycle-recovery amendment is frozen at:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_lifecycle_recovery_amendment_2026-07-13.md` | `1fa4279c604b1a8be825e082a367a5404381154fe1784394e43aee35924caa90` |

It authorizes source construction and independent review, but not execution.
It freezes the V1/V5 experiment exactly: seed `20260710`, N=5, fresh model
initialization, 400 updates, batch size 5, 2,000 frame exposures, four
equally-weighted losses, final-update checkpoint selection, matched and cyclic
wrong-RGB evaluation, GPU0 on the AMD Radeon AI PRO R9700, and schedule
SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

The only admitted V5 exact-attempt payloads are these terminal lifecycle
receipts:

| Receipt | File SHA-256 | Content SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5/reservation.json` | `f8062f2ed2bdb1589ca806fb9331ce7f1ec0675d4466e96c0a78530080ea501a` | `1427a5524cbc7e72ac24d78c221775bab3c943d36967b88df6e780743faafc15` | 4532 |
| `.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1/attempts/seed_20260710/n5/failed.json` | `7ead760085f5365ac83ebfc8875910cbc076437fa972d48d008aa3b2127e50af` | `84cfa81aa2db9fa7cd7233e314e7d3da50b4fc23af863ab38e9ab948ac51358b` | 802 |

The V6 static preflight validates those exact hashes, byte counts, schemas,
terminal status, no-retry state, failure binding, and the exact two-file V5
attempt inventory. It rejects any V5 numeric payload. V6 uses no V5 model state
or training observation.

## Additive V6 artifacts

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `75b987dc97c21e2689caea8df4fb316a80b6602cf8a612e47abe02bf14a5d549` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `791103400c6093c40abed5c87009d4a18feceda1c5155c2d06dae97b2bb38a3d` |
| `lewm/tests/n5_full_panel_v6_synthetic_execution.py` | `8df835debcc24f7fd1b77f5cc0f559215023c9111d3c2ff5ae367129296a496f` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `2af8b43439ce2b72cc9c22cd1a3d48028c66e3b18cd2b2b742ddf0b147ce017b` |

These hashes are the author freeze. Any source or test change requires new
hashes and a new review binding.

## Lifecycle replacement

The V5 failure was caused by an unrelated retained test creating and removing
a direct child under the shared repository `.generated` ancestor. V5 bound
that shared directory's size and timestamps, so the legitimate concurrent
churn invalidated its ancestry check before the first result publication. No
V5 numeric result, checkpoint, or completion was published.

V6 keeps the component-wise no-follow descriptor chain while splitting the
ancestry predicate by ownership:

- shared ancestors above the canonical V6 output root bind device, inode,
  type/mode, owner, and group, while tolerating link-count, size, mtime, and
  ctime changes caused by unrelated direct children;
- the exclusive V6 output root and every descendant bind the full stable
  fingerprint, including link count, size, mtime, and ctime;
- symlink, rename/alias, inode replacement, type, permission, owner, and group
  changes remain fatal throughout the chain; and
- full metadata is refreshed only after an explicit executor-owned mutation.

The fresh canonical namespace is:

`.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v6`

V6 retains the V5 private staging, process-death recovery, atomic no-replace
claim, parent fsync, isolated canonical-only entry, exclusive publication,
self-hashed receipts, owned-artifact cleanup, and durable no-retry behavior.
The production module exposes no importable lifecycle operation; all executor
definitions remain under the script's `__main__` branch.

## Claimed-directory closure

V6 additionally binds the claimed directory itself with a full fingerprint and
descriptor-relative exact inventory. Its expected inventory is
`reservation.json` plus the executor's registered owned claim artifacts.
`_refresh_claim_directory` is permitted only after an exact owned file
mutation. A foreign child, unexpected deletion, claim replacement, or
create/delete history therefore blocks success publication even if the final
name set is restored.

Failure terminalization deliberately checks retained claim identity rather
than requiring the success predicate. This permits `failed.json` publication
after an ancestry or claim-inventory violation while preserving foreign or
changed artifacts. The attempt remains consumed and retry authorization
remains false.

## Frozen execution contract

The V6 authority scope is
`one_exclusive_fresh_infrastructure_replacement_attempt`. The exact entry
rejects an existing canonical attempt, while allowing only the frozen
pre-claim process-death recovery path in the new namespace. It grants no
scientific retry, caller path or stage authority, V5 numeric-payload read,
second V6 attempt, N16, second seed, later-model training, G2, holdout,
selection, calibration change, runtime, hardware, navigation, production, or
promotion authority.

## Author verification

Commands disabled external pytest plugins, capped OMP, MKL, OpenBLAS, and
NumExpr threads to one, and hid HIP, CUDA, ROCr, and ordinal device variables.

```text
V6 author/adversarial/source/lifecycle suite:       40 passed in 1.37s
V1-V5 applicable retained author closure:          103 passed, 8 deselected
V6 isolated CPU contract smoke:                    PASS
py_compile for all four additive V6 artifacts:     PASS
canonical V6 output root:                          ABSENT
canonical V6 source-review JSON:                   ABSENT
exact execution:                                   NOT RUN
```

The eight deselected retained assertions require V5 review and output state to
be absent. They are obsolete historical-state assertions now that V5 has a
different-agent PASS review and terminal exact lifecycle receipts; no security,
science, schedule, or lifecycle assertion was deselected. The applicable
retained matrix completed as 103 passes in 2.13 seconds.

The isolated CPU smoke reproduced 400 updates, 2,000 exposures, complete
five-frame panels, schedule SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
and frozen synthetic total loss `0.265`. It did not optimize a production model
or open experiment inputs.

The V6 suite covers direct create/delete churn under the literal shared
`.generated` ancestor and the nested production chain, persistent aliases and
rename swaps, shared permission mutation, exclusive-root and claim-directory
foreign children, claim replacement, identity-only terminalization, foreign
artifact preservation, pre-claim process-death recovery, staging cleanup,
durable no-retry behavior, source-reader regressions, import safety, frozen
GPU/science/schedule bindings, and transformed reservation validation against
the retained V1 contract.

## Required different-agent review

A reviewer other than `/root/camera_v5_independent` must:

1. rehash this handoff, the recovery amendment, all four V6 artifacts, all
   retained V1-V5 bindings, the V5 PASS review, and both V5 terminal receipts;
2. rerun the 40-test V6 suite, the applicable retained author closure, isolated
   CPU contract smoke, and compilation with threads capped and accelerators
   hidden;
3. independently verify that ordinary imports expose no lifecycle operation
   and that only the canonical isolated script can claim the fresh namespace;
4. attack shared ancestry with direct child churn, aliases, rename/restoration,
   inode replacement, symlinks, and permission/owner/group changes, proving
   only unrelated metadata churn is tolerated;
5. attack the exclusive output chain and claim directory with foreign children,
   create/delete history, replacement, metadata changes, and restored names at
   reservation, training publication, metric verification, and finalization;
6. inject process death and verification/finalization failures, proving only a
   valid frozen pre-claim staging record can recover, exact owned partials are
   removed, foreign/changed artifacts are preserved, `failed.json` is durable,
   and no retry is possible;
7. confirm the V5 attempt inventory contains only the bound reservation and
   failure receipts and that no V5 numeric payload is admitted;
8. confirm the canonical V6 review JSON and output root remain absent; and
9. only after every check passes, publish the canonical different-agent source
   review JSON at
   `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_independent_review_2026-07-13.json`
   from the exact `expected_source_review_core(...)` contract, with a separate
   reviewer report.

Until that review passes, exact execution fails closed. This handoff grants no
execution itself and authorizes no exact run during implementation or review.

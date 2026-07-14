# V4 N5 full-panel V6 independent review

Date: 2026-07-13

Reviewer: `/root/raw_auditor_author`

Verdict: **BLOCK**

The frozen V6 candidate preserves the V1 experiment and its ordinary owned
write paths work, but its two directory-fingerprint refresh operations cannot
prove that the observed metadata change was caused only by the executor-owned
write. A foreign child can be created and removed between the previous
fingerprint and refresh; V6 accepts the restored final inventory and replaces
the old fingerprint with the changed one. This directly contradicts the V6
handoff's claim that create/delete history blocks success even when names are
restored.

The candidate files were not modified. Review work was CPU-only and isolated
under pytest temporary directories. It did not run exact execution, training,
optimization, or GPU work and did not open experiment data, RGB, checkpoints,
protected roles, G2, held-out, runtime, hardware, navigation, production, or
promotion payloads.

## Blocking finding

### Refresh absorbs interleaved unowned create/delete history

`_refresh_claim_directory` verifies only the final expected inventory and the
directory identity/security fields, then unconditionally replaces
`reservation.directory_fingerprint` with the current full fingerprint. It has
no evidence that the metadata delta between the previous and current
fingerprints came exclusively from `_write_claim_file_exclusive`.

`_refresh_directory_chain` has the same problem for a mutable exclusive
directory descriptor. After checking the final named/open identity, it assigns
the current full fingerprint to the retained chain entry without proving the
history or exact delta that produced it.

Relevant implementation locations:

- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py:154`
  (`_refresh_directory_chain`);
- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py:444`
  (`_refresh_claim_directory`); and
- `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py:491`
  (`_write_claim_file_exclusive`, which calls claim refresh after commit).

The distinct reviewer regression inserted and removed an unrelated temporary
entry immediately before each real frozen refresh call. Both paths accepted
the interleaving. This is not a persistent-inventory alias or a shared-parent
false positive: the same suite proves a normal owned claim write succeeds and
unrelated create/delete churn in the intentionally shared ancestor succeeds.

## Frozen candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `75b987dc97c21e2689caea8df4fb316a80b6602cf8a612e47abe02bf14a5d549` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `791103400c6093c40abed5c87009d4a18feceda1c5155c2d06dae97b2bb38a3d` |
| `lewm/tests/n5_full_panel_v6_synthetic_execution.py` | `8df835debcc24f7fd1b77f5cc0f559215023c9111d3c2ff5ae367129296a496f` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `2af8b43439ce2b72cc9c22cd1a3d48028c66e3b18cd2b2b742ddf0b147ce017b` |
| author handoff | `4ca14a5d8392d88c4d9779d82ef4eb3f1655317ed61c8e51490651877e3e57e1` |
| lifecycle amendment | `1fa4279c604b1a8be825e082a367a5404381154fe1784394e43aee35924caa90` |
| distinct reviewer regression | `bd2379d7aab8e20be2d87ac857b1086da5aa6e6d9efa58f2ea3cd3095d406e51` |

All candidate and authority hashes reproduced exactly during review.

## Verification

The author suite was rerun with external pytest plugins disabled, native math
threads capped at one, accelerator visibility empty, and Torch supplied only
from the existing project environment. It completed as 40 passes.

The distinct regression suite contains one frozen-candidate hash check, two
positive controls, and the two expected-rejection cases. Its result is three
passes and two failures: the only failures are the claim-refresh and
derived-refresh checks because neither raised `PermissionError`.

An independent standalone `TemporaryDirectory` reproducer recorded:

```text
claim_interleaving=MISSED
derived_interleaving=MISSED
normal_owned_write=PASS
shared_parent_churn=PASS
```

```text
V6 author focused suite:                         40 passed in 1.34s
distinct directory-consistency regression:       3 passed, 2 failed in 0.18s
py_compile for distinct reviewer regression:     PASS
git diff --check for review artifacts:           PASS
exact execution or training:                     NOT RUN
GPU opened:                                      NO
```

The canonical V6 PASS review JSON and canonical V6 experiment output remain
unauthorized. This BLOCK review creates no PASS authority.

## Required successor

An additive successor must preserve the frozen numerical experiment and prior
evidence while making refresh contingent on a verifiable owned-mutation
transaction. The refresh must prove the exact expected before/after mutation
or otherwise retain event provenance strong enough to reject any interleaved
unowned create/delete, even when final names and identities are restored.

Passing counterparts are required for both frozen blocker tests. The normal
owned-write and shared-ancestor positive controls must continue to pass. Until
then, source closure, exact execution, retry, training, checkpoint use,
selection, G2, held-out, runtime, hardware, navigation, production, and
promotion authority remain false.

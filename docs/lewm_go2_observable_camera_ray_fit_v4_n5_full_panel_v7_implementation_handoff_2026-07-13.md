# V4 N5 full-panel V7 owned-directory transaction implementation handoff

Date: 2026-07-13

Implementation author: `/root/camera_v5_independent`

Status: **AUTHOR COMPLETE, DIFFERENT-AGENT REVIEW REQUIRED**

V7 is the sole additive infrastructure successor to the terminal V6 BLOCK. It
does not authorize or perform a scientific retry. No exact optimization was
run, no GPU was opened, and no dataset, RGB, model, checkpoint, protected-role,
G2, held-out, runtime, hardware, navigation, production, or promotion payload
was opened. The canonical V7 output and canonical V7 PASS review remain absent.

## Frozen authority

The pre-implementation V7 amendment is frozen at:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v7_owned_directory_transaction_amendment_2026-07-13.md` | `17ca6b726d1eaa25662a1823b4c153d496f1e51502b764350ddd6a3a34f249da` |

It binds this immutable V6 closure:

| Artifact | SHA-256 |
| --- | --- |
| V6 lifecycle amendment | `1fa4279c604b1a8be825e082a367a5404381154fe1784394e43aee35924caa90` |
| V6 policy | `75b987dc97c21e2689caea8df4fb316a80b6602cf8a612e47abe02bf14a5d549` |
| V6 executor | `791103400c6093c40abed5c87009d4a18feceda1c5155c2d06dae97b2bb38a3d` |
| V6 synthetic lifecycle | `8df835debcc24f7fd1b77f5cc0f559215023c9111d3c2ff5ae367129296a496f` |
| V6 author tests | `2af8b43439ce2b72cc9c22cd1a3d48028c66e3b18cd2b2b742ddf0b147ce017b` |
| V6 author handoff | `4ca14a5d8392d88c4d9779d82ef4eb3f1655317ed61c8e51490651877e3e57e1` |
| V6 directory-consistency blocker test | `bd2379d7aab8e20be2d87ac857b1086da5aa6e6d9efa58f2ea3cd3095d406e51` |
| V6 independent BLOCK report | `c1ac98c38f19d6b141ff6306956317cb08914c5be22606a86de03fe0439d4692` |
| V6 independent BLOCK JSON | `ff1becd9d5b1173cc43f898d9982b1327fbcf87eb385e66b5f16f20cb3674d1b` |

The V6 BLOCK JSON has canonical content SHA-256
`98260f2b1af7845af6cf1312698b7a5c0d6a0579705f4ff522801eaa02d41fb1`.
V7 does not edit any retained V1-V6 source, review, receipt, or evidence.

## Frozen V7 candidate

| Artifact | SHA-256 |
| --- | --- |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v7.py` | `ed50a00c0449c41031f076c5627f6501b93ee2931deaf4cbcd06a0f9e89d16e0` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v7.py` | `5043d42aaabb5a4852e9339a7d3e98c9d530c7ff403e5a2f1ac7a21999fbc14e` |
| `lewm/tests/n5_full_panel_v7_synthetic_execution.py` | `9743786550ede91023b3d96cfa6650c04bd02a2c1a5d3fbb2364728b09980bf1` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v7.py` | `0bf0f77ff5c773891ddd6ab5ed933b74132f0c8194e0aa237d93175619b7a858` |

These four hashes are the author freeze. Any source or test change invalidates
this handoff and requires new hashes and a fresh review.

## Transaction implementation

Every mutation at or below the exclusive V7 root is represented by a closed,
operation-scoped transaction over retained no-follow directory descriptors.
A dedicated nonblocking, close-on-exec inotify instance records provenance.
Each operation drains prior events, snapshots all watched directories, drains
snapshot races, performs one declared mutation, captures the exact post-state,
matches the exact event sequence and move cookie, proves the declared state
delta, and commits only that captured post-state.

There is no generic refresh path, mutable descriptor allow-list, `shutil`, or
recursive path deletion. Queue loss, unknown watches, generations, names,
masks, order, cookies, self-events, unmount, and unexpected changes poison the
journal permanently. The frozen V6 claim and derived create/delete blockers now
reject, while ordinary owned writes and unrelated shared-ancestor churn pass.

The lifecycle covers exclusive scaffold and lock creation, staging recovery,
reservation and manifest writes, staging-manifest unlink, no-replace atomic
claim, all claim and derived files, owned cleanup, failure receipts, and
descriptor closure. Artifact registries are populated at commit before any
fallible post-commit verification. Post-training exceptions terminalize while
the retained claim and parent descriptors are still live. A poisoned failure
path cannot return a successful binding.

Canonical staging publication uses Linux `renameat2(RENAME_NOREPLACE)`. The
source name is bound to the retained child identity and full fingerprint before
the syscall. A racing foreign destination is preserved and the claim fails.

## Recovery grammar

The fresh V7 namespace never repairs or adopts a historical partial scaffold.
An existing root must already contain the exact private output, attempts, seed,
metric, and gate directories, an empty private lock, and at least one complete
staging. Missing scaffold, zero staging, foreign inventory, unsafe lock, unsafe
leaf, oversized leaf, or incomplete/mutated staging is preserved and blocks the
claim.

Only the exact producer grammar is recoverable:

- `.n5.reservation-v7-` followed by exactly 32 lowercase hexadecimal digits;
- one private `0600` `reservation.json` and one private `0600` `staging.json`;
- each regular leaf is singly linked, current-UID/GID owned, and at most 1 MiB;
- canonical self-hashed JSON, exact attempt path, current recovery policy, and
  the complete frozen authoritative reservation core; and
- no predecessor namespace or `claim.json` alias.

One complete candidate is rehashed and resumed. Multiple complete candidates
are compared after discarding only the prior recovery ledger and its derived
hash. If all authoritative fields match, the lexical first is retained and
only the proven authority-equivalent duplicates are removed with recorded
evidence. Any conflicting authoritative state cannot classify as complete and
is preserved invalid.

## Destructive syscall boundary

Linux provides a true no-replace primitive for the claim rename, which V7 uses.
It does not provide an unprivileged inode-conditional `unlinkat` or `rmdirat`.
The unlink/rmdir transactions therefore bind the full source fingerprint at
the last userspace boundary and require the exact kernel event/post-state after
the syscall. Same-UID replacement at that final boundary poisons success and is
covered by adversarial tests, but the kernel cannot guarantee preservation of
the raced replacement object. Review must treat this explicit platform limit
separately from claim publication, where foreign-destination preservation is
absolute through `RENAME_NOREPLACE`.

## Frozen science

V7 changes only the lifecycle namespace and transaction implementation. It
retains seed `20260710`, N=5, the same five training frames, fresh model
initialization, AdamW at `1e-4` learning rate and `1e-4` weight decay, 400
updates, batch size 5, 2,000 frame exposures, float32 without autocast,
gradient clipping at `1.0`, four equally weighted losses, final-update-only
selection, unchanged matched/wrong-RGB evaluations, unchanged verifier/gate,
and schedule SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`.

An exact run, if separately authorized after review, is GPU0-only on the AMD
Radeon AI PRO R9700. The Raphael iGPU remains forbidden. Author verification
used CPU only and hid every accelerator.

## Author verification

External pytest plugins were disabled, native math threads were capped at one,
and HIP, CUDA, ROCr, and ordinal visibility were empty.

```text
V7 author/adversarial/source/lifecycle suite:       102 passed in 5.23s
Independent recovery/startup cross-check:           102 passed, no blocker
Independent policy/science cross-check:              102 passed, no blocker
Retained V6 author suite:                             40 passed in 1.31s
Applicable retained V1-V5 closure:                   103 passed, 8 deselected
V7 isolated CPU contract smoke:                      PASS
V7 static authority rehash:                          PASS
py_compile for all four V7 artifacts:                PASS
static forbidden-refresh/cleanup scan:               PASS
git diff --check for V7 artifacts:                   PASS
canonical V7 output root:                            ABSENT
canonical V7 source-review JSON:                     ABSENT
exact execution or optimization:                     NOT RUN
production payload or GPU access:                    NONE
```

The eight retained deselections are historical assertions that the consumed V5
review/output do not exist. The same eight were documented by V6; no security,
science, schedule, lifecycle, or numerical assertion was deselected.

The isolated smoke reproduced 400 updates, 2,000 exposures, full five-frame
panels, the frozen schedule hash, and synthetic total loss `0.265`. It did not
train a model or open experiment inputs.

The 102-test V7 suite includes exact transaction positives; frozen V6 blocker
adaptations; six restoration-race hook positions; overflow/watch/cookie/order
failures; pre-watch mutation; shared/exclusive ancestry; no-replace claim
races; syscall-boundary replacement; terminal-receipt poison; post-commit
cleanup; descriptor leaks; process-death recovery; exact recovery grammar;
missing/unsafe scaffolds and locks; insecure/oversized leaves; equivalent and
conflicting complete candidates; import safety; and frozen science/GPU/schedule
bindings.

## Required different-agent review

A reviewer other than `/root/camera_v5_independent` must:

1. rehash the V7 amendment, this handoff, all four frozen V7 artifacts, the
   complete V6 BLOCK closure, retained V1-V5 authority, and the two admitted V5
   terminal lifecycle receipts;
2. rerun the 102-test V7 suite, 40-test V6 suite, applicable retained closure,
   isolated CPU smoke, static authority preflight, and compilation with threads
   capped and accelerators hidden;
3. independently attack every transaction hook, event-loss path, watch
   generation, move cookie, shared/exclusive ancestor, claim, derived write,
   cleanup, failure, and descriptor-closure boundary;
4. independently attack recovery names, manifests, leaf security/size,
   scaffold/lock provenance, complete-candidate equivalence, conflict handling,
   and historical/foreign-byte preservation;
5. confirm that ordinary imports expose no lifecycle operation, only the
   isolated canonical script can claim, and all frozen numerical/GPU bindings
   remain unchanged;
6. confirm the destructive-syscall platform boundary above and the distinct
   absolute no-replace claim guarantee;
7. confirm the canonical V7 output and source-review JSON remain absent and no
   exact execution, payload read, or GPU use occurred; and
8. only after all checks pass, publish the canonical different-agent source
   review JSON from `expected_source_review_core(...)`, plus a separate review
   report that binds this exact handoff and frozen candidate.

Until that review passes, exact execution fails closed. This handoff grants no
execution, retry, checkpoint use, selection, G2, held-out, runtime, navigation,
production, or promotion authority.

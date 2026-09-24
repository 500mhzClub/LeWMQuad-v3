# N5 full-panel V7 owned-directory transaction amendment (2026-07-13)

## Status and scope

This is an additive, pre-implementation amendment to the frozen N5 full-panel
experiment. It authorizes construction and different-agent source review of one
V7 infrastructure-replacement lifecycle. It does not authorize exact
execution. No V7 implementation source existed when this amendment was frozen.

V6 source review is terminal BLOCK. V6 did not run an exact attempt and did not
create its canonical output. V7 is therefore not a numeric or scientific retry
and may not use V5 or V6 numeric state. It is the sole source successor for the
same one fresh infrastructure replacement authorized by the V6 recovery
lineage.

## Frozen V6 lineage

V7 binds the following immutable V6 closure:

| Artifact | SHA-256 |
| --- | --- |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_lifecycle_recovery_amendment_2026-07-13.md` | `1fa4279c604b1a8be825e082a367a5404381154fe1784394e43aee35924caa90` |
| `lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `75b987dc97c21e2689caea8df4fb316a80b6602cf8a612e47abe02bf14a5d549` |
| `scripts/execute_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `791103400c6093c40abed5c87009d4a18feceda1c5155c2d06dae97b2bb38a3d` |
| `lewm/tests/n5_full_panel_v6_synthetic_execution.py` | `8df835debcc24f7fd1b77f5cc0f559215023c9111d3c2ff5ae367129296a496f` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v6.py` | `2af8b43439ce2b72cc9c22cd1a3d48028c66e3b18cd2b2b742ddf0b147ce017b` |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_implementation_handoff_2026-07-13.md` | `4ca14a5d8392d88c4d9779d82ef4eb3f1655317ed61c8e51490651877e3e57e1` |
| `lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_directory_consistency_review.py` | `bd2379d7aab8e20be2d87ac857b1086da5aa6e6d9efa58f2ea3cd3095d406e51` |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_independent_review_2026-07-13.md` | `c1ac98c38f19d6b141ff6306956317cb08914c5be22606a86de03fe0439d4692` |
| `docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_independent_review_block_2026-07-13.json` | `ff1becd9d5b1173cc43f898d9982b1327fbcf87eb385e66b5f16f20cb3674d1b` |

The V6 BLOCK record has canonical content SHA-256
`98260f2b1af7845af6cf1312698b7a5c0d6a0579705f4ff522801eaa02d41fb1`.
Its two decisive failures prove that `_refresh_claim_directory` and
`_refresh_directory_chain` can absorb an unrelated create/delete history when
the final inventory is restored. The V6 positive controls prove ordinary owned
writes and unrelated shared-ancestor churn work.

V6 files and review evidence are immutable. V7 must adapt both frozen blocker
tests to PASS without weakening either positive control.

## Required transaction model

Every V7 mutation at or below its exclusive output root must be one closed,
operation-scoped owned-directory transaction. Directory descriptors opened by
a component-wise `O_NOFOLLOW` walk remain authoritative. A dedicated Linux
inotify instance opened `IN_NONBLOCK|IN_CLOEXEC` supplies event provenance; it
is evidence, not filesystem authority.

Each transaction must:

1. drain all retained watches and reject any prior unexpected event;
2. capture complete descriptor-relative pre-operation inventories, directory
   fingerprints, and leaf fingerprints for every affected exclusive directory;
3. drain again and reject events that raced with the pre-operation snapshot;
4. perform only the declared owned mkdir, exclusive create/write/close,
   rename, unlink, or rmdir sequence;
5. capture complete descriptor-relative post-operation inventories and exact
   committed leaf identities, fingerprints, and payload hashes;
6. drain and require the exact expected watch, name, mask, ordering, and move
   cookie sequence, rejecting every extra create, delete, move, or attribute
   event;
7. require the post-state to equal the exact declared delta from the pre-state;
   and
8. commit the already captured post-operation fingerprints as the new baseline.

The commit must never call `fstat` or inventory again and then accept that later
state. An event after the final drain remains queued for the next boundary and
cannot be folded into the captured baseline. No generic `mutable_fds`,
`refresh_directory_chain`, `refresh_claim_directory`, or equivalent
unproven-refresh API is permitted.

The journal must permanently poison success on queue overflow, unknown watch,
unknown name or mask, move-cookie mismatch, `IN_UNMOUNT`, unexpected
`IN_MOVE_SELF`, unexpected `IN_DELETE_SELF`, or unexpected `IN_IGNORED`.
Deliberate staging-to-claim rename and owned postorder staging deletion may
consume self events only when the same transaction also proves the exact parent
move/delete event, cookie, retained descriptor identity, and full post-state.
Watch-descriptor reuse must be generation-safe.

## Required lifecycle coverage

The closed transaction protocol applies to:

- creation and validation of every exclusive V7 directory and lock leaf;
- new and recovered private staging directories;
- staging reservation and manifest creation or replacement;
- staging manifest deletion;
- the atomic staging-to-canonical-claim rename;
- every claim file, including checkpoint, result, completion, and failure;
- metric-verification and gate parent creation and leaf publication;
- every owned partial cleanup unlink and staging postorder deletion; and
- failure paths before and after claim.

Cleanup must be descriptor-relative and inventory-bound; unqualified
`shutil.rmtree` is forbidden. Missing, changed, linked, replaced, or foreign
artifacts are preserved invalid. Failure terminalization may continue through a
retained claimed-directory identity after journal poison, but must record that
journal integrity failed, may not refresh a success baseline, and may never
restore success eligibility.

All shared ancestors above the V7 exclusive root retain V6's identity and
security binding: device, inode, type/mode, owner, and group changes reject,
while unrelated direct-child link-count, size, mtime, and ctime churn remains
permitted. V7 may not weaken no-follow walks, source revalidation, claim
identity, leaf single-link checks, cleanup ownership, fsync, atomic no-replace
publication, terminal no-retry, or descriptor closure.

## Frozen namespace and science

The new canonical root is:

`.generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v7`

It must be absent before the sole reviewed exact claim. V7 must preserve the
V5/V6 numerical experiment exactly:

- seed `20260710`, fit size N=5, and the same five train frames;
- fresh `ObservableCameraRayEvidenceV4Model` initialization with no V5/V6
  checkpoint or state input;
- AdamW at learning rate `1e-4` and weight decay `1e-4`;
- 400 updates, batch size 5, and 2,000 frame exposures;
- schedule SHA-256
  `62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`;
- float32, no autocast, gradient clipping norm `1.0`;
- the same four losses weighted `0.25` each;
- final-update-only checkpoint selection;
- matched-RGB and wrong-RGB-with-target-calibration evaluations, unchanged
  metric verifier, and unchanged final gate; and
- GPU0 only on AMD Radeon AI PRO R9700, Raphael iGPU forbidden, at most five
  RGB workers, and one native math thread per process.

## Required adversarial evidence

Before different-agent review, V7 tests must prove:

1. both frozen V6 create/delete interleavings reject while ordinary owned claim
   and derived writes pass;
2. create/delete and move-in/move-out restoration at every transaction hook
   reject even when final names and fingerprints appear restored;
3. overflow, watch loss/reuse, unknown events, unpaired self move/delete,
   unmount, cookie mismatch, and event-order mismatch permanently prevent
   success;
4. exact event sequences pass for mkdir, create/write/close, replace, unlink,
   staging-to-claim rename, derived publication, cleanup, and failure receipt;
5. process-death staging recovery and failure cleanup remain durable and
   no-retry;
6. unrelated shared `.generated` direct-child churn still passes, while shared
   identity/security and exclusive-subtree changes reject; and
7. retained V1-V6 source, science, GPU, schedule, publication, cleanup,
   terminalization, and import-safety regressions remain applicable.

Tests and CPU contract smokes must hide every accelerator and open no
production input or canonical V7 output.

## Authority boundary

This amendment authorizes only V7 source construction and different-agent
review. Exact execution remains forbidden until a reviewer other than the V7
implementation author passes the complete frozen V7 closure. It grants no
retry, scientific retry, V5/V6 numeric read, second seed, N16, later training,
checkpoint use beyond frozen metric verification, G2, holdout, selection,
calibration change, runtime, hardware, navigation, production, or promotion
authority.

# Two-resolution navigation integration gap audit

Date: 2026-07-13

Status: **read-only architecture audit; navigation readiness remains open**

## Purpose

This audit traces the preregistered `0.05 m` physical / `0.10 m`
configuration design from perception through memory, exploration, target
belief, routing, and physical claim evaluation. It distinguishes an exact G3
control PASS from an actually usable development navigation pipeline.

No dataset, model output, checkpoint, GPU, G2, held-out, sealed input, or
result was opened. No implementation file was changed by the audit.

## Current blockers

1. **G3 V2 is not yet a downstream authority.** The first candidate lacks a
   distinct configuration-frame identity/revision, accepts forged and stale
   snapshots, omits execution-block projection and `frontier_cells`, and does
   not bind components/paths to a current snapshot. These findings supplement
   the formal source `BLOCK` in
   `lewm_go2_g3_exact_physical_equivalence_v2_source_review_2026-07-13.md`.
2. **Qualified learned admission is absent.** No reviewed adapter converts raw
   V4 source/ray evidence, calibration, and pose uncertainty into complete
   closed `0.05 m` world-cell transactions. The legacy memory transaction
   defaults to the same-grid projection contract.
3. **G4 is same-grid.** Viewpoints/routes are configuration cells, while camera
   visibility and swept evidence must be physical cells. The current code uses
   one `0.10 m` cell size and sometimes passes a configuration index directly
   to physical memory, which addresses the wrong map quadrant under a 2:1
   lattice.
4. **G5 is same-grid and production issuance is stubbed.** Its context is built
   from the legacy snapshot/morphology, binds one map frame, and compares target
   cell size to the physical frame. Target hypotheses must instead live on the
   `0.10 m` configuration frame while visibility evidence originates on the
   `0.05 m` physical frame. The runner-owned production adapter still rejects
   unconditionally.
5. **Target routing is missing.** No deterministic, revision-bound router turns
   posterior hypotheses into safe reacquisition/claim routes over the V2
   planner, and no adapter turns configuration cells into world-centre
   waypoints with a receipt.
6. **Composition is missing.** There is no non-test command that performs one
   reviewed inference, physical transaction, current V2 projection, G4
   selection, G5 update, target route, controller attempt, and observer-only
   canonical claim finalization with an actual-open/raw ledger.

The canonical physical claim evaluator is grid-independent and needs no
two-resolution geometry change. It already consumes full-precision world pose
and manifest geometry. Only correct orchestration and observer-only use remain.

## Required implementation order

1. Close G3 V2 source review with exact frozen hashes, two frame identities,
   current issued snapshots/components/paths, mapped execution blocks, and
   frontier enumeration.
2. Implement the qualified raw-V4-to-physical adapter at native `0.05 m`, with
   conservative pose uncertainty, retraction, and explicit projection-contract
   binding. Upsampling a derived `0.10 m` raster is forbidden.
3. Version G4 to keep routes/view history on `0.10 m` configuration cells and
   visibility/sweep/entropy on `0.05 m` physical cells, converting only through
   the shared world origin and binding both frames.
4. Version G5 context issuance so candidate/posterior cells use the
   configuration frame while positive and negative observations are derived
   from runner-owned physical evidence. Preserve the synthetic issuer only for
   posterior unit tests and retain exact one-writer/clone protections.
5. Implement a deterministic target router plus configuration-cell-to-world
   waypoint adapter over the V2 planner. It must reject stale snapshots and
   posteriors, UNKNOWN space, target cells, and unsafe paths.
6. Add one CPU-only development smoke command and test for the complete chain,
   including one-inference-per-tick accounting, raw outcomes, actual-open
   ledger, no evaluator feedback, and explicit development/non-hardware status.

## Required discriminating tests

- Use nonzero, high physical indices so configuration/physical same-index bugs
  cannot accidentally pass.
- Assert separately that one route step is `0.10 m` and one ray cell is
  `0.05 m`.
- Reject wrong frame, origin, 2:1 shape, profile/support hash, calibration,
  revision, resolution, copied outcome, replay, and legacy 89/69 context.
- Cover rotation, translation, closed-boundary FREE proof, OCCUPIED
  supercover, UNKNOWN omission, contradiction/retraction, mapped execution
  blocks, frontier aggregation, and exact candidate revalidation.
- Preserve the existing V1 89/69 and canonical claim-evaluator regressions
  byte-for-byte; they remain valid controls rather than downstream interfaces.

## Readiness rule

Navigation work is ready only when the six steps above have reviewed source
and the development smoke passes. An exact `96/96` G3 V2 audit, a V4 fit, or a
sound G5 posterior in isolation is necessary but not sufficient.

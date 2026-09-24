# Go2 G3 qualified learned-projection plan

Date: 2026-07-13

Status: **preregistered design; implementation blocked on V4, V5, and G2**

## Boundary

Promoted physical memory must never accept caller-created labels, logits,
probabilities, or aggregate metrics. The only learned admission input is an
instance-issued inference outcome produced by the canonical runtime runner.
That outcome must bind:

- the exact V5 checkpoint/model-state bytes and passed immutable G2 report;
- calibration, UNKNOWN/KNOWN and FREE/OCCUPIED-given-KNOWN thresholds, class
  semantics, native `0.05 m` local physical-cell geometry, and the selected
  raw source/ray output tensor bytes;
- source RGB/frame identity, camera transform, timestamp/synchronization ID,
  reset-local physical and configuration map frames, current physical and
  configuration revisions, and pose mean/covariance;
- the runner, inference implementation, projection implementation, and access
  ledger source identities.

The projection adapter independently reopens all of those canonical records.
Caller mappings and copied outcomes are not authority. While any production
identity is pending, promoted learned admission remains structurally disabled.

## Conservative projection

The model predicts observable physical UNKNOWN/FREE/OCCUPIED cells. Body
inflation is not part of the per-frame target or adapter.

1. Convert every admitted local cell to its closed body-frame square under the
   frozen camera/local-grid contract.
2. Construct the finite pose/camera uncertainty transform set from the frozen
   G2 calibration rule and the current covariance. Frames outside the admitted
   covariance envelope reject completely.
3. A destination `0.05 m` physical cell becomes a FREE witness only when its
   complete closed square is covered by admitted FREE support for **every**
   transform in the uncertainty set. Intersection, rounding, centre-only
   support, and upsampling a derived `0.10 m` raster are insufficient.
4. OCCUPIED witnesses use the closed union supercover of every possible
   transformed surface location. A missed or ambiguous surface remains UNKNOWN;
   it never becomes FREE.
5. Projection is origin-aware on the registered `0.05 m` reset-local physical
   lattice. Configuration morphology then uses the separately bound `0.10 m`
   planning lattice and the preregistered 316/276 cross-grid supports. Every
   output binds both exact map frames and the pre-commit revisions.

The adapter emits an opaque single-use transaction admission; memory reopens
the exact outcome and projection record before atomic commit. Source inference
outcomes are immutable, but their learned contributions remain retractable by
observation identity.

## Fusion and recovery

- Persistent fusion stores per-observation physical evidence and recomputes a
  cell after retraction; current-frame-only is a distinct registered ablation.
- FREE confirmation requires the frozen calibration-selected number and
  geometry of independent views. Near-identical pose/yaw observations count as
  one support. No numeric diversity threshold is selected before G2
  calibration evidence exists.
- OCCUPIED/FREE contradictions become unresolved until sufficient newer,
  diverse evidence resolves them.
- Verified executor traversal may retract a learned obstacle only inside its
  measured swept support. It cannot clear an execution/contact block.
- Execution/contact blocks dominate learned FREE. Unknown and conflicted
  physical cells remain non-traversable after the preregistered 316/276
  cross-grid morphology.
- Serialization preserves evidence and retraction history but never serializes
  live inference/admission authority.

## Cold start and view history

The first promoted snapshot may be seeded only by a reviewed reset-clearance
certificate or a complete frozen yaw scan whose observations pass this same
adapter. Stable stance alone is not a 0.47 m clearance certificate. G4 visual
history consumes the same runner-issued camera-view outcome and derives swept
cells from the registered camera model; callers cannot mark cells as viewed.

## Required tests

- exact FREE full-square coverage and OCCUPIED union-supercover properties over
  translated, rotated, boundary-touching, and covariance-extreme cases;
- rejection of a `0.10 m` physical frame, same-grid 313/277 support, centre
  alignment, wrong 2:1 shape, or any physical/configuration origin mismatch;
- excessive covariance, wrong camera/map/revision/checkpoint/G2/calibration,
  stale outcome, replay, transaction transfer, source substitution, copied
  adapter/memory, `object.__new__`, mutable alias, and serialization reload;
- view-diversity boundary and near-duplicate rejection; contradictory evidence
  recovery and exact retraction; traversal correction confined to measured
  sweep and execution-block precedence;
- ordinary-import/preloaded-module and temporary source-replacement rejection;
- exact adapter and direct learned/caller transactions remain forbidden in
  promoted runtime.

After a different-agent source PASS, the adapter first runs against exact
synthetic inference outcomes, then the passed V4/V5 checkpoint on development
roles. G3 output remains unauthorized until exact morphology equivalence,
cold-start, source isolation, and the preregistered fast coverage gate all pass.

# Go2 first-principles execution-plan corrections

Date: 2026-07-11

Status: active correction, written before G2, G3, G4, or G5 promotion.

## Reason

An independent source-and-contract review found that the ordered plan had
several gaps that could let an implementation pass a local test without proving
the deployment-valid information chain. These corrections change no observed
metric and open no held-out payload. They tighten the work required before later
gates may run.

## Corrections

1. N32 is not proven. The width-24 V2 model is only the controlled base for the
   next pose/hierarchy intervention. Dataset-backed shared-JEPA training remains
   blocked until the complete registered N32 ladder passes both seeds and both
   train-role holdouts.
2. A dynamic projection may consume only deployment-valid camera attitude and
   height derived from calibrated extrinsics plus IMU/proprioception. Recorded
   simulator world camera pose is training/audit metadata, never a promoted
   runtime input.
3. After one-shot G2, the encoder, physical-evidence head, calibration, and
   thresholds are frozen. Fine-tuning them requires a complete preregistered
   requalification on a fresh eligible untouched role. G2 role consumption is
   keyed by dataset, role, and protocol generation; changing a checkpoint hash
   cannot authorize another attempt.
4. The physical map and the body-configuration map are separate semantic
   layers. Learned physical evidence, verified traversal, and body-center
   execution/contact blocks remain distinct and reversible.
5. Post-fusion morphology uses two supports. FREE requires every physical cell
   whose closed square intersects the closed 0.47 m footprint disc. OCCUPIED
   requires an occupied physical-cell center inside the disc. Each support has
   a separate frozen hash and the dense dataset helper is the brute-force test
   oracle.
6. Rotated FREE projection must prove complete global-cell coverage by admitted
   source FREE squares. OCCUPIED may use conservative supercover. All
   registration includes the map origin, observation identity, pose source,
   covariance, and transform provenance. FREE support must hold for every
   transform in the admitted uncertainty set, OCCUPIED covers their union, and
   frames above the frozen uncertainty limit are rejected.
7. Repeated near-identical views cannot accumulate as independent FREE proof.
   Evidence admission and view-diversity thresholds use only current physical
   train sequences plus the registered probability-calibration role and freeze
   before any V4 closed-loop output. A stable stance and successful traversal
   certify only their measured actual body polygon, not the larger 0.47 m disc.
   Cold start therefore needs either a recorded reset-clearance certificate or
   enough bootstrap-scan evidence; otherwise it fails closed.
8. Exact zero-inflation physical evidence must pass through the same fusion,
   two-support morphology, configuration snapshot, frontier, and A* path as the
   learned arm. Scene-level equivalence and cold-start connectivity must pass.
   A separate privileged-target G1 regression must retain 96/96 claims, while
   the G3 exact-map reference uses the same no-anchor controller boundary as
   learned arms and is scored on coverage/opportunities.
9. G3 uses a new nonprivileged runner and runtime, not the legacy closed-loop
   benchmark. Exact simulator pose and geometry may execute physics and score
   outcomes, but they cannot enter controller inputs. Candidate and promoted
   checkpoint load modes are separate.
10. Frontier candidates are reachable viewing poses `(cell, yaw)` or frozen
    scan sequences, not cells alone. G3/G4 must gate beacon-visibility
    opportunities as well as area coverage; the 2,400-tick development target
    is 96/96 beacons receiving at least one valid visibility opportunity. A
    pre-output G3 binding freezes the opportunity evaluator/denominator,
    corrected baseline hashes, clustered interval method, and numeric safety
    tolerances; a pre-output G4 binding freezes the deterministic candidate and
    information-gain baseline.
11. Per-color target memory is a reversible multimodal spatial belief with
    positive and negative evidence, competing hypotheses, age, and uncertainty.
    Controller-declared claims and ground-truth-verified claims are separate.
12. One canonical claim evaluator is shared by labels, oracle, runtime traces,
    and scoring: correct requested target, inclusive distance radius, physical
    line of sight, and inclusive absolute bearing threshold at the claim tick.
    Historical distance-plus-LOS rescoring remains diagnostic until it is
    rerun through this evaluator.
13. The JEPA branch and current-frame physical head are parallel consumers of
    one encoder. A matched no-JEPA-loss development ablation is required before
    claiming that predictive representation learning improves generalization;
    it is evaluated at the promoted arm's preselected update and cannot select
    its own checkpoint.

## Revised order

1. Finish the fit-only pose audit and pass the ordered N32 intervention.
2. Build and qualify the shared JEPA plus physical-evidence model; freeze it
   after one-shot G2.
3. Build the two-layer map, exact morphology equivalence, cold-start evidence,
   strict pose provenance, checkpoint-v5 runtime, and isolated G3 runner.
4. Pass G3 with area and beacon-visibility-opportunity coverage.
5. Train a frontier-viewpoint value head with oracle future gain and DAgger;
   pass G4.
6. Add reversible target beliefs and learned observation/claim heads under the
   unified physical claim contract; pass G5.
7. Iterate on development and fresh rolling qualification cohorts until G6 and
   G7 pass, then create and execute a fresh opaque G8 role once.

This correction review caused one additional broad source search to surface
only the common `claim_radius_m` line from legacy sealed-manifest files. That is
separate from the earlier V4 incident, which exposed sealed metadata/scene
identity and permanently invalidated V4 as recorded in
`docs/lewm_go2_v4_sealed_invalidation_2026-07-10.md`. No fresh final sealed role
exists. Ordinary ripgrep did not honor the repository's `.rgignore`, so a real
`.ignore` guard was added and verified before any future sealed role is created.
No G2 image, label, checkpoint/model output, non-train payload, or held-out
model output was opened to make these corrections.

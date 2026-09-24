# Go2 held-out maze authority and architecture correction

Date: 2026-07-24

Predecessor commit:
`99370af8d895a8de30a7d6a3ea663e080b535af8`

Status: **GOVERNING CORRECTIVE AMENDMENT; DOCUMENT AND SOURCE-REVIEW
AUTHORITY ONLY; NO DATA, CHECKPOINT, GPU, TRAINING, G2, DEVELOPMENT-RUN,
RUNTIME, HARDWARE, HELD-OUT MATERIALIZATION, OR HELD-OUT EXECUTION AUTHORITY**

## Scope and precedence

This narrow amendment supersedes every contrary custody, architecture, freeze,
promotion-order, and clean-checkout statement in:

- `docs/lewm_go2_heldout_maze_full_handoff_2026-07-24.md`; and
- the older committed bytes of
  `docs/lewm_go2_generalization_execution_contract_2026-07-09.md`.

Historical records remain immutable evidence. Commit order cannot erase a
factual access incident. The broader dirty edit of the execution contract is
not included in this amendment and remains separate pre-existing work.

## Sealed-role custody

This amendment accepts the recorded 2026-07-10 access incident as governing
scientific fact: a broad repository search byte-read
`config/go2_generalization_v4/sealed_test.json`. No image, label, model output,
navigation result, or aggregate sealed performance was opened according to the
incident record, but the manifest-opacity requirement was broken. The
canonical incident record is
`docs/lewm_go2_v4_sealed_invalidation_2026-07-10.md`, SHA-256
`696f2256144b6e2516a3276c76d8afa009a2dbbb96f2bd9269a4eccb52594605`.

Effective immediately:

1. V4 train and development evidence remains development evidence.
2. The V4 sealed role is permanently ineligible for G8 and must not be
   materialized, evaluated, or accessed again. Its commitment and hashes
   remain only as historical provenance.
3. No current or historical sealed manifest may guide architecture, data,
   threshold, calibration, checkpoint, or runtime choices.
4. G8 is `BLOCKED — NO ELIGIBLE SEALED ROLE EXISTS`.
5. A replacement role may be generated only after G3 through G7 pass and the
   complete deployment code, source graph, checkpoint, calibration,
   thresholds, evaluator, environment, and output schema are frozen.
6. Replacement requires a separately reviewed preregistration, independent
   custody, a new generation namespace and seed, and scene-hash exclusion
   against training, development, physical-authoring, every previous sealed
   role, and every otherwise visible scene.
7. The active replacement manifest must live outside the model-facing
   checkout under operating-system access control. Model-facing processes may
   receive only its commitment and aggregate integrity counts before
   execution.
8. The replacement may be accessed only by its frozen one-shot G8 launcher.
   This amendment does not authorize its creation, materialization,
   inspection, or execution.

Tracked `.ignore`, `.rgignore`, and repository-agent instructions are
defense-in-depth against ordinary recursive searches. They are not an
access-control boundary and do not replace external custody.

## Access record

| Date | Event | Consequence |
|---|---|---|
| 2026-07-10 | A broad read-only search byte-read the V4 manifest and exposed limited metadata/scene identity. | V4 permanently invalidated for G8. |
| 2026-07-11 | A later broad search surfaced only the common `claim_radius_m` line from legacy sealed-manifest files. | No final role was eligible; the exposure grants no authority and motivated a real `.ignore` guard. |
| 2026-07-24 | Filename-only verification established that ordinary `rg` review searches honored the pre-existing `.ignore`. | Those searches read no manifest bytes and are not access incidents. |
| 2026-07-24 | A delegated source-closure audit used broad `git grep`, which bypassed `.ignore`, and exposed one commitment-SHA line from the already-invalid V4 manifest. | Recorded as a re-access; V4 status is unchanged, the exposed line is excluded from decisions, and closure findings require clean independent reproduction. |

The access rows are recorded in
`docs/lewm_go2_v4_sealed_invalidation_2026-07-10.md` and
`docs/lewm_go2_first_principles_plan_corrections_2026-07-11.md`, SHA-256
`b1c5e6087e4956a71cf048cccdd8408384305761a64d9405e08906fd84cc8042`,
with the 2026-07-24 re-access recorded in
`docs/lewm_go2_v4_sealed_reaccess_2026-07-24.md`, SHA-256
`1d10377b6562dd1902a8ca07afa5990717d760f85ee77edcdf60aacf0659c204`.
Exact command details absent from those records must be recovered from retained
tool logs, not guessed.

## Meaning of fully learned JEPA navigation

The repository objective is not satisfied by a supervised visual encoder whose
features feed a classical navigator while the JEPA predictor is discarded at
deployment. The final claimed arm must meet all of the following:

1. Egocentric RGB, deployment-equivalent odometry, IMU/proprioception, and
   executed-command history are its only controller observations. Exact
   simulator pose, scene geometry, analytic color masks, privileged target
   vectors, ground truth, and evaluator feedback remain development-oracle or
   scoring inputs only.
2. Learned visual evidence feeds persistent, reversible physical and
   per-color beliefs with uncertainty and provenance. One reset-local episode
   maintains all four canonical colors on every tick and may attempt at most
   one physically verified claim per color. Four separate single-color
   invocations are not 4/4 completion.
3. An action-conditioned JEPA predictor or latent rollout causally affects
   deployed candidate or action scores. A predictor used only as a training
   loss does not satisfy the claim. The deployed artifact and action-source
   trace must bind the predictor.
4. Learned target, frontier/viewpoint, route/subgoal, and ordinary-motion
   primitive selection determine promoted behavior. Deterministic evidence
   bookkeeping, candidate enumeration, feasibility checking, a fixed low-level
   gait/primitive executor, and a fail-closed safety veto are allowed. They may
   execute or restrict a learned decision, but may not choose the target,
   frontier/subgoal, route/waypoint, or ordinary motion primitive. Emergency
   stop is the only deterministic action override.
5. A matched development arm with JEPA losses disabled is mandatory before
   attributing a generalization improvement to JEPA. It shares initialization,
   data, architecture, optimizer schedule, presentation count, selection
   update, and downstream protocol; it may differ only in the preregistered
   predictive objective or treatment and may not select its own checkpoint.
6. Promoted physical memory persists learned evidence across frames and
   supports selective retraction or contradiction.
   `FusionMode.CURRENT_FRAME_ONLY` is forbidden in the claimed arm.
7. Promoted frontier value consumes the frozen map, candidate, view/history,
   uncertainty, yaw, route, path-cost, and safety features.
   `relative_dx_dy_distance_zero_pad_v1` is not a promotable map-conditioned
   value input.

The existing Shared V5 development runner remains useful only for its reviewed
execution shell: artifact cross-binding, one RGB/shared encode per tick,
cached-feature fanout, reset/revision checks, fault sealing, hash-chained
evidence, output custody, and post-controller observer isolation. Its
current-frame-only memory, fixed single-target invocation, exact-pose
kinematic backend, frontier/A* action selection, and hand-coded commands are a
development diagnostic, not the promotable fully learned controller.

## Encoder and promotion freeze

Perception qualification freezes an output contract, evaluator, data roles,
calibration procedure, and acceptance thresholds. It does not silently convert
a supervised-only encoder into a JEPA backbone.

Before G2, JEPA training may update the shared encoder only under a
preregistered separated optimizer/clipping contract that retains predictive
and anti-collapse objectives. Any such encoder update invalidates the earlier
perception checkpoint and requires complete physical
selection/calibration qualification before a G2 candidate is published.

After one-shot G2, the encoder, physical-evidence head, checkpoint,
calibration, and thresholds are immutable. Changing any of them creates a new
candidate that requires a fresh eligible untouched G2 role under a new protocol
generation. A consumed role cannot be reopened because the checkpoint hash
changed.

## Binding promotion order

1. Close and independently review the committed source graph while every
   production authority identity remains fail-closed.
2. Qualify one materially different perception mechanism on development
   roles.
3. Train the separated action-conditioned JEPA stage, run the mandatory
   no-JEPA development arm, and requalify perception after any encoder change.
4. Freeze the complete candidate and execute a fresh eligible G2 role once.
5. Pass G3 using persistent physical/configuration belief,
   deployment-equivalent pose inputs, exact-path equivalence, cold-start
   controls, area coverage, and beacon-visibility opportunities.
6. Pass G4 using reachable viewing-pose candidates and genuinely
   map-conditioned learned value.
7. Pass G5 using reversible multimodal per-color target belief and the
   canonical heading-aware physical claim evaluator.
8. Pass full-development G6 and locomotion/odometry/noise/physical G7.
9. Freeze everything, create the fresh guarded G8 role, and execute it once.

A valid one-shot G8 failure completes an experimental campaign but does not
achieve the repository objective. The objective is achieved only by successful
novel-maze, all-four-beacon navigation under the frozen fully learned JEPA
stack.

## Repository closure rule

No G2 or Shared V5 runner invocation may claim clean-checkout reproducibility
until a machine-checkable recursive source manifest:

- names every fixed entrypoint and local import/runtime-source dependency;
- binds every source file by SHA-256;
- classifies production authorities and runtime artifacts as generated or
  pending rather than source dependencies;
- passes from committed bytes in an isolated exported tree; and
- preserves every production authority hash as `None` until separately
  reviewed artifacts actually exist.

The current development controller is included only to make the reviewed shell
reproducible. It remains barred from all G3 through G8 promotion evidence until
a separately versioned fully learned controller successor passes review.

Committing source closure never licenses G2, navigation, production, or
held-out execution by itself.

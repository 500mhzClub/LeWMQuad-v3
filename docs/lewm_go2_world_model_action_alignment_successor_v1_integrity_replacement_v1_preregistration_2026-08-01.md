# Action-alignment successor V1 integrity replacement V1

Date: 2026-08-01

Status: **PREREGISTERED SCIENCE-IDENTICAL INTEGRITY REPLACEMENT; NOT EXECUTION AUTHORITY**

## Closed predecessor and purpose

The sole original attempt,
`world_model_action_alignment_successor_v1/attempt_v1`, is terminally
consumed. It failed in the first arm of the first microbatch before the
objective returned, before any backward call or optimizer step, and before a
training update, evaluation panel, snapshot, metric bundle, result, checker,
or scientific verdict existed. It is neither a statistical stall nor evidence
for or against the action-alignment hypothesis.

The exact audit is
`docs/lewm_go2_world_model_action_alignment_successor_v1_terminal_preupdate_source_integrity_failure_result_2026-08-01.json`,
SHA-256 `3f8350528c4985b792d22b5d4002b3cc34d926c7a2d8a84431009d6668bd63ed`,
7,173 bytes. Replacement authority must bind that audit and the original
authority, reservation, failure, and terminal-supervision receipts. Those
documents are identity evidence only. No tensor, checkpoint, pack view,
partial state, RNG state, or other runtime artifact from the failed attempt is
eligible as an input.

This is one final, science-identical implementation-integrity replacement. It
does not authorize execution and grants no later integrity replacement.

## Established source defect

For each 32-row microbatch, the original selector flattened the eight wrong
actions per row and evaluated the resulting 256 row/action pairs in two
no-gradient batches of 128. It then recomputed the selected wrong action with
gradients using the original 32-row order. The GRU and multihead-attention
path therefore saw different autograd modes, batch shapes, and row placement.
On the bound ROCm runtime the unchanged `1e-6` consistency guard observed a
finite maximum energy difference of `0.010936617851257324` and failed closed.

A payload-free exact-runtime reproduction using a fresh synthetic predecessor
and synthetic B32 tensors reproduced the same error class with maximum
difference `0.0020215511322021484`. This isolates an implementation comparison
between noncommensurate numerical routes; it is not experiment evidence.

## Sole functional correction

For every original 32-row microbatch, construct an integer tensor of shape
`[32, 8]`. Each row contains the eight absolute action IDs other than its
factual action, in ascending order. For slots zero through seven:

1. preserve the original 32 rows in their original positions;
2. substitute that slot's wrong action for each row;
3. run one B32 prediction and energy computation with autograd enabled, so its
   dispatch matches the later selected-wrong recomputation;
4. immediately detach the 32 energies before assigning them into a
   preallocated non-gradient `[32, 9]` scan at their absolute action columns;
5. delete the prediction and energy references before the next slot so no scan
   graph survives.

The factual column remains positive infinity. Absolute-column `argmin`
retains lowest-action-ID tie breaking. Factual and selected-wrong energies are
then recomputed with gradients at B32 exactly as before. The scan/recompute
maximum-error threshold remains `1e-6`; it may not be relaxed.

This remains exactly eight wrong scans plus one factual and one selected-wrong
recomputation: ten head row-presentations per arm per training row. Both arms
execute the identical shapes, row order, candidate coverage, and timing. Scan
energies and their temporary graphs contribute no gradient. The baseline's
total remains factual loss alone and must still reproduce every frozen V3
anchor within `1e-15` at terminal evaluation.

## Frozen scientific identity

Everything else remains exactly the original registered comparison in:

- `docs/lewm_go2_world_model_action_alignment_successor_v1_preregistration_2026-08-01.md`;
- `docs/lewm_go2_world_model_action_alignment_successor_v1_plan_2026-08-01.json`.

In particular, this freezes the same spatial predecessor, immutable V3 pack,
16,000 training rows, 2,048 validation rows, initialization, seed, hash-ordered
179,200-presentation schedule, 700 updates, batch 256, microbatch 32, masks,
optimizer, learning rates, clipping, frozen substrate, arm coefficients
`0.0/1.0`, margin `0.01`, objective, u500/u600/u700 panels, full-train panel,
paired bootstrap, thresholds, retention gates, decision precedence, and claim
boundary. It also freezes the 9,000-second wall cap, 7,200 GPU-second discrete
cap, exact one-device runtime, and no-RGB/no-generation/no-network custody.

There is no architecture, model-value, data, objective, coefficient,
schedule, threshold, tolerance, evaluation, or scientific-accounting change.
A normalized plan comparison must report zero scientific differences after
removing lifecycle identity and the single scan correction above.

## Mandatory source-only gates

Before authority, focused tests and one payload-free exact-ROCm B32 probe must
establish all of the following:

- eight calls of shape B32 preserve row order and cover every wrong ID exactly
  once per row while excluding factual IDs;
- scan tensors have no gradient function, scan forwards leave parameter
  gradients and model state unchanged, and at most one discarded scan graph is
  live;
- absolute-ID ties select the lowest ID;
- objective value matches a fully differentiable unique-minimum reference at
  absolute tolerance `1e-6`, and all 36 parameter gradients pass float32
  `rtol=1e-5, atol=1e-6` against that reference; exact maximum errors are
  recorded rather than used to tune these tolerances;
- arm state hashes and CPU/CUDA RNG states are exact before versus after the
  scan/objective forwards;
- coefficient zero retains exactly the factual gradient;
- selected scan and recomputed energies agree within the unchanged `1e-6` on
  the exact ROCm path;
- both arms retain the same ten-presentation call contract; and
- the exact baseline reproduction, authority, reservation, inventory,
  terminalization, custody, and no-follow-on gates remain fail closed.

If the exact-ROCm probe exceeds `1e-6`, changes state or gradients, or cannot
fit the frozen caps, this branch stops before authority. The threshold and
batching may not be tuned again.

## Fresh lifecycle and terminal logic

The only replacement identity is
`world_model_action_alignment_successor_v1_integrity_replacement_v1/attempt_v1`
at
`.generated/dev/world_model_action_alignment_successor_v1_integrity_replacement_v1/attempt_v1`.
Its root must be absent. Reservation consumes its sole attempt. Retry, resume,
refill, overwrite, splice, warm-start from the failed attempt, and further
integrity replacement are false.

Source/preregistration/tests, independent source review, and one-shot authority
must be separate commits in that order. Any runtime integrity, custody, or cap
failure is terminal and closes this implementation path without a scientific
alignment conclusion.

If and only if replacement V1 reaches u700 and passes its checker, the original
decision logic applies unchanged: repair, retention failure, meaningful
incomplete, statistical stall, or inconclusive. A stall or retention failure
closes alignment tweaking. Meaningful incomplete or inconclusive retains only
the one separately preregistered scientific continuation or identical
replication already allowed by the original plan; nothing is automatic.

Even a repair remains an exploratory optimized latent proxy. It does not
establish factual learnability, executed-action causality, planning utility,
navigation, WM-A, WM-D, promotion, deployment, or production readiness.

# Partial layout audit and localized raster failures

The first independent-layout batch is terminal, not running. It stopped under
its original visibility gate after 64 of 120 prescribed episodes. Its frozen
terminal auditor completed successfully (51564, exit 0), reconstructing all 64
recorded episodes and retaining all 56 unattempted entries. There are 148,800
native physics samples and 2,080 RGB-D frames. All 64 setups, departures and
command schedules completed; no positive contact targets were observed.

Of the 64 episodes, 63 pass their per-episode visibility checks and materialize
63 train-role samples. These are not independently validated training authority:
the one layout remains training-only, and exact matched-action prefixes fail.
Of 100 prescribed non-reference comparisons, 53 have both prefixes available:
1 matches exactly, 52 differ, and the other 47 are unavailable. No threshold is
relaxed, no failed episode is rescued, and no new recorded-data fit is launched.

Output is the external navigation root child
`go2_independent_layout_collection_v1_l00_attempt_001`.
Collection failure SHA-256:
`b97a6d5d3dad66fdd0fbffaad811dfb3b6feee5e9f4c7363e8aec3f5a04a556b`.
Terminal audit SHA-256:
`33a309f4f90dca5d96c22d36c1cb7ed9ceed8fb36b68bbe7bc4a471cdb89aed1`.

## Depth failure: foreground silhouette, not near clipping

The retained failure is `l00_junction_recent_forward_nominal_a3`, frame 11,
sampled pixel (row 124, column 252). The ideal centre-ray reference expects
2.679856833m on a distant wall; native depth is 1.527716875m. No sampled opaque
ray is clipped by the 5mm near plane and no below-range ray is falsely public-valid.

The native ray endpoint is approximately (1.83997425, 0.55998960, 0.74402523)m,
next to the foreground wall's vertical edge at (1.84, 0.56, z). Its image-space
distance from the exact projected edge is only 0.000441242 pixels. The saved
3x3 neighbourhood straddles foreground/background depth. This supports a
raster-coverage/precision boundary hypothesis, not a metre-scale interior
geometry error; the native subpixel precision has not yet been independently
measured, so the precise mechanism remains unqualified.

Source review identifies a limitation in the existing reference: it excludes
edges of the winning centre-ray box only. A background interior can therefore
be scored even arbitrarily close to a different, foreground silhouette. The
original failure remains valid under its frozen contract. A future measurement
contract must account for projected foreground boundaries and finite pixel
footprints without declaring those pixels free, dropping near-plane occluders,
repairing observations from privileged geometry, or hiding coverage losses.
It must retain the old strict score alongside new ambiguity accounting and be
tested prospectively; this diagnosis does not authorize posthoc batch salvage.

## RGB failure: floor/wall edges and draw-order hypothesis

The completed union-wall static bench preserves identical depth and passes all
four sampled physical-visibility checks, but exact RGB differs by 3 pixels at
pose 0 and 5 at pose 1. Thus `passes_fixed_bench` is false. Its result SHA-256 is
`6a64c9e07d87ae91f0a07c46189dc74b07b1f9625c87c4a7a54f2799c617189d`.

The read-only diagnostic 20735 completes with exit 0, verifies original source
and input/output bindings before/after, and independently reproduces the saved
depth failure and every RGB difference. All eight residual RGB differences lie
within 0.382 pixels of physical floor/wall edges. The diagnostic JSON has SHA-256
`ab7f86f5ddedb7fdc13de970741ee4f72edb29f8c25d4e9350d86886e4f31f61`:
[recorded diagnostic](go2_recorded_raster_failure_diagnostic_2026-09-06.json).
Projected edge proximity includes hidden edges and is diagnostic, not causal proof.

Native renderer source starts node sorting from a set and sorts solely by
distance to node origin. The two world-coordinate surface nodes share an origin,
so their relative order is not fixed. The original bench did not record draw
order. A separate [two-order intervention bench](go2_ordered_union_rgb_repeatability_probe_v1_2026-09-06.md)
tests explicit floor-first and walls-first ordering, each in two independent
builds, with original geometry, materials and camera settings unchanged.

The instance-local ordering helper and union geometry pass 25 focused tests
(13 ordering, 12 geometry; 0.62s). First read-only preflight 7397 fails before
output creation because the inherited verifier requires the exact original
native-binding roster. The additional native scene.py witness is now a separate
explicitly checked binding; no installed renderer or predecessor is modified.

The corrected preflight 44861 passes with 729 source bindings. Native experiment
22092 then completes with exit 0: all four same-order pose comparisons have exact
RGB and native depth, and all eight captures pass the original sampled 1mm
visibility checks. Reversing order changes exactly 3 pixels at pose 0 and 5 at
pose 1, identically in both repeats. The actual JIT order and vertex identities
are checked after every render. The experiment uses zero physics steps.

Independent read-only verification 49992 exits 0 and checks all 40 output files
(42,230,776 bytes including launch metadata). Each original failed-bench repeat
is bit-identical to the corresponding controlled-order RGB image: original
repeat 0 matches floor-first, original repeat 1 matches walls-first, at both
poses. The exact changed-pixel masks also agree. This establishes a reproducible
draw-order mechanism for these static failures, not dynamic or hardware repeatability.
Ordered launch SHA-256:
`faf4511c45f9781f85413e9b41e3155eb2a51e1b1560d0826f8b901e68d5dbdd`.
Ordered result SHA-256:
`6f8a966e2bcd7268bf2b0b274a565820e8c25d2102ddcd2318e3603c084fc0bf`.

## Prospective footprint accounting preparation

`lewm/raster_footprint_visibility_development.py` adds evaluator-only coverage
accounting. It derives a conservative full-pixel boundary mask from all physical
box edges, including foreground silhouettes and hidden/internal edges. It reports
the unchanged original strict score, stable-interior and ambiguous-boundary ray
counts and errors, and preserves near-clipping/false-valid failures independently.
There is deliberately no overall qualification or training-eligibility grant.
No controller receives the privileged boundary mask; no native depth is repaired.

Nine new synthetic tests cover the foreground-silhouette counterexample, genuine
interior corruption, measurement-independent partitioning, invalid geometry and
near-clipping preservation. An initial fixture accidentally intersected a thick
wall's side face; it was corrected to miss the far vertical edge before the final
test run. Combined with ordering, union and original first-surface tests,
58 tests pass in 2.25s (49581, exit 0). The full explicitly enumerated 224-file
regression completes with **2,870 passed in 220.68s** (5552, exit 0). These are
implementation tests, not additional physical or scientific navigation trials.

Read-only posthoc diagnostic 24100 also completes (exit 0), applying this coverage
accounting to saved frames 10, 11 and 12 without changing their strict scores.
At frame 11 the 4,236 original compared rays partition into 4,137 stable-interior
and 99 boundary-ambiguous rays. The single large residual is in the latter;
stable-interior maximum error is 0.217972mm. Frames 10 and 12 retain their strict
passes. This is explicitly posthoc explanatory evidence, not a newly passed
qualification. Output SHA-256:
`e90bdc245a8991e0da7e8aa18168ee8e722cb6105c38e67738e26026c5f4c736`.

## Scientific next steps remain substantive

1. The fixed draw-order intervention is complete. Next establish mounted dynamic
   repeatability and a prospective finite-pixel visibility contract with honest
   boundary uncertainty. Keep all previous failures and missing contexts.
2. Only then define a distinct bounded data acquisition using the predetermined
   layout roles, complete action/history/support/context coverage and actual
   hazard outcomes. Do not substitute this partial one-layout dataset for it.
3. Compare direct prediction, supervised rollout and JEPA across paired seeds
   and independent layouts, including action/time and zero-motion baselines,
   RGB/history/action ablations and the matched cumulative-contact head.
4. Establish useful prediction and reliable local physical execution, then test
   online rollout and memory/backtracking contributions on actual maze missions.
   Real-time execution, deployment-valid sensors and hardware remain outstanding.

The concrete next implementation is the
[eight-episode moving sensor pilot](go2_ordered_union_dynamic_sensor_next_steps_2026-09-06.md),
including the failed junction context and an uncollected near-wall/low-friction
context. It is a plan, not an executed or frozen collection.

The previous status turn is classified as progress because fresh terminal files
changed the next action from monitoring to auditing two failed runs. This turn
completes that audit and yields localized evidence, rather than counting status
restatement as progress. The full scientific objective remains active/unachieved.

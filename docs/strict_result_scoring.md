# Strict navigation result scoring

`lewm.benchmarks.strict_result_scorer` scores a closed-loop result without
importing or trusting the benchmark implementation that produced it.

## Inputs and trust boundary

The scorer requires:

- the result wrapper or raw per-tick log;
- the exact `SceneManifest` used by the run;
- the versioned `GeometryContract`;
- the sealed benchmark manifest when the scene belongs to one.

Scene identity, target positions, LOS geometry, body inflation, claim radius,
and reachable area come from the manifest and geometry contract. Stored claim
distances, claimed colors, success flags, map coverage counters, and aggregate
stall/contact counters are proxies used only for discrepancy reporting.

The output binds the source payload, scene manifest, and geometry contract by
SHA-256.

## Metrics

- Claims are accepted only when the reconstructed pre-action pose at the claim
  tick is within the inclusive claim radius and has manifest LOS to the exact
  target. Each target counts once.
- Completion reports all-target completion, explicit 4/4 completion when the
  manifest has four landmarks, and the tick of the final first valid claim.
- Coverage uses the 10 cm coverage grid and the exact fixed-spawn reachable
  configuration-space component. Every consecutive pose segment is
  supercovered, so sparse poses cannot skip crossed cells.
- Normalized coverage AUC is the trapezoidal integral of cumulative reachable
  coverage fractions from the initial pose through every logged tick, divided
  by the number of observed tick intervals. It is in `[0, 1]` and uses the
  observed run duration.
- Canonical collision ticks are recomputed by substepping every trajectory
  segment at the geometry contract's maximum translation substep and testing
  continuous body-inflated obstacle clearance. Logged contacts remain a
  separate evidence stream.
- Stall and hard-stall ticks come from per-tick log fields. Legacy aggregate
  counters are retained as proxies and compared with the tick reconstruction.

## Sealed evaluation

A scene whose split is `sealed_test`, or any score supplied with a sealed-test
manifest, is rejected before geometry or result scoring. The caller must pass
the one-shot `authorize_sealed_final_evaluation=True` argument. The CLI
equivalent is `--authorize-sealed-final-evaluation`; it is not persisted.

## Legacy and missing-field limitations

Current legacy benchmark logs store rounded `post_xy` on action rows and omit
pose fields on stationary `CLAIM` rows. The scorer uses the prior tick's
`post_xy` as the claim tick's pre-action pose and records the precision
limitation. A claim near a strict boundary should therefore be rerun with
full-precision event poses before publication.

If a log, tick, initial pose, action post-pose, claim target, or claim pose is
missing, the dependent metric is returned as `null` or the claim verdict is
unverifiable. Stored proxy distance is never substituted for missing pose
telemetry. Tick gaps, duplicate ticks, trajectory discontinuities, malformed
rows, missing contact/stall fields, non-four-target scenes, and summary/log
counter disagreements are all emitted in `limitations` or `discrepancies`.

Slice logs starting at nonzero ticks are reconstructable only when they provide
an explicit first pre-pose or `wall_metrics.slice_start.start_xy`. Preclaims
before the slice remain unverifiable unless their event pose is included.

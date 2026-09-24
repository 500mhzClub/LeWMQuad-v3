# Observed traversal: completed integration, no learned-method separation

All20 fixed trials completed and the full raw audit passed. Actual RGB opening
proposals now initiate physical motion, followed by a new sensory arrival
candidate and explicit zero release. Each of the three learned-model arms
meets the fixed one-transition integration criteria on4/4 local fixtures.
However, all three choose forward on every decision and have identical physical
trajectories. This is an integration result, not evidence that JEPA—or learning
at all—is necessary for this task.

## Fixed outcomes

The [protocol](go2_observed_traversal_development_v1_2026-09-05.md) pairs four
1.2 m local corner/tee fixtures across five methods, with no oracle route
following after zero-command settling. There are no native contact stops,
body-stability stops or sensor faults in any of the20 trials. No trial was
rerun, no threshold changed and no checkpoint fitted or selected.

| Method | Integration successes /4 | Arrival candidates failing geometry /4 | Physical arrivals without a candidate /4 | Other outcome |
|---|---:|---:|---:|---|
| Always stop |0|0|0|4 timeouts|
| Directional gait |0|3|1|1 insufficient visual-change termination|
| Direct-only prediction |4|0|0|All34 choices forward|
| Supervised recurrent prediction |4|0|0|All34 choices forward|
| JEPA recurrent prediction |4|0|0|All34 choices forward|

Integration success requires correct candidate-time whole-body crossing and
destination membership, then sustained final crossing and stable complete
release. It does not establish place identity, safe subsequent turning, beacon
discovery, return or navigation on independent mazes. Every provisional ledger
still contains zero trusted edges and no identified destination place.

Learned arms travel1.227,1.235,1.557 and1.235 m in fixture order, with total
post-settle durations5.3,5.3,6.6 and5.3 s. The same geometry/initialization
produces exactly identical raw physics arrays across all three learned methods.
There are only eight distinct raw-array trajectory groups across all20 trials;
the repeated fixtures/frames cannot supply independent-maze confidence intervals.

The non-learned directional controller travels about1.04 m in its three
premature-candidate cases. Candidate-time rear-most articulated extent is
0.00940 m or -0.00457 m past the opening, below the fixed +0.02 m requirement.
The first case has cleared the mathematical plane but not the prescribed margin;
the other two still straddle it. These are failures of the fixed criterion,
not three collisions. The remaining tee case travels1.415 m and has a viable
physical arrival but never satisfies the visual-change rule. Appearance change
and commanded-distance accumulation are therefore inadequate arrival evidence.

## What the comparison does and does not establish

The deployed control architecture remains hybrid: learned RGB/body consequence
prediction, a fixed action-ranking cost, fixed RGB floor-extension proposals,
hand-designed progress/arrival rules and a separate gait controller. JEPA does
not directly output a learned navigation policy, and the recurrent head uses
only a single half-second transition rather than multi-step trajectory search.

All102 learned choices select the same nominal0.3 m/s straight command. The
directional baseline instead uses0.2 m/s and100 ms heading feedback, versus
500 ms learned decisions and unchanged50 Hz adapter cue transport. Consequently,
the success-count difference must not be attributed to learning: speed, cadence,
heading control and stopping-phase interaction are confounded. This panel did
not execute a matched0.3 m/s fixed-forward arm. Identical commands make that
control an important next comparison, not an already observed physical result.

The longer tee trajectory also shows arrival sensitivity to scene appearance
and gait/observation timing. Do not repair this panel by increasing a distance
margin or lowering the mask threshold after seeing the result. The model-based
arms succeeding here does not calibrate their arrival rule on novel geometry.
The existing palette observer still fails appearance controls; current body
extent is not future swept clearance, and command integration is not odometry.

## Verified evidence

Collection session86794 completed20, exit0. Audit session29661 passed20, exit0,
reproducing all1,399 actual-packet decisions, including102 fitted ensemble
selections. It checks88,950 physics samples,88,950 live high-rate gyro samples,
8,895 ordinary sensor samples and1,499 actual RGB packets, plus mounted camera
transforms, native contact attribution, paired settling, command slew, ledger
state and raw geometry/release reductions. Only inference_ms and adapter_ms are
excluded from exact scientific-decision equality; their validity is checked.

Mean adapter timing is21.18 ms direct,22.81 ms supervised and22.65 ms JEPA.
These measurements exclude image acquisition/ingestion and simulation waits;
they are not real-time hardware guarantees. Ideal gyro/force acquisition and
privileged native emergency stops remain simulation assumptions. The inherited
calf/foot ground-contact grouping does not certify foot-only ground support.

All177 source/test/protocol paths,162 input bindings and two gait bindings are
unchanged at full audit. Before launch,828 tests passed across74 explicit files,
including25 new integration/evidence tests. No protected benchmark material was
accessed. No source-only export or real-hardware action was performed.

Root: `.generated/go2_observed_traversal_development_v1_attempt_001`.

- Launch SHA-256: `5597956c5847d56788112bb5029caabd4ed0b5c8918787b7ec95bed538ca54d0`.
- Result SHA-256: `6498693d8741a328b09f5c78df580e902c56af2a282e82468ffd802260cba33d`.
- Full audit SHA-256: `b0c6bfea4284d47664f31533ad64cf292dca9c79c30db0f63abdaec9d9d901f1`.

## Next steps toward the actual objective

1. Continue from a sensor-proposed arrival into fresh junction observation,
   a non-forward exit selection and a second physical traversal in the same
   episode. Keep continuous raw state; do not teleport or inject destination
   coordinates/place labels. A successful first crossing is not a qualified
   location from which an in-place scan is safe.
2. Include a non-learned0.3 m/s forward primitive with the same decision/hold
   cadence and arrival wrapper in that continuation panel. Retain the current
   slower directional baseline as a separate engineering comparison. This
   avoids spending another panel solely rediscovering straight-line equivalence.
3. Replace arbitrary image change as a trusted-arrival candidate with repeatable
   local junction evidence and an explicitly uncertain relative-motion estimate.
   A named grid cell is evaluation bookkeeping, not directly observable place
   identity. Log ambiguous hypotheses; do not manufacture trusted graph edges.
4. Integrate physically visible beacon acquisition and observation-supported
   directed return with online memory. Then test predictive training versus
   online multi-step planning on independent layouts, with matched baselines,
   seed variation and full failure accounting. Hardware sensing/execution and
   bounded real-Go2 evidence remain separate required deliverables.

The immediate priority is continuous observation-driven branch/arrival/memory
integration, not another model-capacity sweep. The ultimate goal remains active
and incomplete; this result establishes only the first local transition.

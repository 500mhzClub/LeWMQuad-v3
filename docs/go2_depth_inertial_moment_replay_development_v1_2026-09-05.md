# Moment-aware fusion replay and prolonged weak-geometry stress V1

This separately named development successor preserves the completed fusion V1
and all original failed navigation trials. It does not run new physics, train a
model, access sealed material, qualify navigation, or certify hardware sensing.

## Estimator change

Replace the 100-ms mean-acceleration approximation with two time moments of five
causal 20-ms bin-average accelerations, individually transported into the current
reference and stripped of the initial gravity hypothesis. With T=0.1 s, the
endpoint-velocity correction weights are [0.002,0.006,0.010,0.014,0.018] s and
the displacement correction weights are [0.0018,0.0014,0.0010,0.0006,0.0002] s².
Retain the old directly constrained motion, distinct predicted weak components,
fault latching, and nonshrinking uncertainty proxies. Require an actual force
sample at the current endpoint. Do not claim the old proxy scales are validated.

Tests must cover equal-mean early/late acceleration, independently integrated
fine-time motion, 12 seconds of varying weak directions and rotating gravity,
old causal faults, and an intra-bin-jerk counterexample. Exact moment integration
under piecewise-constant bins does not reconstruct unmeasured within-bin motion.

## Fixed nominal replay

Use all 10 previously audited development histories and 3,610 frames specified
in the original fusion replay, without selecting successful frames. Feed only
original causal policy and depth/gyro observer records. Score physical references
after each trajectory's predictions exist. Keep the same descriptive position,
step, proxy-exceedance and budget checks. Independently report endpoint velocity
errors over 5 mm/s and interval acceleration errors over 0.02 m/s²; passing
position checks is not a substitute for these assumption diagnostics.

## Fixed live-observer stress population

Use the first 181 frames of the completed depth-floor-hold north trajectory.
The original RGB/depth/body/fast-gyro artifacts remain unchanged. Four explicitly
labelled synthetic sensor interventions are applied to copies:

1. Retain only depth columns 280–359 on ticks 80–139 inclusive (six seconds).
   Invalidated pixels have zero depth and false validity. No fake motion rank is
   injected: recompute surfaces, depth registration and gyro fusion through the
   actual observer, then restore original depth at tick 140.
2. The same depth intervention plus a +0.02-m/s² body-y accelerometer bias from
   measurement timestamp first_ns+8 s onward, including after depth recovery.
3. The same depth intervention plus a +0.2-m/s² body-y bias on that schedule.
4. Total depth dropout on ticks 80–82. Retain the first actual observer fault;
   do not reset or retry the estimator on later frames.

Bias applies by measurement time to maintain immutable overlapping histories.
RGB, gyro, commands, geometry and physical trajectories are unchanged. Nominal
calibration identifiers remain schema identities, not a claim that perturbed
values are native ideal measurements; the intervention is recorded separately.

These are sensor-degradation replays of one reused physical trajectory, not four
independent physical trials or newly executed navigation. Process through the
fixed endpoint unless a sensor-contract fault latches. After the first proxy-
budget violation, any further predictions are explicitly diagnostic only: an
online consumer would have to stop. Report the first required stop, actual fault,
complete versus truncated population, all physical errors and proxy exceedances.
Do not require these negative controls to pass nominal acceptance. Do not claim
that restoring depth relocalizes the accumulated reference.

## Integrity and interpretation

Bind the original fusion launch/result/verification, inherited and new recursive
source closure, all consumed original RGB/body/depth/gyro/physics artifacts, and
this protocol before launching one fresh output. Recheck at completion. The
verifier replays every nominal estimate and every degraded raw-sensor observer
record/fault exactly, and uses independent physical error/component/composition
checks. Source tests, protocol and completed artifacts become frozen on launch.

No parameter search, calibration, physical-navigation retry or JEPA claim is
part of this study. Next decisions must follow its actual limits: couple a
validated conditional state estimate to historical ray evidence and stopping,
address action-conditioned turn drift, and execute genuine navigation and
matched supervised/JEPA/multistep comparisons. Observer passes alone do not
complete discovery, return, memory, generalization or hardware requirements.

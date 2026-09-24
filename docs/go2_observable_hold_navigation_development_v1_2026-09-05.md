# Observable approach and active heading hold V1: fixed development experiment

## Scientific question and scope

Can sensor-grounded blocking/view constraints and active heading stabilization
advance the complete mission beyond the release-aware failures? This joint
engineering intervention does not isolate either component or claim learned
navigation/JEPA benefit. Preserve all earlier0/2 whole-task results, the south
beyond-wall target, vertical-rank failure and individual failed depth checks.

Exactly the original two north_dogleg/south_branch development layouts and seeds
retain the same geometry, episodic memory, marker detector, gait, gains, sensors,
clocks,360-s/36-leg budgets and independent physical mission metrics. Method:
`observable_hold`. Fresh output:
`.generated/go2_observable_hold_navigation_development_v1_attempt_001`.
No retry, source edit or alternate coefficient after launch. No sealed or
independent evaluation, hardware experiment or new sensing configuration occurs.

## Blocking and floor-view constraints

Fit the actually observed ground-like depth points: normals aligned within0.97
to the gyro-transported initial gravity estimate, points below-0.15 m, at least100
supports, nondegenerate two-dimensional extent, maximum fitted-plane residual
<=0.01 m, fitted normal aligned within0.97 to gravity. Missing support remains
unknown; no fixed height or zero vertical translation is substituted.

Intersect the calibrated central ray at row400 with this measured floor plane.
Use its forward body coordinate plus0.15 m as a view-dependent front standoff,
leaving the lower80 image rows for possible floor observations. The final
standoff is at least nominal turn radius plus0.15 m. This view margin is a
declared development constraint, not a proof of100 future floor correspondences,
calibrated uncertainty, continuous collision clearance or hardware safety.

For each observed body-height surface with forward normal component>0.8 and
actual lateral support intersecting the nominal transverse footprint, cap the
forward target by observed plane-intersection distance minus that standoff.
Do this even when the original target comes from a corner hypothesis. A cap
<=0.15 m at selection produces no feasible target, not a fabricated arrival.
No unseen gap or unverified opening extent is declared traversable.

During TRAVERSING/BRAKING, recheck the current constraint every decision and
tighten the target if the cap is more than0.01 m nearer. Never expand it when
support disappears. A newly negative cap stops with an explicit failure.
Record every target update and its observed surface/floor evidence. This is an
observed-blocker/view guard, not a claim that all unobserved swept space is free.
The existing measured braking, sampled nominal turning support, sensor-rank
failure, contacts and independent native-physics outcome remain in force.

Read-only predecessor replay91063 found a2.23455-m new cap for the actual south
frame327 target4.13436 m, before its directly observed wall at3.54542 m. It uses
1,281 actual floor points and1.31086-m view standoff. This validates the static
constraint on that observation, not a counterfactual physical trajectory.

## Active heading contract, explicitly distinct from zero release

Keep a continuous PI servo with gains1.5 and0.4/s, integral cap0.12 rad/s and
yaw cap0.35 rad/s, saturation anti-windup and no reset on tolerance entry or
sign crossing. Alignment hands off the same observed target/integral to later
operational holds; it does not claim zero-command rest. Require tighter handoff
readiness of absolute error<=0.01 rad and rate<=0.02 rad/s for0.3 s, still within
the original outer0.02-rad/0.1-rad/s criterion and12-s total deadline. These
stricter readiness values were introduced after a prelaunch synthetic biased-
plant test exposed overshoot after a boundary-only handoff; that failure remains
part of the development chronology.

After at least one measured local arrival, operational HOLD_ALIGN/HOLD_SCAN/
HOLD_TRAVERSE and subsequent traversal warmup may use measured heading feedback.
Fresh holds anchor the current observed heading. Completed alignment transfers
its existing servo without discarding history. Normal traversal and active scan
retain their own commands. Every added nonzero holding-yaw request requires the
same current sampled nominal turn-volume support; missing support fails and
requests zero. Initial unobserved holds remain zero. Mission terminal commands,
invalid-sensor commands and the actual0.5-s physical terminal release remain zero.

## Full evidence and interpretation

Unit tests include floor-height/FOV dependence, missing/degenerate floor,
partial blockers, near-corner contradiction, monotonic target tightening,
biased heading hold/hand-off, no response, invalid input and fresh-operator
installation. Collection and full physical-audit function bodies remain
identical; explicit controller imports differ. Replay all observations/commands,
retained depth/native identities, target constraints, active holding commands,
memory, contacts and complete physical mission metrics.

Report every local failure, alignment/holding result, lost motion component,
marker discovery, actual home return, false claims and depth/motion check.
Full audit PASS means faithful replay, not successful or safe navigation. If
motion becomes unobservable, add independently informative deployment-valid
evidence with explicit uncertainty or change the observation action; never lower
the rank test or assume zero motion. If operational holding fails, diagnose its
actual dynamics rather than relabel a nonzero request as zero release.

The ultimate goal still requires reliable complete exploration/discovery/return,
matched online-memory and supervised/JEPA comparisons, genuine multi-step online
rollout tests, independent layouts/seeds, sensor robustness and bounded real Go2
evidence when available. This study is not a replacement for those requirements.

# Depth-floor / active-hold integration V1: fixed full-mission development run

The observable-hold predecessor failed both first scan initializations because
its old gravity/foot estimator required an all-zero applied-command history,
incompatible with genuine nonzero holding corrections. Preserve both failures
and all source/command/result bytes. This is a distinct integration intervention,
not a retry or a relaxed measurement guard.

## Fixed population and retained controls

Exactly the original north_dogleg and south_branch development layouts/seeds use
episodic memory, unchanged geometry, marker, gait, gains, camera/sensor paths,
clocks,360-s/36-leg budgets and native-physics whole-mission scoring. Method
`depth_floor_hold`; output:
`.generated/go2_depth_floor_hold_navigation_development_v1_attempt_001`.
No source edits, retries, coefficient changes or alternate attempts after launch.
This is not independent/held-out evaluation, learned navigation, JEPA benefit,
new hardware sensing or deployment qualification.

Retain all observed blocker/view target limits, per-decision tightening, rank
and missing-motion stops, sampled nominal turn checks and active heading-hold
commands. Holding alignment readiness remains0.01 rad/0.02 rad/s/0.3 s inside
the original outer tolerance,12-s total budget. Actual operational corrections
remain logged as nonzero; mission terminal and fault release remain zero.

## Changed projection input, without fictitious history

Install `ObservedDepthFloor` before each newly constructed traversal and scan's
first observation. It supplies the current depth-fitted visual floor normal and
body-to-plane distance already computed by the observable-approach front end.
It does not bootstrap a new gravity estimate from command intent, infer that a
foot is in contact, or overwrite applied-command history. Initial whole-mission
gravity initialization is retained; the current depth stream and its timestamp,
RGB identity, gyro-relative transport and support requirements remain enforced.

The adapter requires a current consecutive observation, valid policy packet,
at least100 floor supports, finite unit normal, fitted-plane maximum residual
<=0.01 m and body-plane height in[0.1,0.6] m. Missing/stale/invalid planes latch
failure; no old height or zero command fills the gap. Output identifies
`current_observed_depth_floor`, records actual fit evidence, and keeps
`ground_plane_qualified=False`. The native visual/collision floor offset remains
explicit; projecting RGB floor pixels onto the observed visual plane is not a
collision-clearance or hardware-calibration certificate.

Synthetic tests cover nonzero holding histories with unchanged input values,
ground-envelope integration, stale/missing/invalid supports and replacement only
before first use. Read-only predecessor frame195 replay45573 obtains a valid
unqualified forward proposal using1,263 actual floor points and measured height
0.316001 m while retaining actual nonzero commands. That only validates the
adapter on the failed input; physical improvement requires the fresh missions.

## Full evidence and next decision

Collection and core full-physical-audit function bodies remain identical; the
explicit controller import changes. Retain/replay every RGB/depth/body/gyro
packet, relative state, target constraint, holding command, gain/contact,
provisional memory and complete mission outcome. Report all failures, including
before alignment or discovery. Full audit PASS is fidelity, not physical success.
No first scan or alignment success substitutes for full discovery and return.

If later target/observability, turning, branch choice or memory fails, diagnose
the actual trace and implement a separately named successor. Preserve all
predecessor results. Matched memory, supervised/JEPA predictive-training and
genuine multi-step rollout experiments, independent layouts/seeds, sensing
robustness and bounded real Go2 evidence remain required for the final goal.

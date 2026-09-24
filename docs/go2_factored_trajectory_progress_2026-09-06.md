# Factored evidence and action-conditioned trajectory interface

Implemented two new development modules without changing the recorded
predecessor consumers or trial. The interface now distinguishes measured
non-floor conflicts, physical ground relations and positive visibility, and
evaluates time-indexed articulated predictions with explicit between-sample
allowances. It still does not approve navigation or establish a learned gait,
validated motion/stopping model, ground support or complete mission. Maze 0/2
and the negative JEPA comparisons remain unchanged.

## Evidence semantics now implemented

`lewm/factored_configuration_evidence_development.py` takes a fresh continuous
owner and an explicitly supplied current-body-relative pose/joint configuration.
It retains the finite setup-region partition, global uncertainty, current-camera
common-pose cancellation and conservative transport into historical observations.
Every full physical query is checked in every retained view; every required
residual needs positive complete-box evidence. Depth hashes, plane hypotheses,
physical primitive identities and observation clocks remain bound.

Non-floor queries use the bound observed plane family without treating uncertain
physical ground intersection as a non-floor obstacle. If no plane hypothesis
exists, all near returns remain eligible conflicts. A wall still vetoes even a
false large setup-region claim. Ground penetration cannot be cleared by a
positive result in the separate non-floor channel.

Positive ground separation requires at least one fully covered observed
footprint, not a union of partial views. The new module retains covered
penetration, possible intersection, separation and exact-foot candidates as
different facts. Broad historical intervals do not refute a compatible narrow
positive interval. Covered plane hypotheses are compared at common physical-box
vertices, retaining every pair with disjoint residual intervals. This is a
necessary compatibility check, **not proof of common surface identity**: the
module explicitly makes no common-surface or calibrated-error claim and never
averages planes. Feet remain contact candidates only. The existing continuous
owner and its original continuation guard are unchanged; the new interface is
not yet installed as an autonomous navigation controller.

The final source accounts separately for virtual comparison vertices: each
coordinate support can move by at most the supplied physical point error e,
so the corresponding box corner uses sqrt(3)*e. Historical transport is computed
at those virtual vertices themselves, rather than reusing a possibly smaller
physical-primitive lever arm. The physical query keeps its own original error.

## Action-conditioned baseline and interval checks now implemented

`lewm/articulated_trajectory_evidence_development.py` validates a fresh
deployment-valid RGB/body/command packet, matches episode/time/current joints to
the owner and binds the joint/command histories. For an explicit sequence of
100-ms commands it predicts the slew-limited ideal-command SE(2) response in the
observed gravity tangent, preserving initial tilt. Joints use measured-velocity
persistence. These are falsifiable non-learned baselines, not observed velocity,
learned locomotion, a physics rollout or qualified stopping dynamics. In
particular a zero command is not evidence of physical rest.

The evidence API also accepts externally predicted pose/joint arrays with an
explicit model identifier. It requires an exact current-state anchor, strictly
increasing integer times, proper rotations, endpoint point-error assumptions
and per-interval **physical-point** speed assumptions. A base-speed cap is not a
substitute for the latter: feet and other articulated points can move faster.

For an interval of duration dt with physical point speed at most V, every point
is within V*dt/2 of its nearest actual endpoint. If that endpoint prediction has
point error e, the predicted primitive expanded by e+V*dt/2 contains that point.
The implementation takes the larger adjacent-interval allowance at shared
nodes and sends every expanded pose/joint configuration through the factored
consumer, with the entire forecast horizon required for setup validity. This
conditional enclosure argument does not validate e, V or the model; all swept
motion, stopping, support and action qualifications remain false.

## Recorded interface check, not executed trajectories

The separately declared diagnostic replays the same 15 startup/tail observations
and exactly reproduces 12 startup decisions and 15 relative estimates. It checks
the new consumer at the two previously diagnosed offsets, not a newly tuned
offset sweep. Original results and all their source identities remain intact.

| Tangent offset | Original conditional count | New non-floor count | New observed separated count | Cleared residuals |
| --- | ---: | ---: | ---: | ---: |
| 0.75 m | 20/27 | 24/27 | 3/27 | 5/8 |
| 1.00 m | 19/27 | 23/27 | 9/27 | 11/15 |

No new non-floor conflict or incompatible/penetrating covered-ground witness
appears in these two queries. Missing volume and support remain unresolved;
the component counts are not a changed success count or executable route.

The final 2.9-s packet also produces three explicit four-tick nominal plans:
zero, two 0.08-m/s forward targets followed by zero, and two 0.35-rad/s yaw
targets followed by zero. The declared interface-only endpoint errors are
[0, .05, .05, .05, .05] m, with 2-m/s physical-point speed assumptions in every
interval. These are **unvalidated hypotheses**, not new acceptance limits.
They produce node expansion radii [.10, .15, .15, .15, .15] m, through 3.3 s,
with the original setup expiry still 3.5 s. All nodes obtain conditional
non-floor clearance from the available evidence/setup condition, not newly
measured future motion. The forward baseline predicts about 16 mm travel.
The recording ends at the forecast anchor: no actual future sample validates
any of the three plans. All three complete trajectory outputs and both static
factored outputs match the reference backend exactly.

## Verification and custody

- Session 28436: 35 focused tests passed; one fixture failed while attempting
  to mutate an immutable observation mapping. The fixture was changed to inject
  a replaced retained entry; no evidence rule or acceptance threshold changed.
- Session 10746: all 36 focused tests passed in 24.03 s.
- Diagnostic 16729: exit 0; its preliminary result remains retained in
  [the pre-correction report](go2_factored_trajectory_recorded_diagnostic_result_2026-09-06.json),
  SHA-256 `451647279f1e139d686f05c7095cd0e69f93bc2ccac16aa44768a07f3d64dd29`.
  It binds the earlier development source, not the final corrected source.
- Expanded regression 72797: 1,785 tests across 144 explicit files passed in
  141.50 s. Only after it and the diagnostic were terminal did source review
  tighten the virtual-corner uncertainty accounting described above.
- Session 91859: all 37 final focused tests passed in 25.42 s, including the
  added rotated virtual-corner/physical-error distinction.
- Diagnostic 24363: exit 0, all recorded component counts unchanged and all
  reference comparisons exact with corrected source. The
  [final machine-readable report](go2_factored_trajectory_recorded_diagnostic_final_result_2026-09-06.json)
  has SHA-256 `7710a2398f0d120ba3ae3dbbbab6f0481aa371021e45397ade73dd1c36c6e749`.
- Final expanded regression 52622: **1,786 tests across 144 explicit files
  passed** in 141.98 s. No tested source changed during any live test or
  diagnostic. All listed handles are terminal.

Coverage includes wide-versus-narrow ground evidence, incompatible tilted-plane
residuals, certain penetration, missing/partial coverage, exact foot identities,
actual wall vetoes, absent planes, expiry, raw/plane identity corruption,
compiled/reference equality, command slew and gravity tangency, joint evolution,
all-node/full-horizon checking, interval expansion, stale/privileged/incorrectly
anchored packets and malformed motion/error assumptions.

The diagnostic verifies the original three trial identities,
385-source/inherited-input/74-artifact/16-native witnesses, the exact preceding
attribution result, and all 20 explicit development paths before/after. No
frozen source/protocol/result, sealed material, training or physics was changed.

## Next: empirical motion validation, then the complete mission

The concrete next-run specification is
[action-conditioned motion identification and validation](go2_action_conditioned_motion_validation_next_execution_2026-09-06.md).
It declares two distinct forward/turn/braking schedules and a new independently
checked calibration-arena condition. Its collector/auditor are not implemented
or launched yet; the current trajectory API tests do not substitute for it.

The next implementation is a fresh bounded motion collector and independent raw
auditor, using the same observer continuously from admission through startup,
forward motion, turns and stopping. Declare any longer initial-region duration
in the NEW protocol and independently verify its geometry/setup; do not extend
the old trial's expiry or call its final state a resumable simulator. Preserve
the old continuation guard's behavior as a comparator; integration of the new
factored guard must be explicit in new source and tested against actual walls,
floor penetration, unknown volume and stale observations before launch.

Collect sustained motion and braking, not only another successful startup turn.
Use recorded causal commands, RGB-D, body/joint/fast-gyro histories and predicted
trajectories as runtime-side records. Keep native body/joint/contact trajectories
evaluation-only. Measure per-horizon body pose, articulated primitive/foot
point errors, physical point speeds, stopping displacement, intersample envelope
violations, contacts and complete-loop latency. Use separate identification and
validation executions and retain failures; do not fit and validate on the same
trace or call empirical coverage a universal safety bound. The illustrative
errors in this interface diagnostic must not become validated limits by reuse.

Then connect the model/evidence interface to a fresh continuous discovery,
marker and return mission. Complete matched geometry/supervised/JEPA training,
one-step/genuine-multistep online rollout, memory ablations, independent layouts,
seeds and robustness, and bounded hardware evidence when available. These
requirements remain part of the full active goal, not optional follow-up polish.

# Paired arrival experiment launched; recorded dynamics coverage

The distinct inner-arrival simulation is launched under the
[frozen paired protocol](go2_inner_arrival_room_return_v1_2026-09-06.md).
Results and independent audit are pending. The latest completed physical
assay remains 0/3 full returns; no launch or passing source test changes that.
The ultimate RGB-plus-sensor JEPA/novel-maze/memory/deployment goal is active.

## What changed and what did not

New collector and auditor use the existing scene/session, gait, sensor pipeline,
multi-reference tracker and nominal pulse table. Only the consistent internal
position-arrival region changes from6cm to4cm. The auditor imports the original
native hold/winding score functions unchanged and additionally verifies exact
750-sample setup dynamics and first-frame RGB against the completed baseline.
Both signed routes and the low-friction challenge remain in the fixed batch.

The13 new physical-pairing tests check identical substrate identities, unchanged
external scoring, preserved budgets, setup/appearance mismatch rejection and
allowing trajectories to differ after setup. Combined with13 existing inner
controller tests,26 focused tests pass. The full205-file regression passes
**2,597 tests in202.82s**, handle46252 exit0. An initial test referred to the
observer by the wrong attribute name; that test-only error was corrected before
source freeze and the successful tests/preflight. No runtime threshold changed.

Read-only preflight passes with679 source paths,59,661 ordinary input bindings
and14,217 external baseline artifact/identity bindings (handle41756 exit0).
The exclusive output did not exist at preflight. Collection handle56322 runs:

`PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=2 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -m scripts.run_go2_inner_arrival_room_return_v1`

Launch SHA256: `53ec5a2a04831ddbc8ae3932038ce90a5f12c797d38087bb6fb95a3e8778667a`.
The live handle has reached nominal-left tick400 with one completed stage.
The first RGB PNG exactly matches baseline SHA256
`83c2e74cf011408ec8014cbeff42d4781b1743ff26f1c35e1ae29a733994c8d9`.
This early image check is not the pending full setup-prefix or native-hold audit.

Output is the explicitly owned external development root ending
`go2_inner_arrival_room_return_v1_attempt_001`. Preserve40GiB reserve and all
partial artifacts. Do not restart because a poll times out. Only after the
collector is terminal with complete artifacts should the frozen module
`scripts.audit_go2_inner_arrival_room_return_v1` start once. A controller
candidate is not a scored return. Independent starts/layouts are still needed
after this paired development comparison.

## Recorded response errors guide the learning/data work

The read-only [pulse residual diagnostic](go2_intent_pulse_residual_diagnostic_2026-09-06.json)
binds sources and the completed intent-return inputs before/after computation.
It reconstructs logged response residuals from measured visual displacement,
gyro yaw and the fixed table; it does not fit a model or read native pose.
Result SHA256: `6efdbac33a2734971364e05cc5fcb233ff1308c6fc8f1244822b3e351e018735`.
Script SHA256: `56b90fda4604f37145d4e3be2edb9fc701571a40770aee24eeb2dbeb83b6799f`.
Handle90314 completed with exit0.

| Recorded condition | Started / completed responses | Planar residual RMSE | Yaw residual RMSE |
| --- | ---: | ---: | ---: |
| Nominal left |90 /90|8.63mm|.0206rad|
| Nominal right |65 /64|9.11mm|.0166rad|
| Low friction |18 /18|41.74mm|.0349rad|

The uncompleted right response is retained explicitly: final-leg pulse11,
started tick1804, negative-yaw5ticks. Tracking fails before the settled response
is logged. It is not assigned zero error or treated as a completed endpoint.
All172 completed response records report20 brake ticks. The diagnostic's
conservative endpoint-timing flag remains false: it reads response logs and
does not independently join exact action-tape/native endpoint labels. That
separate target derivation is required for training; longer-settling cases,
if present in future data, must not be silently merged with brake20 targets.

For5-tick forward pulses, mean observed start-body XY is approximately
(7.09,-.05)cm left, (6.81,-.10)cm right and (3.20,2.58)cm low friction,
versus table (6.79,-.18)cm. Low-friction forward planar RMSE is4.95cm across
12 correlated pulses. For3 low-friction short negative turns, mean yaw is
-.0998rad versus table-.0464rad. Aggregate RMSE also depends on the different
action mixtures; none of these closed-loop comparisons isolates a causal
friction coefficient effect or establishes an error bound.

There are **zero2-tick forward pulses in all three streams**. Low friction
contains no positive-yaw pulses of either duration. More room-return samples
alone will not guarantee action/state coverage: the fixed planner chooses its
own narrow distribution. These three runs are not independent maze layouts.

## Next actions

1. Poll the existing collector; preserve every terminal outcome. On complete
   acquisition, run the frozen auditor and compare all holds, tracking failures,
   pulses, path/time and compute against the matched baseline. Do not relax
   success or drop low friction if the tighter region fails.
2. Prospectively collect a balanced command-duration assay across body histories,
   support conditions and independent scenes, including the missing short
   forward and positive-turn cells. Keep sensor-only inference boundaries and
   retain stopped/unobserved outcomes with explicit censoring. Establish
   backward/arc controllability before giving those commands to a planner.
3. Integrate the existing pulse-timed input/target/loss interface into an actual
   dataset/sampler/runner with frozen independent layout roles, identical sample
   schedules and multiple seeds across direct, supervised-rollout and JEPA arms.
   Compare task-relevant prediction and decisions, not latent loss alone.
4. Continue observed branching, useful memory and physical backtracking, matched
   online-rollout/memory studies, realistic sensor/timing/body-sweep validation
   and bounded hardware when available. None is replaced by this room assay.

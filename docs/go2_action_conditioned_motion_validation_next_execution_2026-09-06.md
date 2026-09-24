# Next execution: identify motion and challenge it on a separate run

Status: A completed; its sensor-only response fit was frozen before B. The single
[B validation run](go2_action_motion_validation_development_v1_result_2026-09-06.md)
has now been executed and independently audited. It stopped after24/28 targets
at5.3s on an exhausted pose-error proxy budget, not a physical collision. The
three real zero-tail ticks ran through5.6s. All96 issued forecast horizons are
scorable but the complete schedule failed. Preserve both runs, fit and all
sources/results; do not retry B or refit to its outcomes. The fixed schedules
below remain the original experiment specification, not a pending launch.
The [post-B whole-mission plan](go2_post_B_observability_and_whole_mission_plan_2026-09-06.md)
now prioritizes RGB-D motion observability and continuous full-task integration,
not more static probes or an increased uncertainty threshold. Fullgoal active.

## Fixed development scope

Use the existing bounded four-wall/aligned-floor Go2 simulator, unchanged learned
low-level gait and gains, deployment-valid RGB-D/body/applied-command/fast-gyro
interfaces and evaluation-only native geometry/pose/contact records. Use CPU
physics and software rendering unless a later explicit protocol changes this.
No sealed material, predecessor resume, hardware actuation or JEPA training.

Declare a NEW initial-body non-floor-clear region [-1.25,1.25]^3 m, static through
8.0 s simulation time, with the same initial velocity condition 0 +/- 0.02 m/s.
Independently check its full geometry against every native static obstacle and
verify the native floor/foot identities and loaded initial support. Record the
new condition and checks under the new protocol identity. This larger/longer
calibration-arena condition is not supplied by a depth observation and must not
be installed as a general maze/deployment assumption. It does not alter the
old [-1,1]^3 region or 3.5-s expiry. Reject setup if the new checks fail.

Keep one continuous observer from the 1.5-s admission through the existing
startup observation maneuver, three-frame measured zero tail and all subsequent
motions. The terminal startup controller must never be called again. Do not
reset position, history or uncertainty at transition. A failed observer or
exhausted uncertainty budget remains failed; record an actually executed stop
tail when the simulator and physical-stop rules permit it, not frozen rest.

## Two predetermined runs, not a fitted-and-tested-on-one-trace result

Use new explicit output directories and record distinct scene/seed identities.
Do not launch the validation run until an identification model and its input,
normalization and error-reporting rules have been frozen from run A alone.
The B command schedule below is already fixed and must not be selected from A's
favourable outcomes. If A fails, preserve its partial trace and diagnose it;
do not re-label B as a replacement A or silently retry either output.

| Segment after completed handoff | Run A: identification | Run B: validation | Duration |
| --- | --- | --- | ---: |
| Forward initiation/sustain | [0.12, 0, 0] | [0.10, 0, 0] | 0.6 s |
| Brake/hold | [0, 0, 0] | [0, 0, 0] | 0.4 s |
| Turn | [0, 0, +0.35] | [0, 0, -0.35] | 0.4 s |
| Brake/hold | [0, 0, 0] | [0, 0, 0] | 0.4 s |
| Forward after turn | [0.15, 0, 0] | [0.08, 0, 0] | 0.6 s |
| Final brake/hold | [0, 0, 0] | [0, 0, 0] | 0.4 s |

These are high-level command targets, not measured velocities. Preserve actual
slew-limited commands and the low-level gait's joint responses. Every segment
has a recorded complete horizon or an explicit truncated/failure status.
The predetermined calibration schedule is not a learned navigation policy.

## Admission of development commands and stopping

Use the independently checked static starting region and the existing all-joint
body-radius envelope plus 4-cm padding, declared 0.3-m/s base-speed assumption,
0.4-s command-plus-stop horizon and unchanged fusion budget. Do not use the
unvalidated trajectory interface errors (.05 m / 2 m/s) as acceptance limits.
Before each command require the full prospective envelope to fit the current
new region through its horizon. Retain actual observed non-floor contradictions
and ground penetration; do not erase a native contact or restart failed state.

Keep 500-Hz evaluator checks for actual speed, body state, every physical
primitive's padded region containment, disallowed obstacle contact and exact
non-foot ground contact. A speed cap or contact failure is a result, not evidence
that the open-loop model was safe. Foot material/compliance/support validity is
still simulator-conditional; the run cannot certify hardware contact dynamics.
Record final native geometry/gains and actual final-window linear/angular speed.

## Prediction and error accounting

At every eligible decision save the causal current packet identity, model ID,
candidate applied-command sequence, predicted body pose and all 12 joints through
0.1/0.2/0.3/0.4-s horizons, including braking switches. Keep actual future/native
information out of prediction generation. The nominal baseline currently uses
ideal command motion and joint-velocity persistence; challenge it rather than
assuming zero target means zero physical displacement.

Run A may fit a small explicitly identified response model (for example a
command/state-conditioned linear dynamics baseline) using only A. Freeze all
fit choices before B. Report the original non-learned baseline alongside any
fit, including per-action and per-horizon failures; do not introduce a model
and report only its best aggregate. These two bounded traces are development
evidence, not enough to claim universal error calibration or JEPA advantage.

The independent auditor must reconstruct sensor histories, commands, native
foot/contact attribution, stopping and all predicted-input identities. Compare
predictions against actual future body pose, joint/foot/primitive trajectories;
report endpoint errors and sampled physical-point speeds, explicitly retaining
the distinction between a finite-difference measurement and a continuous speed
bound. Check full recorded prospective envelopes and report truncated horizons
separately. Include complete sequential wall-clock loop latency; CPU simulation
time at 10 Hz is not deployment real-time evidence.

## Next after these runs

Integrate the tested motion/evidence interface into a fresh complete
discovery/marker/return mission with explicit support/unknown-space handling.
Then perform matched geometry/supervised/JEPA predictive-training and actual
multistep-online-rollout comparisons, memory ablations, independently generated
layouts/seeds/robustness and bounded real-platform work when access permits.
Keep the present 0/2 whole-mission result and all learning negatives unchanged.

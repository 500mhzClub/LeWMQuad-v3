# Observed initial alignment before continuous branching: fixed development protocol

The predecessor continuous panel completed16 trials with corrected full audit:
every method, including matched fixed-forward, succeeded2/4. The remaining
fixtures crossed successfully but contacted the south wall during scanning.
This successor tests one proposed remedy inside the same two-leg task: align
to a fresh observed exit bearing before the first translation. It does not
change or rerun predecessor evidence or claim that bearing is a corridor center.

## Population, models and preserved endpoints

Sixteen trials: the same four local1.2 m corner/tee geometries and coupled
±0.06 m lateral/±0.1 rad heading conditions, each with fixed-forward,
direct-only/direct, supervised recurrent and JEPA recurrent prediction.
Seeds2026092900–2026092903 are paired across methods. These remain reused
development fixtures, not independent final mazes. The three learned arms use
their exact previous three-seed final ensembles; no training, checkpoint
selection, action-bank or learned-input change. Fixed-forward retains the same
0.3 m/s command and500 ms hold cadence under the traversal wrapper.

All previous two-leg physical criteria, provisional ledgers, scan/side-branch
selection, later coarse alignment, fresh forward reobservation and zero-release
requirements remain unchanged. Initial alignment adds an operator and observation/
hold time, not a relaxed crossing or contact criterion. Comparison with the
predecessor is the whole added operator sequence, not an isolated effect of the
new tolerance or proof that elapsed time/initial gait state has no effect.

## Added initial operator

After the same1.5 s zero-command settling, process four actual RGB/body frames
at100 ms cadence with the same nominal gravity/floor-extension observer. Choose
the closest-to-forward observed proposal within0.35 rad, tie-breaking by greater
support. No proposal gives FAILED_INITIAL_NO_EXIT. No map, true pose/velocity,
known cell center, teacher route or intended destination is an input.

Transport that observed ray into the uninterrupted initial-body gyro reference.
On the following command frame, align using the same proportional gain1.5,
0.35 rad/s cap,0.1 rad/s measured-rate tolerance,0.3 s dwell and12 s deadline as
the earlier bearing controller, but with heading tolerance0.02 rad instead of
0.08 rad. The coarse tolerance is nearly the entire0.0873 rad initial correction
seen in a failed predecessor fixture. The new tolerance is intended to limit
heading-only lateral error to roughly3 cm over1.5 m nominal straight motion;
this is a small-angle design motivation, not a calibrated physical error bound.
Slip, bearing bias, lateral offset, translation during turning and future gait
sweep remain unresolved. Source-comparison tests restrict this fine-alignment
kernel change to the heading tolerance.

Alignment success triggers an explicit zero hold of at least1.5 s; processing
the transition on the next100 ms frame gives1.6 s to the first traversal start.
The unchanged traversal instance then acquires its own four fresh frames and
must independently propose a forward exit. The old ray does not substitute
for those frames. There is no initial traversal command before that warmup.
One global gyro reference/history spans initial observation, alignment, hold
and the full subsequent task, with no physical reset or teleport.

The global80 s controller budget is unchanged and now includes the added
initial operator. At most801 post-settle observations and final0.5 s zero
release are permitted. CPU Genesis, frozen gait kp=20/kv=0.5, ideal body sensing
and live500 Hz virtual gyro are unchanged. Native contact/stability stops end
physics immediately; sensor failures latch and receive bounded zero release
unless native stopping interrupts it. Retain every outcome and partial trace.

## Evaluation and interpretation

The primary endpoint is the unchanged continuous two-leg integration success:
both actual candidate-time/final crossing and release checks, completed scan,
observed side selection, complete controller and no native stop/contact or
sensor fault. Evaluate the first leg at its own actual500 ms post-terminal
zero window, not after the later scan. No trusted graph edge/place/beacon is
created by these development endpoints.

Additionally report evaluation-only actual initial-phase translation, observed
target heading versus true relative heading at first-traversal start, arrival
position, scan drift and scan contacts. Actual simulator pose enters only this
post-execution reduction. Smaller gyro error alone is not success: the intended
test is improved useful arrival and continuous-task completion. If alignment
does not solve the contacts, retain that negative result rather than expanding
the margin, changing the floor mask or retrying this bound panel.

Keep matched fixed-forward in every fixture. Scanning and alignment are still
hand-controlled and outside the frozen learned five-action bank; this panel
does not isolate a JEPA contribution to those maneuvers or test multi-step
latent planning. Report all four fixture outcomes per method and exact repeated
trajectories, not confidence intervals from correlated frames/duplicate physics.

## Evidence custody and next task

Fresh root: `.generated/go2_initially_aligned_continuation_development_v1_attempt_001`.
Bind recursive sources/tests/protocol, exact predecessor launch/result/corrected
audit and reader-correction witness, frozen models, URDF and gait before physics.
The collector preserves actual images, both sensor histories/streams, native
contacts, poses/joints, all commands/decisions and provisional ledgers. The full
auditor uses the already-tested806-frame continuation reader, reconstructs raw
sensing/cameras/commands, replays every scientific field, and recomputes the
unchanged leg/task metrics plus initial-alignment diagnostics. Only the two
explicit learned inference timing fields are excluded from equality.

No tuning, in-place retry, erased failure or predecessor-source edit. A useful
next step must remain continuous observation-driven navigation: measured visual
centering/repositioning if needed, uncertain online place/branch hypotheses,
actual beacon acquisition and directed return. Independent maze comparisons,
predictive-training versus online-rollout contrasts, robustness and real-Go2
evidence remain required; this local intervention does not redefine completion.

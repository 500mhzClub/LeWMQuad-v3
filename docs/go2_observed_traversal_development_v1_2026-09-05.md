# Observed one-transition traversal: fixed development protocol

This is the first integration of an actual RGB-proposed direction, repeated
local control and an explicitly provisional sensory arrival. It is not an
end-to-end learned navigation policy. Learned consequence prediction is one
component of a hand-designed controller; the exit proposal, ranking objective,
arrival rule and ledger are not learned. No maze, place-recognition, long-horizon
planning or real-hardware qualification follows from this experiment.

## Fixed population and comparisons

Twenty trials: four local fixtures, each with always-stop, directional gait,
direct-only/direct prediction, supervised recurrent prediction, and JEPA
recurrent prediction. Fixtures use 1.2 m corridors and a 1.28 m cell pitch, with
a source space connected east into either a corner or a tee. Each motif has
two coupled starting conditions: lateral offset/heading (-0.06 m, -0.1 rad) or
(+0.06 m, +0.1 rad). These are four engineering fixtures, not four independent
full mazes, nor a factorial separation of heading and lateral offset. Seeds
2026092700 through 2026092703 are paired across methods. There is no narrow-wall
clearance claim; the preceding 0.9 m dead-end contact failures remain unresolved.

The three learned arms reuse the three-seed final ensembles from the completed
temporal RGB/body study, with no fitting, checkpoint selection or replacement
by the newer coverage-study models. All learned arms use the same five candidate
actions, half-second consequence horizon and ranking cost: ten times mean
predicted contact probability plus distance from mean predicted translation to
the transported 0.8 m directional cue. The cue is a direction preference, not
a known destination point. The recurrent heads use their first transition only.
This panel compares integrated local execution and cannot identify the value of
multi-step planning or separately establish both training and inference effects.

Directional gait requests 0.2 m/s forward with gyro direction feedback updated
every 100 ms; learned arms select every 500 ms and hold their selected command.
It is an engineering baseline, not a compute/cadence-matched learned-policy arm.
All methods share the high-rate orientation-based progress proxy and arrival
rule. The frozen learned adapter retains its original 50 Hz cue transport and
unchanged model inputs; directional gait uses 500 Hz transport. This asymmetry
is explicit and must not be interpreted as an isolated learning effect.

## Runtime observation and execution boundary

CPU Genesis, frozen gait checkpoint, effective gains kp=20 and kv=0.5. After
1.5 s zero-command settling, every 100 ms the controller consumes the actual
mounted RGB frame, causal body histories and a separate live ideal 500 Hz gyro
history. The existing RGB/body learned tensors are unchanged. Body geometry is
the hash-checked Go2 URDF's 27 articulated collision primitives evaluated at
current sensed joints. No world pose, true velocity, scene graph, destination
coordinate, opening plane, teacher route, hidden beacon or future observation
enters the controller. Scene construction/evaluation retains first-edge geometry
but never calls the oracle route follower.

Three zero-command warmup ticks provide four real RGB observations at control
start. The unchanged floor-extension observer proposes candidate bearings.
Choose the closest-to-forward candidate within 0.35 rad, tie-breaking by greater
support. No candidate ends as FAILED_NO_EXIT without creating a traversal record.
The proposal is not a qualified exit and is never converted to a trusted edge.

Progress integrates actual applied commands over each ending 100 ms interval,
rotated using the high-rate relative orientation. This is commanded-distance
accumulation, not measured translation; slip or lack of execution can fool it.
Required nominal progress is max(0.8 m, current articulated span along travel
direction + 0.35 m), recalculated from body joints. Neither this span nor its
margin is a calibrated swept-volume clearance guarantee.

Visual change is the XOR fraction between current and initial bottom-connected
floor masks on the fixed 8-pixel grid. At a half-second decision boundary,
progress above the requirement and three successive changes >=0.10 initiate
zero-command braking. Otherwise progress >=1.4 m gives FAILED_NO_VISUAL_CHANGE,
or 12 s traversal gives FAILED_TIMEOUT. Appearance change is not place identity.

Braking requires at least 0.5 s and 0.3 s continuously quiet proxy readings:
the last five valid 50 Hz gyro norms <=0.15 rad/s and valid joint-speed RMS
<=1 rad/s. Unavailable readings cannot establish quietness.
At completion, current progress and visual change are rechecked; insufficient
progress gives FAILED_PROGRESS, insufficient change FAILED_NO_VISUAL_CHANGE.
Two seconds without settling gives FAILED_SETTLING. A successful rule emits
only ARRIVAL_CANDIDATE with current image identity, no place identity, zero
trusted graph edges and all arrival/traversal qualification flags false.
Quiet joints/gyro do not establish zero translational velocity.

The controller has at most 144 observations after settling (ticks 0 through
143), followed by five explicit zero-command release ticks. Native contact or
body-instability stops terminate simulation immediately, with no release after
that stop. A sensor-contract fault latches FAILED_SENSOR and receives bounded
zero release unless native stopping interrupts it. Retain every failure and
partial trace, including contact during release after a nominal arrival.

Native stops are privileged evaluation/simulation protection, not a deployment
safety mechanism. The inherited contact filter permits ground contacts for
calf/foot rigid groups; because feet are merged into calf groups, this does not
establish foot-only support. No hardware calibration, latency or sensor-rate
availability is certified by these ideal virtual measurements.

## Fixed physical endpoints, separate from controller decisions

Use recorded true pose/joints and known opening/destination only after execution:

- At the candidate timestamp: destination contains the base, and the entire
  nominal articulated body is at least 0.02 m beyond the opening plane.
- At the final timestamp: the same two geometry conditions; base at least
  0.15 m beyond the opening for the final 150 physics samples (0.3 s).
- No native contact/physical stop. All 250 release samples exist; final 100
  have planar speed <=0.1 m/s and world yaw rate <=0.25 rad/s. Final base height
  >=0.2 m and absolute roll/pitch <=0.5 rad.

Report candidate geometry agreement, false candidates, viable physical arrival,
physical arrival without a candidate, candidate without viable release,
actual progress, contact/stability stops, sensor faults and all terminal states.
Integration success requires both the timely candidate geometry check and all
final physical checks, with no sensor fault. No endpoint creates a trusted
runtime edge retroactively. Report all four fixtures for every method; no
confidence interval treating correlated frames or deterministic fixtures as
independent mazes. Timing excludes capture/physics waits, not hardware latency.

## Evidence and next decision

Fresh root: `.generated/go2_observed_traversal_development_v1_attempt_001`.
Bind sources, tests, this protocol, predecessor identities, exact final models,
URDF and gait before the first physical step. Preserve raw physics, native
contacts, actual RGB, slow/fast causal histories, all commands, decisions and
the final provisional ledger. Pair the settling physics/body prefixes across
methods. Full replay reconstructs raw sensing/camera transforms, commands and
outcomes, and reproduces every scientific decision field; only the explicitly
recorded inference_ms and adapter_ms are excluded from equality.

This protocol is fixed before execution. No tuning or rerun of this panel after
seeing outcomes. Inspect failure mechanisms to choose a separately specified
next development intervention. Even success leaves reliable place association,
beacon detection, directed return, long-horizon planning, independent final-maze
comparisons and real-Go2 evidence outstanding.

# Local control factorial development V1

Specified before collection, 5 September 2026. This is a new development study,
not a rerun of an interrupted attempt, a novel-maze benchmark, or a JEPA test.
All previously collected material and source remain unchanged.

## Questions and matched design

Does initial heading alignment reduce turning-approach collisions? Does feedback
during arrival improve the residual motion state? Does a missed instantaneous
arrival threshold actually prevent continuation?

Use the eight development geometries from contact-attributed execution V1:
straight, offset, left90, right90, at widths 0.75 and 1.0 m. Reusing these observed
geometries is explicit. Four arms per case, in fixed order: baseline,
prealignment only, arrival feedback only, both. All arms for case i share seed
2026090600+i and identical fresh initial conditions, but have new scene/trial
identities. No previous runtime snapshot or candidate payload is loaded.
Compare the recorded settling prefixes across arms to verify pairing.

Use the same gait checkpoint, assets, limits, collision ontology, recording,
fixed camera mount and physics/policy/command periods as the completed V1.
Record settling for 1.5 s. Each edge has a common maximum of 85 command ticks
(8.5 s), including at least five arrival ticks when no physical stop occurs.
Alignment/approach must yield to arrival by tick 80, reserving that minimum.
Baseline uses the existing pure-pursuit command formula and five zero-command
arrival ticks. It must be checked against V1's baseline implementation on
identical input states and its actual observed baseline outcomes, rather than
assumed equivalent from a method name.

## Fixed interventions

Prealignment: before translation, turn toward the first segment of that edge's
route using the existing yaw gain 1.5 and +/-0.45 rad/s command limit, with zero
forward command. Proceed after two consecutive command-boundary observations
have heading error <=0.10 rad and absolute world-z angular speed <=0.25 rad/s.
Alignment consumes the same edge budget. There is no retrospective selection
of the most favorable alignment time.

Arrival feedback: after sustained crossing, or when the approach budget ends,
request forward velocity clip(-0.5 * measured body-forward velocity, -0.08,0.08)
and yaw rate clip(1.5 * port-normal heading error - 0.3 * measured world-z angular
velocity, -0.45,0.45). Lateral command remains zero. Stop online after at least
five arrival ticks and a complete trailing 0.20 s window meets the existing
arrival-motion/geometry thresholds; otherwise finish at the common deadline.
This feedback is an oracle-reference capability test: simulated base pose and
velocity are privileged inputs, not yet deployment-qualified sensor estimates.

The instantaneous motion, height, attitude, lateral and beyond-port thresholds
remain those of V1. The arrival-phase completeness check becomes **at least**
250 physics samples, because variable-duration feedback is the intervention;
baseline still uses exactly 250. Do not relax force, motion or clearance limits.
Report instantaneous and sustained-window arrival separately, including deadline
exhaustion and elapsed command/physics time.

## Actual continuation

After the first edge, continue from its measured state without resetting or
teleporting, including when instantaneous arrival checks fail. Only a physical
or integrity stop prevents continuation. The second directed boundary is at
local x=1.2 m inside the same exit corridor, with the same opening width and
normal. It is a straight-corridor continuation test, **not** a second junction
or proof of general maze routing. This isolates whether the first terminal
motion state prevents an immediately available next traversal. The next route
ends at local x=1.7 m. Each arm uses its same controller on both edges.

Primary task outcome: two sustained, correctly directed crossings without
disallowed contact, and a usable final arrival under the unchanged motion
thresholds. Also report first-edge usable arrival, two usable arrivals, sustained
arrival windows, and continuation conditional on a missed first-arrival proxy.
Never drop first-edge failures from the overall denominator. No claim of
independent-maze uncertainty is made from these correlated deterministic cases.

## Measurement, stopping and audit

Stop immediately at measured disallowed contact, base height below 0.15 m, or
absolute roll/pitch above 0.70 rad; retain the terminating physics/contact sample.
An infrastructure or measurement-integrity error stops the study, preserving
partial evidence. No automatic retries or extra seeds.

Persist native contact packets, attributed events, all physics samples, edge and
controller-stage identity, pre-action decision records, first/second edge terminal
results, and fixed-mount RGB at initial/first-edge/final boundaries. Compare
source/config/gait hashes before collection and audit the raw packets, timing,
command limits, prefix pairing and endpoints afterward. Parameter or endpoint
changes after observing outcomes require a separately identified follow-up.
The study has exactly 32 planned trials; completion is an interpretable result,
not a requirement that an intervention win or that all cases pass.

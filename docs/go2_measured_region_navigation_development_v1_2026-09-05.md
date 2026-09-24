# Measured-region continuous navigation V1: fixed development experiment

## Scientific purpose

Connect the completed depth/gyro estimator to control, replacing the global
floor-mask-change/command-distance local arrival rule. Exercise the entire
exploration, hidden-marker discovery and tentative route-return mission. This
is a transparent geometric control baseline, not a learned policy or a JEPA
advantage experiment. All earlier failed attempts and metrics remain unchanged.

Two fixed existing development layouts, north_dogleg and south_branch, use
episodic memory, the same seeds, scene geometry, gains, gait, RGB/depth mount,
native physics/sensor clocks, marker detector, budgets and physical task metrics
as the moving observation study. Method is explicitly `measured_region`.
Output: `.generated/go2_measured_region_navigation_development_v1_attempt_001`.
No retry or source change after launch. Record all failures, including before
the first traversal. This is not an independent-maze or hardware population.

## Changed controller, fixed before execution

The controller receives actual causal depth and the live depth/gyro relative
state, not raw physics, wall boxes, coordinates of the beacon or a teacher route.
State must bind the current RGB/depth hashes and clock and compose full observed
translations. A missing translation or inconsistent history stops control; no
command-distance fallback fills an unobserved component.

Track intersections of adjacent nonparallel wall supports with nearby observed
endpoints. Intermediate sampled returns must be valid and supported by one of
the two planes; unknown gaps are not bridged. Outside corner orientation and
longitudinal support distinguish a near versus far candidate boundary. Store
up to64 approach-conditioned corner records in the arbitrary initial body frame;
these are not recognized places or verified opening widths. Limit target reuse
to30 s and compatible approach direction. For a far boundary select a target
behind it by the current nominal turn radius plus0.10 m; for a near boundary,
ahead by that amount. If no outside corner is observed, a directly observed
front wall may supply an end-of-corridor target with radius plus0.15-m standoff.
If no measured target exists, fail rather than inventing a fixed travel length.

Initial approach is along the current forward direction. Skip the predecessor's
initial floor-bearing yaw alignment, because its surrounding yaw volume has not
yet been observed. Later branch alignment remains, subject to the same sampled
turn-volume gate as scans. This initialization is a development condition, not
generalization to arbitrary initial misalignment.

Drive toward the measured target at0.20 m/s, reducing to0.10 m/s within0.25 m,
with bearing feedback gain1.5 and yaw cap0.25 rad/s. Begin braking when remaining
forward distance is <=0.5 times the last measured forward speed plus0.03 m.
Keep the existing gyro/joint quiet criteria,0.5-s minimum braking,0.3-s quiet dwell
and2-s settling limit. A local attempt is bounded by30 s. After stopping, require
the sampled nominal turn volume to be supported by observations and target
distance within0.20 m, rejecting overshoot beyond0.15 m. If still short by more
than0.04 m, bounded low-speed correction can continue within the same attempt.
Unknown volume at the target fails; no relaxed unknown-count threshold is used.

The old arrival ledger is not populated with fabricated visual-change values.
A distinct measured-arrival record retains target, actual measurement time,
remaining distance and sampled-volume outcome; place identity and qualified
arrival remain unasserted. The outer marker/search/return logic is shared.
Home-appearance stopping additionally requires estimated distance <=0.25 m
from the initial relative origin; an empty route farther away fails explicitly.
Physical home return is still judged independently from native physics.

## Sampled volume and important limits

Keep up to64 pose-diverse depth keyframes, plus the latest frame even when
stationary. Ray evidence is in the estimated initial body frame. Require valid
neighbouring depth returns, a4-cm foreground margin, calibrated image bounds
and explicit unknown outcomes outside acquired views. Contradictory near-surface
evidence overrides old free-ray evidence. No unobserved side/rear volume is
declared free. Static scenes and the existing ideal sensor calibration are
assumptions, not hardware guarantees.

The nominal yaw profile includes all27 URDF primitives over the20 observed joint
postures in the current body history. Primitive transverse AABBs bound radii in
gravity-aligned height bands. Use4-cm model padding and8-cm volume sampling,
including circular boundaries and interiors. Low calf/foot bands may use actual
observed horizontal ground returns within6 cm of the queried height; this is
reported separately from free space and is not allowed for torso bands.
The initial quiet specific-force mean supplies a conditional gravity direction.

This profile bounds observed postures, not arbitrary future gait, slip, contact,
pose drift or between-sample geometry. Its sampling and margins are explicit
development assumptions, not a continuous-volume proof, calibrated probability
or hardware safety authorization. Scan/alignment requests require its current
sampled support; native2-ms contact/body stops and independent complete physical
metrics expose failures. Do not relabel a passed sample check as a certified
future sweep. Actual low-level gait effects remain part of the experiment.

## Full evidence and next decision

Preserve all RGB/depth/body/gyro/relative-state packets, controller decisions,
command tape, gains, contacts, actual object/camera identities, route ledgers,
memory and terminal metrics. The collector changes only its explicit controller
call. A distinct core auditor retains the entire predecessor physical audit,
changing only the controller class and its sensor-input replay. The full depth,
relative-motion and metric replay is retained, including the prior numerical
wall-edge failure semantics. New code and all predecessors are source-bound.

Report first local completion, scan/alignment failures, discovery, actual return,
contacts, sensor/observability failures and every whole-task outcome. Neither a
local arrival nor successful code tests count as the final goal. If the new
controller reveals loss of translation observability during a turn, retain it
and implement a separately tested state-estimation successor; do not pretend
that pure-turn commanded translation equals zero. Once continuous execution
works, require matched memory and supervised/JEPA/multistep-planning comparisons,
independent layouts and sensor robustness before transfer claims.

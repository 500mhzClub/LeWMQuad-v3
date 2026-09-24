# Release-aware alignment navigation V1: fixed development experiment

## Question and comparison

Does explicitly measuring the release/settle phase allow the existing measured-
line controller to proceed through alignment and the continuous mission? The
predecessor completed both local arrivals/scans but failed alignment0/2 because
entering tolerance then requesting zero did not maintain its dwell. Preserve
those source-bound runs and every earlier failure without rescoring.

Exactly the same two original development layouts, north_dogleg and south_branch,
use their original seeds, physical construction, marker, episodic memory, gait,
actuator gains, clocks, sensor paths and360-s/36-leg mission budgets. The method
is `release_aware`. Fresh output:
`.generated/go2_release_aware_navigation_development_v1_attempt_001`.
No retry, source change, threshold adjustment or new coefficient after launch.
Neither this development population nor a passing local operator is independent-
maze, learned-policy, JEPA-contribution or hardware evidence.

## Changed operator

Retain line approach, measured targets, local braking, nominal sampled clearance,
arrival rules and the entire outer mission. Replace only an unstarted alignment
operator. Continue bounded PI control toward an inner heading target0.005 rad
(one quarter of the existing outer tolerance). Keep proportional gain1.5,
integral gain0.4/s, integral command cap0.12 rad/s, command cap0.35 rad/s and
anti-windup/sign-change reset. Do not release merely upon crossing0.02 rad.

When absolute heading error<=0.005 rad and observed projected rate<=0.1 rad/s,
request zero and enter SETTLE. Reset integral, observe at least0.5 s of actual
zero-command intervals, and require the existing outer absolute error<=0.02 rad,
rate<=0.1 rad/s and0.3-s continuous dwell from fresh observations. Complete only
after these release requirements. If outside outer tolerance/rate after0.5 s,
reenter bounded correction with the original start time. If quiet but dwell is
not yet complete, keep observing zero. The total alignment deadline stays12 s;
failure and terminal commands remain zero. No constant release-shift correction,
oracle pose, privileged velocity or inferred zero translation is supplied.

Validate consecutive100-ms clock, current sensor/attitude timestamp, proper
rotation and valid finite latest gyro. Invalid observations latch a stop. The
parent full sensor histories, depth state and turn-volume checks still apply.
The test suite must cover threshold-edge non-release, actual release intervals,
synthetic recoil/delay/overshoot, failed settling under the original deadline,
no response, invalid input, fresh-operator installation and exact unchanged
collection/full-physics-audit function bodies. Synthetic dynamics are stress
fixtures, not a calibrated Go2 model or closed-loop navigation proof.

## Evidence and interpretation

Record every sensor packet, requested/applied command, gains, contact, stage,
alignment phase, release time/count, provisional memory/ledger and physical task
outcome. Retain complete depth/state replay and independent native-physics mission
scoring. Full audit PASS certifies fidelity, not task success. Missing translation
still stops control; no rank-threshold relaxation or command fallback is allowed.
Nominal sampled turn volume still does not qualify unobserved future gait,
continuous-volume collision clearance, sensor uncertainty or hardware safety.

Report whether alignment actually completes after zero release, whether further
local execution remains observable, marker discovery, physical home return,
contacts/faults/false claims and all depth/motion failures. Do not stop evaluation
at a successful alignment. A fresh whole-task failure changes the next engineering
decision but is not a successful mission. After reliable continuous execution,
matched memory, supervised/JEPA predictive-training and actual multi-step rollout
comparisons on independent layouts/seeds, sensor robustness and bounded real Go2
evidence remain required for the full scientific objective.

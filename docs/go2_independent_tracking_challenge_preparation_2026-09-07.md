# Independent long-motion tracking challenge: source preparation

Status: constructor, fixed excitation and measured-coverage evaluator implemented;
161 focused/adjacent tests pass, including50 new challenge tests. **No native collection, replay,
model fitting or controller adoption has been launched for this challenge.** This
document is a prospective preparation record, not an execution receipt or a claim
of new independent observations. The full scientific goal remains unachieved.

## Scientific purpose

The completed temporal-anchor replay recovered three one-frame interruptions on
reused controller recordings. Those recordings include the old controller's stop
behavior, so they cannot establish continued-turn success under the candidate.
The next evidence must change observations and measure sustained movement, not
rename that replay or integrate requested commands as if they were physical pose.
This challenge addresses a prerequisite for navigation, not JEPA value by itself.

The original and temporal-anchor observers remain unchanged. Do not adjust the
10-frame bridge allowance, 2cm anchor/increment consistency check or registration
gates from exposed results. The original0/3 physical-simulation room-return result
and both failed optical-flow initialization experiments remain negative evidence.

## Fixed population and acquisition contract

Eight trials: two newly specified metric scenes (`offset_niche`,
`unequal_baffles`) by two friction conditions (1.0 and0.15) by left/right first
turn. Within a scene, starts, appearance and physics seeds are shared across the
direction/friction treatments; friction is construction/evaluation metadata, not
a selector or observer input. These are **two scene clusters, not eight independent
mazes**. Both are observation-assay rooms with different boundary dimensions and
internal walls; neither is a novel-maze navigation benchmark.

New starts are(-0.83,-0.47,0.17rad) and(-0.61,0.39,-0.23rad). The geometry identity
ignores names, material labels, order and random seeds. Source tests compare the
actual wall-box metrics against the12-layout learning inventory and the former
single-wall motion scene. This is exact geometry nonidentity, not isomorphism
proof, a native construction check, or proof of different images. The eventual
auditor must bind actual native wall geometry, settled state and raw prefix
identities against the completed inner/intent recordings before making an
independent-observation claim. Do not silently relabel a duplicate prefix.

Each trial has15 settling ticks followed by442 requested100ms command intervals:

| Segment | Intervals | Forward / yaw request |
| --- | ---: | --- |
| Initial hold |20|0 /0|
| Approach |40|0.12m/s /0|
| Brake |20|0 /0|
| Turn out |126|0 /signed0.25rad/s|
| Brake |20|0 /0|
| Translated view |30|0.12m/s /0|
| Brake |20|0 /0|
| Opposite turn |126|0 /opposite0.25rad/s|
| Final hold |40|0 /0|

Maximum443 RGB-D observations and22,850 actual500Hz physics samples per trial;
3,544 observations and182,800 physics samples across the full population. No
extra tail, adaptive extension, replacement or implicit retry. Requested yaw
integrals are±3.15rad; actual signed turns and achieved translations are evaluated
from recorded native poses only after the paired sensor-only phase is complete.

The separate `IndependentTrackingSession` uses the existing CPU union-wall scene
builder, floor-first raster order, core-profile precision readback and5mm render
near clip. It retains the public0.2–5m depth validity contract, raw50Hz body and
500Hz gyro recording, learned gait, command bounds, contact identities and native
stop-only supervision. A new physical constructor accepts only the exact new
specifications; it does not masquerade as the frozen12-layout inventory or patch
that collector. Partial construction after scene allocation destroys that scene
on failure. The eventual runner must own Genesis shutdown and persist any partial
recorded evidence. Startup articulated geometry, gains, support and velocity
checks remain mandatory; a geometric drawing is not startup safety evidence.

This remains a hidden-robot, ideal simulated sensor experiment with paused
physics. Reusing its renderer/recorders does not qualify self-occlusion, latency,
calibration, bias, noise or real hardware. No contact or inertial translation is
silently substituted for unavailable current visual measurements.

## Measured coverage, not commanded coverage

`measured_coverage` requires the full finite500Hz prefix, consistent sample/tick
counts, unit XYZW quaternions and explicit terminal stop accounting. It reports
each complete, partial or unavailable segment without dropping failed cases.
It unwraps yaw from native orientation, not from commands or observer output.

The fixed intended-motion coverage requirements are both signed net turns≥150°,
both translational displacements≥0.20m, and the entire final second at linear
speed≤0.02m/s and angular speed≤0.05rad/s. They describe exposure to substantial
turning/translation and stopping, not accurate command tracking. If a low-friction
run stops, turns the wrong way or does not reach these amounts, it remains a
negative/incomplete coverage case—not an excuse to extend its tape. The helper
also reports path lengths so endpoint displacement is not confused with travel.

The numerical helper is evaluation-only; it does not authenticate artifacts.
Before parsing native arrays the future runner must authenticate and persist the
**complete paired sensor phase for all eight trials**. The source tests are not
that phase boundary or an authorization to inspect any sealed/predecessor output.

## Required before execution and before adoption

1. Finish a bounded collector, exact artifact roster, terminal audit and paired
   replay integration. Preserve all current771 supervisor/765 child,786 learning
   study and701 completed tracking-replay bindings. Do not launch a second native
   job while the independent-learning collector owns collection resources.
   The new native attempt should start only after the collector is terminal and
   scheduling/storage are checked; it must not starve the prepared matched study.
2. Freeze the full population, source/input/native bindings and a fresh exclusive
   output before exposure. Provision at most3GiB per trial and24GiB total with
   the existing40GiB free-space reserve; check actual serialized metadata and
   image/native arrays before admitting these bounds. Enforce bounds during
   acquisition and at persistence, including failed/partial runs. These figures
   are proposed bounds, not an implemented/enforced storage guard in this module.
3. Collect identical raw sensor tapes for both observers with commands determined
   solely by the fixed direction/tick and causal packet validity. Tracking loss
   is a shadow result, not a command stop; native physical limits and invalid
   acquisition still stop the tape. Save all commanded intervals, stops and
   actual initial/prefix/scene witnesses. Authenticate before replay.
4. Score original and frozen temporal-anchor observers over every available raw
   frame, with terminal latching and no silent reset or per-trial method choice.
   Report unavailable frames, bridge spans, rejoin disagreements and native
   absolute/incremental errors separately. A preliminary task allocation is
   max2cm position and2° orientation error: over a0.4m local leg, the latter
   contributes about1.4cm lateral error, leaving approximately2.6cm of the6cm
   tolerance for other errors. This is a conservative design allocation, **not**
   a calibrated probability bound or a guarantee over accumulated maze travel.
   Require complete current-pose availability over all completed challenge tapes
   for adoption; report partial/stopped cases separately and do not drop them.
   Incomplete intended-motion coverage prevents claiming the full challenge passed.
5. Add fixed stress arms before exposure: longer old-anchor absence versus
   missing current RGB-D, contradictory associations, repeated appearance and
   specified depth/gyro errors. Specify exact perturbation amplitudes, times,
   sensor-clock handling and success/failure semantics in the runnable protocol.
   They are not implemented here and must not be replaced with threshold tuning.
   Artificial old-anchor denial tests state-machine behavior; it must not be
   presented as naturally observed occlusion or a changed production estimator.
6. Instrument acquisition, feature extraction, both registration paths,
   selection, command dispatch and persistence separately. Record monotonic wall
   times and count missed100ms deadlines. Sequential paired shadow computation
   is not one deployable controller's timing; summed/replay timings are estimates,
   not real-time qualification. A wall-clock-driven fresh closed-loop assay still
   follows only after accuracy/coverage checks and a frozen execution review.

Even a passing challenge will not fix low-friction dynamics, demonstrate useful
JEPA prediction, or prove memory/backtracking. Continue the12-layout collection
and36-fit matched learning experiment, then test predictor/online-rollout/memory
contributions in actual independent-maze execution with matched sensors and gait.
Deployment-valid sensing and bounded hardware evidence remain separate requirements.

## Verified source-preparation evidence

Test session89730 terminates exit0:158 passed in34.79s across seven explicit test
files, including47 new cases. The other111 tests cover the existing inventory
adapter, ordered/core capture, temporal-anchor runtime/replay and its independent
verifier. This is a focused/adjacent regression, not a new full-repository run.
Durable JUnit:
`.generated/navigation-development-staging.m6MDz1/independent_tracking_preparation_v1.xml`,
SHA256`75031d1afa5ee8ce52ca94237a1d5fa8815d853f03475b15384ef2c83048e00e`;
158 tests, zero errors, failures or skips.

The initial focused invocation66009 had44 passes and one failed resource-count
assertion: the declared segment lengths sum to442 ticks, not the handwritten542.
The assertion was corrected to442/443/22,850 ticks/frames/physics samples; the
command schedule did not change. Two subsequent stop-prefix tests bring the new
test count to47. No data were collected or examined to make this correction.

Coverage checks include a complete command tape with no actual motion, wrong-way
turns, final-second speed violations, physical stops at every partial-tick edge,
acquisition stops, setup-only stops, malformed clocks and forged completion
metadata. Source-construction tests compile actual pack and union-boundary
geometry without launching Genesis. Mock-backend tests check exact friction,
render settings, no randomized spawn, and scene destruction on constructor
failure. They do not substitute for the future native startup/prefix audit.

Final review added explicit retention of stops before settling is complete,
including zero/one/749 available physics samples. They now report every motion
segment unavailable rather than dropping the case or inventing an initial pose.
Final session19144 terminates exit0:161 passed in34.76s, including50 new challenge
tests and the same111 adjacent tests. JUnit
`.generated/navigation-development-staging.m6MDz1/independent_tracking_preparation_final_v1.xml`
has161 tests and zero errors/failures/skips; SHA256
`b468d773e4930e83c3ad5415597a6a39ae9928eda30dcf31fd13766aed972bc6`.

Read-only guard63535 terminates exit0 and confirms the completed replay's701
source bindings and the prospective learning study's unchanged786-source
definition`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.
The new files are outside both source sets. The final early-stop changes affect
only the new coverage module and its new test file. No frozen/live source was
edited. Collector session25963 remains live; its l04 child2090031 has reached
54/120 prechecks, with no terminal layout audit yet. This is a live progress
observation, not a fifth completed layout. The matched scientific fit output
remains absent.

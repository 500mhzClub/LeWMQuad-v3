# Next implementation: observable local geometry within the whole mission

Latest: [RGB-D point correspondence diagnostic](go2_rgbd_correspondence_motion_diagnostic_v1_result_2026-09-06.md)
rejected80/80recordedpairs and13/13Bweakpairs; the latter have zero keypoints.
The low-texture scene explicitly disables textures. Tests1833/148files pass and
independent sensor reconstruction is exact. Implement the new
[geometry-preserving appearance experiment](go2_rgb_information_and_whole_mission_next_steps_2026-09-06.md),
not a looser acceptance threshold or the existing collision-mesh texture switch.
Then fresh continuous full missions and matched JEPA/rollout/memory tests. No
new physics, recovered observer or maze success is claimed by this diagnostic.

Current next step: [the post-B observability/whole-mission plan](go2_post_B_observability_and_whole_mission_plan_2026-09-06.md).
The A-only response was fitted/frozen and B was executed once. B stopped after
24/28 targets at5.3s when depth rank2 persisted for1.3s and the unchanged80mm
pose proxy budget was exceeded (85.52mm). Raw replay exactly reproduced the
failure and checked the real3tick tail; no physical guard violation. The learned
model improves body translation but fixed posture has lower mean worst primitive
centre error at all four horizons on the stopped B trace. All37 fully timed
moving loops exceeded100ms (median234.34ms). Tests1816/147files pass. Next test
RGB-D correspondences for the depth-weak direction, then continuous full missions
with uncertainty-reserve observation actions and matched JEPA/memory/multistep
comparisons. No B retry/refit, threshold relaxation or maze/hardware success.
The entries below retain the earlier development chronology, not current status.

Latest executed evidence: [bounded motion identification A](go2_action_motion_identification_development_v1_result_2026-09-06.md).
All28 declared forward/turn/braking targets and three final zero ticks executed;
raw audit passed3000 physics samples/46 RGB-D frames/43 decisions with no stop,
disallowed contact or padded-body region violation. Continuous state survives
to6s, under the new explicitly checked calibration condition. The gait-motion
baseline is inadequate: at0.4s qdot persistence gives316mm worst foot-centre
error; a posthoc fixed-posture control gives71.7mm. Both remain recorded, neither
validates50mm. Next A-only model identification/freeze and the predetermined B
validation run, correcting end-to-end timing in new source. No B run, fitted
model, maze success or JEPA advantage yet; full goal remains active0/2.

Latest implementation: [factored evidence plus articulated trajectory API](go2_factored_trajectory_progress_2026-09-06.md).
New non-floor/ground/visibility channels retain actual contradictions and unknown
residuals. Every time-indexed pose/joint node includes explicit intersample
point-motion allowances; illustrative errors and the non-learned command/qdot
baseline remain unvalidated. Recorded 0.75/1m queries now have24/27 and23/27
conditional non-floor facts, only3/27 and9/27 observed separations, and5/8 and11/15
residual coverage. No configuration/action is qualified. Next implement/run the
[separate motion identification and validation schedules](go2_action_conditioned_motion_validation_next_execution_2026-09-06.md),
then continuous whole missions and matched JEPA/memory/genuine-multistep tests.
No new physics/training or whole-mission success is claimed here.

Latest evidence: [ground-veto attribution](go2_configuration_ground_veto_progress_2026-09-06.md).
All 270 per-view configuration witnesses were checked. At each farther tangent
offset, all 12 lower-leg veto witnesses arise from floor-family returns after
33–65-mm historical pose allowances disable the original floor exemption.
The current 1-m view separates and covers those four legs; 0.75-m whole-box
visibility remains incomplete. No original outcome or decision rule changed.
Next implement factored obstacle/ground/coverage evidence alongside a real
action-conditioned body/foot/braking trajectory interface, validate on fresh
bounded executions, then integrate a complete mission. Do not substitute more
static probes for motion validation. Regression: 1,749/142 files passed. No new
physics/training, mission success or JEPA benefit; all handles terminal.

Latest consumer: [observed residual clearance for articulated configurations](go2_observed_setup_configuration_progress_2026-09-06.md).
The initial-region partition now queries retained depth with per-view source
bindings, complete residual coverage and whole-query vetoes. Current-camera
pose cancellation is preserved without erasing global setup uncertainty. Fixed
recorded body-axis poses clear27/27 at0–0.5m,19/27 at0.75m and20/27 at1m; farther
queries retain ground/leg negatives and unknowns. The separately declared
gravity-tangent probe removes the16.2mm/m downward body-axis component and foot
penetration flags, but four lower-leg vetoes remain; no action is approved.
Tests1745/141files pass. Next per-view ground/error diagnosis and explicit
support/contact plus validated action-conditioned trajectories/braking, then a
fresh full mission and matched JEPA/memory/multistep independent-layout tests.
No new physics/training, no maze success or realtime claim; all handles terminal.

Latest integration: [continuous startup state and clearance provenance](go2_continuous_startup_handoff_progress_2026-09-06.md).
One observer/memory consumes all 15 saved startup/tail frames, reproduces 12
startup decisions and 15 relative records, and reaches a state-only handoff at
2.9 s without resetting the retained 12-mm initial-velocity position contribution.
All 27 expanded current primitives are conditionally inside the supplied starting
region; observed clearance remains empty. Exact region/residual partitioning
preserves unknown space, expiry and all-view vetoes. Tests pass 1,726/140 files.
Next observation-bound residual queries, explicit support and action-conditioned
body/foot/braking predictions, then a fresh continuous full mission and matched
JEPA/memory/multistep independent-layout tests. No new physics or training; maze
0/2 unchanged, no real-time claim, all handles terminal.

Latest physical result: [verified startup observation turn](go2_startup_observation_turn_result_2026-09-06.md).
The new turn recovers rank-3 depth at 2.2 s, completes braking at 2.6 s and records
a quiet three-tick tail through 2.9 s. All 1,450 physical samples, 15 RGB-D frames,
setup/native-foot/contact checks and controller decisions pass raw replay. The
regression passes 1,692 tests across 138 files. No new maze or learned-policy
success: 0/2 remains. Capture plus controller exceeds 100 ms at all 12 decisions,
and terminal memory still has zero own-body floor-covered primitives/positive
clearances. Next implement the persistent startup-to-navigation handoff, explicit
initial-versus-observed clearance/support semantics and action-conditioned motion
in [the continuous-navigation plan](go2_post_startup_continuous_navigation_plan_2026-09-06.md),
then fresh complete missions and matched JEPA/memory/multistep comparisons.
Do not silently extend this trial's setup expiry or equate local success with
the full goal. All execution and diagnostic handles are terminal.

Latest controller implementation: [setup-conditioned observation turn](go2_startup_observation_turn_implementation_progress_2026-09-06.md).
The new sensor-driven controller turns promptly, brakes at full-rank depth and
requires quiet rank-3 observations; all old uncertainty and observed-conflict
stops remain. Its all-joint URDF radius0.7444116m plus padding/prospective motion
requires a separately declared/validated new1m-halfwidth starting cube. It checks
the envelope through the full0.4s horizon. Native foot-geometry capture/matching
and an extra non-foot-ground contact check are implemented but not live-executed.
Next implement the fresh launcher, admission/contact wrapper and raw auditor for
the proposed startup-turn protocol, then execute and verify the bounded physical
action. No new physics or maze result yet;0/2 and all prior failures unchanged.

Latest setup verification: [recorded startup snapshot](go2_setup_snapshot_result_2026-09-06.md).
The fixed initial-body region [-1,-0.75,-0.5]..[1,0.75,0.6]m contains all27
padded primitives and is separated from every recorded native wall (minimum
distance lower bound1.4894m). The20mm/s velocity ball contains the reference
1.7004mm/s initial velocity. All four native calf support groups carry positive
upward force; no other loaded contact; nominal non-foot floor gap>=28.199mm.
This is one evaluator-only setup snapshot, not future foot/contact qualification.
Next explicit new startup controller/admission, native per-geometry foot identity,
ground/contact and prospective-motion assumptions, and a fresh promptly-started
observation turn to test recovery of full-rank motion before budget expiry.
Keep all old stops and0/2 intact; then full missions and matched learning/memory/
multistep comparisons with the same setup condition across arms.

Latest setup implementation: [explicit bounded priors](go2_setup_priors_progress_2026-09-06.md).
The new velocity-prior adapter labels initialization as supplied setup evidence,
propagates its conditional ball through weak subspaces, and adds its radius to
the unchanged inherited proxy/budget. On the existing 26-frame tape, the proposed
0.02-m/s ball contains the reference initial velocity (0.00170m/s); memory gets
past frame 1 but stops at frame 12 on an85.42-mm combined scale. No prior was
retuned, no stopped memory restarted, and no unknown floor filled. A separate
finite/expiring setup-region contract respects observed conflicts but establishes
neither support nor contact. Next independent setup validation, ground/contact
semantics, prospective motion/stopping and timely observation actions in a new
complete controller, then fresh missions and matched JEPA/memory/multistep tests.
This is conditional recorded diagnosis, not independent setup or maze validation.

Latest startup evidence: [recorded plane-memory diagnostic](go2_plane_memory_startup_result_2026-09-06.md).
The new per-observation plane consumer finds a hypothesis in all 26 bounded-Go2
frames, but no current own-body floor footprint is covered. All foot centres
are behind the camera. Historical fusion stops at frame 1: all 25 depth
increments are rank 2 and initial velocity is unavailable. This failure is
preserved; later local diagnostics do not restart historical memory. Next define
an explicit, independently checked development starting-state/support-region
contract shared across baselines, or provide deployment-valid motion/contact
evidence. Separate supplied priors from measured free space; retain no-prior
failure. Then contact-aware prospective control, full-loop timing and fresh
complete missions, followed by matched JEPA/memory/multistep comparisons. More
plane-mask tuning cannot supply missing startup information. Original 0/2 remains.

Latest Go2 integration: [bounded/aligned robot interface](go2_bounded_floor_robot_interface_development_v1_result_2026-09-06.md).
The explicit new builder/init/session executes2000physics samples and26RGBD
frames with the same gait/nonfloorphysics. A separate7-slot native-box reader
correction preserves the original audit failure; corrected raw reconstruction
passes. Floor/wall maximum depth errors0.0453/0.0190mm; no native disallowed
contact. Nominal feet reach-3.68mm duringsettle/-2.06mm active, while allnonfeet
stay>=23.14mm aboveground. This is a short4s interface trace, not a contact bound
or maze result. Next per-observation measured plane hypotheses and explicit
modelled foot contact, prospective motion/fusedspeed/fulltiming, thenfreshfull
missions andmatchedJEPA/memory/multistep/independentlayouts. Original0/2 remains.
Final regression passes1,552 tests across131files; all process handles terminal.

Latest rendering evidence: [paired floor extent precision](go2_floor_extent_precision_development_v1_result_2026-09-05.md).
Eight new paired views show that a32m aligned visual reduces native depth error
in8/8 cases; maximum0.095515mm versus1.479955mm for the1000m control. All raw
buffers reconstruct native pixels exactly; float64 conversion changes assessed
depths by at most0.016470mm and does not fix failing control views. Next explicit
bounded-floor Go2 builder/session/contact-identity integration with verified
finite support, then realistic dynamiccontact, per-frame observed surfaces,
full-loop/prospective control and fresh complete missions plus matched JEPA
comparisons. No calibration or navigation claim. Tests1,524/129files; allhandles
terminal. Earlier contact-assay failure and whole-mission0/2 remain unchanged.

Latest physical-interface evidence: [aligned floor V1 result](go2_aligned_floor_interface_development_v1_result_2026-09-05.md).
The fresh scene removes the visual/collision construction offset, but its strict
audit fails: a dropped22-mm sphere penetrates3.69mm during impact; two yawed
camera views have1.76/1.35mm depth error, exceeding the proposed1-mm allowance.
Native camera rounding does not explain the range errors. Settled sphere error
is0.108mm, not a Go2 contact qualification. Next investigate raster geometry/
precision and a dynamics-aware contact model, then an explicitly new Go2 scene/
session, observed per-frame surface hypotheses and fresh complete missions.
Tests pass1,507/128files; this does not override the actual interface negative.
All handles terminal; original0/2 and full scientific objective unchanged.

Latest runtime: [compiled primitive beams and shared floor families](go2_primitive_beam_kernel_progress_2026-09-05.md)
preserves every recorded reference/compiled result and view-cell count. Queries
fall from1.7-3.2s to313-559ms with new masks,80-137ms on identical-observation
repeats. This does not meet the complete100ms loop or prove next-tick cache reuse.
Tests pass1,487 across127files. Seedless-view vetoes/contact ambiguity remain;
known5mm visual/collision-floor mismatch also exceeds the current1mm plane
hypothesis and must be explicitly handled before a contact-aware successor.
Next provenance-bound per-frame ground hypotheses and surface alignment/error
model, actual sequential timing, prospective/contact control and fresh full
missions with matched JEPA comparisons. Original0/2 unchanged; allhandles terminal.

Latest consumer: [observation-bound primitive obstacle memory](go2_primitive_obstacle_memory_progress_2026-09-05.md)
binds current joints/depth/fused poses and preserves all-view obstacle vetoes.
Exact pixel-cell beam/box intersections replace independent rectangle/depth
extrema; observed incident cells classify boundary floor without making partial
views clear. At180,21 primitives conditionally clear and2 rear feet remain
candidates;4 primitives still have seedless-view near-return vetoes. No overall
clearance/contact permission. An exact incomplete-view early-out preserves all
decisions, but queries still take1.7-3.2s. Next query-independent observed floor
hypotheses with provenance/error accounting, shared/compiled reductions and
explicit modelled foot contact/prospective motion, then fresh complete missions
and matched JEPA comparisons. No original controller/radius/result changes;0/2.

Latest physical semantics: [primitive floor gaps and observed footprints](go2_primitive_floor_relation_progress_2026-09-05.md)
separates unpadded physical minima, padding and exact foot-sphere contact roles.
At tick180, all540 primitive/posture floor footprints are observed in one
retained view;460 non-foot cases have positive lower gaps (minimum9.94mm),
while80 foot cases straddle zero and remain candidates only. The20 measured
joint vectors are evaluated at one current body pose, not historical poses or
a future sweep. Non-floor clearance/contact admissibility, error calibration,
full-loop timing and navigation remain unresolved. Next clock/pose-bound
floor-versus-obstacle consumer and explicit modelled contact semantics, then
prospective motion and fresh complete missions with matched JEPA comparisons.
No controller/radius/threshold change and0/2 unchanged.

Latest geometry: [coupled finite-error floor footprints](go2_coupled_floor_enclosure_progress_2026-09-05.md)
derives point/height coupling with explicit plane/up error allowances. The four
previously uncovered samples now have measured-cell/plane-family coverage in
14/16/16/13 views, but none meets the unchanged height band. First-view upper
heights are63.9mm; even nominal-plane point-box extrema are62.6mm versus60mm.
These are padded turn-volume samples, not measured foot contacts. Next separate
observed floor coverage, physical non-support clearance and permitted contact,
with missing-floor/wall/penetration negatives; do not tune away this rejection.
Then validated error assumptions, shared-geometry full timing, explicit fused
speed/observation actions and fresh full missions with matched JEPA comparisons.
No controller/radius/threshold or original0/2 mission result changed.

Latest integration: [shared nominal evidence and supplied-bound floor coverage](go2_shared_nominal_floor_bounds_progress_2026-09-05.md)
uses one nominal depth observation for ray and floor consumers. All181 saved
nominal records match exactly and both fusion consumers agree. A summed-cell
index checks entire supplied point/height projection rectangles, including
interior missing data;252/256 old blocked ground samples at180 have this fixed-up
nominal-surface coverage. Four remaining rectangles contain non-ground cells,
not missing depth. Independent point/height bounds may over-enclose the coupled
region, but cannot be cropped without justification. Bounds/up/surface errors
are not calibrated or fully covered; all approval flags remain false. Tests
pass1,377 across122 files. Measured pipeline subset139–157ms still misses100ms.
Next coupled point/floor/up/surface enclosures and efficient shared geometry,
then fresh whole-mission control and matched supervised/JEPA comparisons.
All handles terminal; no controller/radius change and whole-task0/2 unchanged.

Latest runtime work: [exact shared-work registration acceleration](go2_certified_registration_reuse_progress_2026-09-05.md)
cuts median paired-observer cost from about101–102ms to52–53ms with exact public
output agreement on181 original and181 narrow-depth recorded frames. Nearest
matches are reused only with a distance certificate; ambiguity/lineage changes
fall back to full searches, and all weights/rank/iterations are recomputed.
The lean wrapper omits unused perturbation-only wall reports but preserves the
full nominal report and all sensor checks. Tests pass1,358 across120 files.
Worst observer time still reaches102.25ms before floor queries/control, so the
full deadline is unproved. Next integrate this shared nominal/paired path,
accelerate full uncertain-footprint coverage, validate error assumptions, and
run fresh whole missions with matched supervised/JEPA comparisons. No clearance
or source scale changed; all handles terminal and whole-task0/2 unchanged.

Latest floor work: [observed footprints and coupled pose/floor errors](go2_correlated_floor_evidence_progress_2026-09-05.md)
implements adjacent-pixel triangle coverage and raw paired floor-height
diagnostics. A missing-data negative shows the old ray/height predicate can
accept nearby floor while the footprint beneath the query is unobserved. In
181 recorded north packets, all nominal fusion outputs remain exact; all 256
blocked ground-role samples at tick180 have nominal and small-pair footprints,
with assumed three-source height std median2.60/max6.02mm. At tick80,83 blocked
ground-role samples still lack located footprints. These are not calibrated
error bounds or whole-envelope coverage; no radius or controller was changed.
Tests pass1,325 across119 files. Next derive an efficient joint registration/
pose/floor model and full uncertain-footprint coverage, validate its assumptions,
then integrate fused-speed/observation actions and fresh whole missions with
matched supervised/JEPA comparisons. All handles terminal; whole-task0/2 unchanged.

Latest uncertainty work: [coupled raw-sensor/registration pose propagation](go2_correlated_sensor_pose_progress_2026-09-05.md)
implements explicit shared-source paired estimators and historical joint pose
factors. An actual 181-frame diagnostic finds that freezing registration misses
up to 29.77 mm of position sensitivity for a declared 0.001-rad/s yaw-bias
source, despite unchanged rank. The raw-registration successor includes that
response and passes two difference-step checks; all nominal fusion outputs match
the existing observer. Tests pass 1,302 across 118 files. This is an offline
reference, not calibrated covariance or a navigation consumer. Next derive an
efficient joint model including registration, validate sensor-error assumptions,
and propagate the same errors into floor evidence before reducing any envelope.
Then full-loop timing, fused-speed/observation-action integration and fresh whole
missions with matched supervised/JEPA comparisons. All handles are terminal;
whole-task success remains 0/2 and previous clearance rejections are unchanged.

Latest prototype work: [exact query acceleration and relative-error propagation](go2_ray_memory_kernel_and_relative_error_progress_2026-09-05.md)
reduces recorded query times to about 15–60 ms with identical reference decisions,
including the unchanged ground-support rejections. All 457 original depth records
recompute exactly. The measured depth/memory/geometry/query subset still reaches
104–107 ms before acquisition and remaining control, so full real-time operation
is not established. Tests pass 1,274 across 117 files. A tested joint-pose Jacobian
shows genuine shared-reference cancellation, but no calibrated joint error model
has been supplied and runtime radii are unchanged. Next establish that model and
supported floor/contact constraints, finish full-loop timing, and integrate fresh
navigation plus matched action-conditioned supervised/JEPA comparisons. No process
remains live and no new maze success is claimed.

Latest consumer work: [fusion-aware uncertainty-dependent ray memory](go2_uncertain_ray_memory_development_v1_progress_2026-09-05.md)
is implemented as an unlaunched development prototype. It preserves observed/
predicted motion separation, checks the entire projected historical pose envelope,
and latches all three actual stress-case budget stops at tick 93. Tests pass
1,255 across 115 files. Actual-packet diagnostics expose two limits: later
clearance queries still take about 155–169 ms, and 256 ground-role samples are
rejected at the recorded arrival. Zero-radius diagnosis supports all samples,
but no runtime radius or clearance threshold was relaxed. Next derive justified
interval-relative error/floor-constraint handling and improve query latency,
then integrate observation actions and fresh physical-simulation missions with
matched predictive-turning comparisons. No current process remains live and no
new navigation/JEPA/hardware success is claimed; latest whole-task success is 0/2.

Latest estimator successor: [moment-aware fusion and prolonged weakness](go2_depth_inertial_moment_replay_development_v1_result_2026-09-05.md)
completed ten nominal and four explicit sensor-stress cases; exact replay and
independent verification passed all 4,233 produced estimates. Tests passed
1,234 across 114 files. Nominal worst position error falls from 7.50 to 2.52 mm,
but prolonged weak geometry and bias retain failures. The large bias exceeds the
proxy before its required stop; the uncertainty model remains uncalibrated.
All 333 launched sources and outputs are frozen; no replay/audit remains live.
Neither fusion estimator is connected to navigation; whole-task success is 0/2.

Next implement fusion-aware historical ray evidence and observation-action
selection, including lateral pixel-projection uncertainty and measured yaw-gait
translation. Use current views to reobserve/reposition before prolonged weakness;
do not fake rank-3 depth, extend blind inertial travel, or call proxy inflation
calibration. Keep exploratory simulation explicitly conditional and preserve
physical guards. Couple the action-choice work to matched supervised/JEPA turning
predictions, then execute fresh full missions. Another observer-only repeat is
not the next navigation milestone.

Latest estimator work: [depth–inertial fusion replay V1](go2_depth_inertial_fusion_replay_development_v1_result_2026-09-05.md)
completed 10 recorded trajectories / 3,610 observations, with independent
verification of every error, constrained component and position composition.
Worst step error is 1.27 mm; worst position error is 7.50 mm. This includes only
18 weak intervals in short stop/release sequences, not continued navigation.
Prelaunch tests passed 1,182 across 110 files; seven separate verifier tests pass.
The 322 launched sources and completed outputs are frozen; no replay or verifier
remains live. No fusion navigation consumer exists. Whole-task success stays 0/2.

The nominal passes do not validate uncertainty: endpoint velocity error exceeds
the 5-mm/s assumption at two of three weak entries, and 94 interval acceleration
errors exceed the 0.02-m/s² assumption. A posthoc time-weighted acceleration probe
reduces the two larger weak-trace errors, while slightly worsening the third.
Next implement a separately named moment-aware fusion successor with independent
time-varying-acceleration tests, then validate extended weak-geometry operation
and couple pose uncertainty to historical ray evidence. Pair the navigation
integration with dynamics-aware turning and matched supervised/JEPA prediction;
do not substitute proxy inflation or observer-only progress for mission success.

Current result: [depth-floor-hold integration](go2_depth_floor_hold_navigation_development_v1_result_2026-09-05.md)
completed both full missions and audit1,040 decisions/1,051 observations after
1,155 tests. Both pass first scan/alignment; south completes two local arrivals.
Whole-task success remains0/2. North stops on weak lateral translation constraints;
south stops when about3.5 cm of scan drift consumes the wall margin. All1,051
depth checks pass; no contact is recorded. No collector/audit remains live and
all316 launched sources are frozen. Next implement explicit uncertain inertial/
depth fusion and dynamics-aware observation actions, with matched supervised/JEPA
turning-prediction tests as detailed in the latest result. Do not merely reduce
the rank threshold or turn margin.

Current result: [release-aware navigation](go2_release_aware_navigation_development_v1_result_2026-09-05.md)
completed both missions and full audit853 decisions/864 observations after1,135
tests across103 files. South passed alignment and advanced onto its second
traversal, but then lost vertical-motion observability as floor support vanished.
Its selected target was beyond a directly measured end wall. North still failed
heading settling. Whole-task success remains0/2; all298 launched sources are frozen
and no run/audit remains live. Next validate targets/stopping paths against
blocking observations, preserve motion observability, and use measured heading
stabilization through holds. Do not treat another tolerance change or estimator-
only recovery as sufficient for the actual planning defect.

The [measured-line/integral successor result](go2_measured_line_integral_navigation_development_v1_result_2026-09-05.md)
is complete and fully audited: both trials reach one arrival and branch scan,
all780 motion intervals and782 depth-frame checks pass, but both still fail
alignment and neither completes the mission. The controller reaches tolerance
then leaves it during zero-command release before the dwell completes. Next
implement release-aware alignment with a tighter inner control target and measured
zero-command settling, preserving outer acceptance and the12-s deadline. Tests
passed1,121 across101 files; all290 launched source files remain frozen. No live
collector or audit remains; do not restart either completed attempt.

Current result: [measured-region closed-loop navigation](go2_measured_region_navigation_development_v1_result_2026-09-05.md)
completed both missions and full audit500 decisions/511 sensor records. Neither
mission succeeded. North now reaches one measured local arrival and a branch scan,
then times out at0.022-rad heading error; south loses translation observability
during near-target steering. Depth checks retain four north failures. The depth
observer is now connected to control; older statements below describe earlier
milestones. Next test a separately named fixed-lookahead approach and bounded
integral alignment successor, keeping acceptance, deadlines, missing-state stops
and physical metrics unchanged. Do not repeat the completed attempt or treat
tests/offline control replay as proof of improved physical navigation.

Latest moving execution: [continuous RGBD relative-state result](go2_moving_rgbd_whole_task_development_v1_result_2026-09-05.md)
completed and full audit passed. All208 intervals have full estimates, worst
step error0.956 mm and worst final error0.164 mm. Moving depth passes209/210
frames; one lower-wall-edge ray failure remains recorded. The observer did not
change commands, and both missions reproduce the original arrival failure.
Next implement measured-geometry control, not further motion-only repetition:
track visible corners and censored opening supports, accumulate full-view ray
evidence with relative-state uncertainty, then select an observed region with
whole-body turning clearance. North's zero depth jumps despite visible corners
rules out relying on depth jumps alone as the portal detector.

Latest execution: [single-sample RGBD actual-render check](go2_single_sample_rgbd_observation_development_v1_result_2026-09-05.md)
completed both cases and full raw audit. All ten prospective visual-surface
checks pass (maximum0.393 mm against5-mm tolerance). Actual framebuffer and
floor identities are verified. The separate physical-floor-reference checks
still fail, correctly preserving the5-mm visual/collision floor offset; the
original V1 failure remains unchanged. Next implement observed boundaries,
relative-motion observability and whole-body arrival/turning clearance within
the continuous mission. Further stationary repetition is not the priority.

The [local-surface measurement front end](go2_depth_local_surfaces_development_v1_2026-09-05.md)
is now implemented with analytic tests and actual-packet replay. It preserves
unknown and thin-obstacle evidence and identifies conditional weak translation
directions. The moving estimator above now uses the full-view depth and causal
gyro; neither observer is yet connected to navigation decisions. Their output
alone does not certify arrival or turning.

## Why the priority changed

The [whole-task pilot](go2_whole_task_navigation_sampling_correction_development_v1_result_2026-09-05.md)
ran and was fully audited, but all four attempts stopped before discovery or
return. The controller's global floor-mask-change gate rejected real translation;
earlier fixtures also showed false arrivals and unsafe scans. Another image-change
threshold or fixed travel-distance adjustment would leave the missing state
unchanged: opening position, body-to-boundary relation and turning clearance.

The goal remains RGB-plus-deployment-valid-sensor JEPA navigation, online memory,
hidden beacon discovery and return in independent novel mazes. This stage is
not a replacement goal and does not require turning the final system into a
purely hand-designed navigator. It supplies the measured state and physical
operators needed for meaningful learned-prediction/planning comparisons.

## 1. Establish a measured geometric observation, not an oracle substitute

Keep the RGB-only baseline. Implement a separately declared RGB-plus-range
condition, initially a rigidly mounted depth-camera stream with an explicit
deployment counterpart requirement. Inspect the installed renderer's depth
semantics before implementation: optical-axis depth versus Euclidean range,
units, projection, near/far clipping, invalid returns and timestamp. Obtain
observations through the actual camera renderer, never analytic queries against
maze wall boxes, simulator pose, cell labels or a teacher route.

Use fixed intrinsics/extrinsics and sensor-local coordinates. Give depth its own
causal packet contract, validity mask, acquisition/availability timestamps,
calibration identity and range limits. Keep evaluator-only geometry out of the
runtime packet. Initial ideal simulated range is a declared development condition,
not a hardware-calibrated sensor; noise, holes, timing errors and extrinsic errors
must be evaluated before transfer claims. No BEV target or map-label supervision
is introduced by this direction.

One-time interface checks must include real rendered frames and metric-depth
verification against evaluation-only geometry, occlusion/invalid returns and
RGB/depth registration. Then connect it to the existing continuous mission;
do not replace navigation work with a long series of isolated sensor probes.

## 2. Replace the missing state, not merely the failed predicate

From current sensor evidence estimate local wall/free-space boundaries and
opening endpoints with uncertainty. Retain unobserved/occluded space as unknown.
Use local point/ray or surface evidence, not privileged cell identities. Track
relative motion from actual observations and body sensing; command integration
may remain a diagnostic baseline, not a distance certificate.

Arrival must refer to an observed boundary and the articulated body's support,
with explicit uncertainty and post-stop motion checks. Turning requires observed
clearance for the motion envelope, not just successful forward traversal or an
instantaneous torso box. A forward camera does not observe a360-degree sweep:
do not mark unseen side/rear volume free. If the footprint cannot be observed
from available views, the controller must reposition under measured clearance,
stop, or use an explicitly distinct sensor configuration with that coverage.

Preserve state-estimation failures separately from gait/control failures. Test
mislocalized boundaries, featureless/repeated views, inadequate clearance,
sensor dropout and estimator uncertainty before enabling motion. Keep the
existing false-arrival and unsafe-turn fixtures as development regressions;
do not enlarge layouts to erase them.

## 3. Exercise exploration, discovery and return end to end

Connect measured geometry to the existing marker and episodic-memory interfaces.
Keep visit-event IDs distinct from recognized places; make uncertain reverse
traversals reobserve their target opening. Initially retain a transparent local
control baseline so sensing/arrival changes can be separated from learning.

Run a new explicitly specified continuous development experiment with the
original two whole-task layouts as regressions and additional topology/appearance
variation. Record every outcome, including failures before discovery. Only make
a memory-contribution claim once trials actually exercise return decisions; match
sensing, motion budgets, home cues and other behaviour, and report route-based
stopping as part of the existing memory ablation. Preserve the0/4 pilot unchanged.

## 4. Make the JEPA comparison answer the scientific question

After continuous execution is viable, collect action-conditioned trajectories
covering the actions actually used: straight motion, braking, pure turns,
alignment, reversals, mixed actions and switched-action prefixes. Include difficult
and unsafe alternatives in simulation; a bank of uniformly safe short actions
cannot test hazard prediction. Separate training and development by layout.

Compare matched supervised predictors and JEPA predictors with the same sensors,
data, action candidates, model budget, runtime and task controller. Separately
compare single-step scoring and genuinely multi-step online rollout. Add a strong
non-learned geometry/memory planner and relevant RGB-only versus added-sensor
controls so sensor information is not mistaken for a JEPA benefit. Inspect whether
models actually choose different actions; identical traces cannot establish a
predictive contribution. A learned actor/distilled policy is a separate design
choice, not implied by training a JEPA world model.

Report whole-maze discovery/return, contacts, stalls, false home claims, path/time,
latency and failures by stage. Freeze final requirements and sample sizes using
development variability before independent evaluation. Multiple model seeds,
independent layouts, sensor/appearance robustness and bounded real Go2 tests
remain required. Hardware absence limits transfer evidence but does not block
the available simulation and software work.

## Immediate next action

Follow the latest depth-floor-hold result: explicit weak-subspace inertial fusion
with validated uncertainty and dynamics-aware observation turns, retaining the
now-executed blocker/view constraints and measured heading holds. Collect action-
conditioned turning data for matched predictive-learning comparisons; preserve
whole-task metrics and do not substitute a local outcome for discovery/return.
The four prior depth
outliers have now been localized to numerical lower-
wall-edge ambiguity and remain FAIL. Preserve rank-deficient state rather than
supplying zero motion. If independent motion evidence is needed, implement causal
sensor fusion with explicit observability and uncertainty.
Once whole-task execution is viable, perform the memory and matched JEPA/planning
comparisons in section4; local progress does not remove those requirements.

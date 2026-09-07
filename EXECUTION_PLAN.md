# Execution plan toward novel-maze navigation

Baseline: 4 September 2026, source freeze `1f7dd8e`.
Direction: [FUTURE_PLAN.md](FUTURE_PLAN.md), the scientific review, and the
physical-interface development result. BEV is excluded.

## Final objective and what counts as completion

A Go2 should explore an unseen maze using RGB and genuinely available body
sensors, discover initially hidden beacons, and return using memory built
online. No hidden map, future observation, simulator pose or oracle goal image
may enter the deployed planner. An ordinary gait controller executes commands.

There are two distinct deliverables: an effective navigation system and a
defensible answer about whether JEPA helps. Completion does not require a
positive JEPA result; a matched negative result is scientifically valid. A
working simulator demonstration alone does not establish physical transfer.

The final evidence package must include independent layouts, paired method
comparisons, maze-level uncertainty, training-seed variation, failure accounting,
runtime/latency and an explicitly bounded real-platform evaluation. Numerical
task thresholds and sample sizes must be fixed from requirements and development
variability before final evaluation, not chosen after inspecting test outcomes.

## Execution order and decision gates

Latest sensor result: [RGBD V1 collection and full audit](docs/go2_rgbd_observation_development_v1_result_2026-09-05.md)
complete two cases/ten actual pairs, but all fixed depth checks fail. Source
inspection identifies visual floor at-.005 m versus collision plane0 and the
combined-render multisample depth path.1,055 tests pass across92 files;253 source
paths are bound and no run is live. Next separately test depth-only single-sample
capture and explicit native visual/collision geometry, without threshold relaxation
or V1 rescoring, then feed measured geometry into the continuous mission below.

Latest result: the [corrected whole-task pilot](docs/go2_whole_task_navigation_sampling_correction_development_v1_result_2026-09-05.md)
completed4 trials and full audit400 decisions. All4 fail the first local visual-
change arrival gate, with no discovery, return, contact or sensor fault. There
is one exact physical trajectory group; memory benefit was not exercised.
The original infrastructure failure remains preserved. Full1,026 tests across89
files pass; collection95031 and audit14299 are terminal, with243 source bindings
frozen. Next [implement measured local geometry within the whole mission](docs/go2_observed_geometry_whole_task_next_steps_2026-09-05.md),
including an explicit RGB-plus-range condition, observed portal/arrival state
and turning clearance. Do not lower the novelty threshold or rescore this pilot.

Preceding physical interface: [beacon-marker capture and corrected full audit](docs/go2_marker_beacon_development_v1_result_2026-09-05.md)
complete six stationary cases, 5,700 physics samples and 30 actual RGB frames.
The correct marker is detected in all five positive frames and registered once
at frame 2; absent/occluded/red-only/reversed/separated cases register nothing.
Original native-box-padding audit FAIL is preserved; the separate strict decoder
correction passes without changing physics or detector criteria.981 tests pass
across84 files. All six original paths and three correction paths are bound;
no collection/audit runs. This is not maze discovery or return evidence.
Next connect this observer and episodic memory to continuous observed exploration
and return, with fresh branch reobservation, physical home evaluation and a matched
memory ablation. Do not substitute further stationary probes for that whole task.

Latest implementation: [episodic route hypotheses and actual-event replay](docs/go2_episodic_route_memory_development_v1_result_2026-09-05.md)
now retain acquired views, provisional visits and tentative return intents without
inventing places or reverse edges. Thirty new tests pass; the full suite passes
946 across 82 files. Read-only replay of four completed trajectories retains 29
event views, 11 visits and seven attempts, including the false arrival and scan
contact. Two physically distinct arrivals have identical featureless RGB, so
single-frame appearance cannot identify them; acquired multiview context is now
retained, not automatically merged. No return or new physics was executed.
Next implement actual physical-marker detection and the continuous beacon/search/
return controller with this memory; keep independent physical outcomes and the
matched memory ablation. This is not a completed whole-task milestone.

Current result: the [task-acquisition factorial and full audit](docs/go2_task_acquisition_continuation_development_v1_result_2026-09-05.md)
complete28 trials and6,502 exact decision replays. Each of seven arms succeeds2/4.
Early scan stopping removes the negative-corner contact but exposes a false
second arrival; stopped observation recovers the tee arrival then exposes its
unsafe scan. Successful tasks shorten45.4→23.2 s under fixed-forward. All three
learned arms remain identical; no JEPA-specific advantage. Seven exact trajectory
groups, not28 independent mazes; all four fresh baseline traces reproduce prior
physics.916 tests pass across81 files. Sources/protocol are bound; no edits/retry;
no collection/audit is running. Task and full-scan metrics remain separate.

Next implement the [whole-task hypothesis-memory/beacon-return stage](docs/go2_whole_task_hypothesis_memory_next_steps_2026-09-05.md).
Keep visit-event IDs and uncertain return intents distinct from recognized places
and trusted reverse edges, detect physical beacons from actual current RGB,
and evaluate actual return rather than controller completion. Improve observed
arrival/clearance state alongside integration; no fixed-margin sweep or erased
1.2 m failures. A declared integration domain is not independent-maze qualification.

Predecessor result: the [persistent-feedback alignment panel and full audit](docs/go2_persistent_alignment_continuation_development_v1_result_2026-09-05.md)
complete16 trials and4,600 exact decision replays. Initial alignment passes4/4
per method, but first-leg integration is3/4 and two-leg success remains2/4.
One negative fixture sees an eligible branch1.04 s before continued scanning
contacts a wall; the other physically arrives but rejects moving visual evidence,
which becomes consistently positive after stopping. No JEPA-specific separation.
893 tests pass across80 files. New sources/protocol are bound; no edits/retry;
that collection/audit is terminal. Next test stop-to-observe arrival and task-directed
scan stopping separately and together, preserving physical endpoints and explicit
partial-scan accounting. Then integrate uncertain memory, beacon discovery/return.

Predecessor result: the [initial observed-alignment successor and full audit](docs/go2_initially_aligned_continuation_development_v1_result_2026-09-05.md)
complete16 trials and4,652 decision replays. Every method still succeeds2/4;
the negative fixtures now time out during initial alignment without traversing,
rather than later contacting a wall. Zero contact is not improved navigation.
The trace reveals zero-command drift repeatedly breaking the fine-heading dwell.
All three learned arms behave identically; no JEPA-specific benefit appears.
876 focused tests pass across79 files. Sources/protocol are bound; no edits or
restart. That study/audit is terminal. Next test drift-aware alignment feedback
without relaxing endpoints, retain the continuous matched task, and progress
to uncertain observation-based memory, actual beacon discovery and return.

Predecessor result: the [continuous two-leg panel and corrected full audit](docs/go2_observed_continuation_development_v1_result_2026-09-05.md)
complete16 trials and4,164 exact decision replays. Every method, including the
matched fixed-forward control, succeeds2/4; the other two fixtures cross and
release successfully before contacting the south wall during scanning. All
traversal choices remain forward; only three distinct raw trajectories exist.
This is partial continuous branching, not reliable navigation or a JEPA benefit.
The original audit FAIL at the legacy341-frame reader cap is preserved; the
separate806-frame reader correction replays the same evidence without changing
any physical criterion or outcome. The suite passes863 tests across78 files.
That collection/audit is terminal. Its observed-heading successor above tests
the continuous task, retaining the matched control and all scan failures.
Use uncertain observation-based memory for subsequent beacon discovery/return;
do not inject cell identities or claim that a first crossing makes turning safe.

Previous result: the [observed traversal panel and full audit](docs/go2_observed_traversal_development_v1_result_2026-09-05.md)
are COMPLETE/PASS for all20 trials and1,399 controller decisions. Each learned
arm crosses and releases successfully on4/4 local fixtures, but all102 learned
choices are forward and the three arms have identical physics. This establishes
one-transition integration, not a JEPA/learning advantage. Directional gait has
three premature candidates and one missed physical arrival; always-stop has four
timeouts. There are no native contacts or sensor faults. All828 focused tests
across74 explicit files passed before launch, including25 new integration tests.
Sources/protocol are bound; do not tune or rerun that completed study.
Its next step was continuation into a freshly observed branch and second traversal, with a
speed/cadence-matched non-learned control and uncertain place/arrival evidence.
Full exploration, beacon discovery/return and hardware evidence remain open.

Latest execution, 5 September: the causal relative-orientation and repeated
temporal adapter replay passed, including exact equality of 4,080 comparable
predictions and 58 reference orientation streams. The fixed
[successive-choice result](docs/go2_successive_choice_maze_development_v1_result_2026-09-05.md)
now completes144 physical trials and full corrected raw audit. JEPA's half-second
latent head has4/24 contact trials versus12/24 for its matched supervised head,
a bounded local benefit with remaining safety/progress failures—not multi-step
planning or maze exploration. The original clock-boundary checker FAIL and all
physical outcomes are preserved. That panel's collection and audits are terminal.
The preceding focused development suite passed803 tests across71 explicit
files; tracked source remains unchanged. The
[action-coverage diagnostic](docs/go2_successive_action_coverage_development_v1_result_2026-09-05.md)
is complete:530 later training windows all repeat their past action;189 online
switches lack later pair support. The new384-trial moving-prefix collection,
full raw audit and actual600-cell tensor check are all complete and pass.
All103 native contacts are retained, including three release-only contacts
excluded from suffix targets. See the [completed data result](docs/go2_moving_prefix_counterfactual_development_v1_result_2026-09-05.md).
The original914-window-plus384-switch view and context-matched sampler are now
implemented; their13 included focused tests and actual metadata preflight pass.
The [18-model paired comparison and full audit](docs/go2_context_matched_coverage_learning_development_v1_result_2026-09-05.md)
are COMPLETE/PASS. Its fixed sources, models, schedules and predictions remain
bound; no retry, resume or edits. Expanded coverage helps three-second contact
prediction/ranking, but half-second switch labels have zero contacts, retention
is mixed, and JEPA has no established matched advantage. Always-stop wins the
offline cost without completing navigation. Advance actual observation/exit/
arrival integration and separately specified physical task comparisons; do not
promote offline scores into maze claims. The new16-scene active scan, full audit
and corrected orientation diagnosis below are terminal; the paired fast-gyro
scan and full audit are also complete. Those predecessor studies are terminal. See the
[follow-through plan](docs/go2_successive_choice_followthrough_plan_2026-09-05.md).

The [observed exploration/controller bridge and translation replays](docs/go2_observed_exploration_and_translation_development_2026-09-05.md)
now add observed-frontier/discovery/directed-return state and connect fresh exit
bearings to the real local adapter. Place/exit/beacon detection and physical
arrival certification remain upstream gaps, not proven by synthetic events.
Two completed no-refit odometry replays cover144 short streams and26 longer
route/turn streams: short mean errors are a few centimetres, but route endpoints
reach13.1 cm and nominal in-place turns drift up to9.5 cm. Keep command odometry
as a baseline, not a calibrated place-association estimate. The
[fitted bridge and RGB/body observation checks](docs/go2_rgb_body_observation_evidence_2026-09-05.md)
now pass1,114 actual fitted-model selections, evaluate818 RGB floor observations
and replay a sensor-derived ground hypothesis on26 streams. The floor baseline
fails appearance controls, and ground attitude/camera-height errors remain;
these are observation baselines, not qualified place/exit/arrival detectors.

The [new ray-range diagnostic](docs/go2_ground_projection_envelope_development_v1_result_2026-09-05.md)
completed all1,498 route frames, independently checking camera/ground projection
and nine explicit uncertainty hypotheses. Point error peaks at1.18 m despite
small average body-height error; wide intervals and the current-view near-field
gap prevent a clearance claim. Next develop fixed causal gravity feedback and
temporal observation/exit/arrival integration without modifying the completed
context-matched learning study. All projection-study source is now bound.

The [fixed gravity-feedback comparison](docs/go2_gravity_feedback_ground_development_v1_result_2026-09-05.md)
has completed all1,498 route packets with22 new tests and exact gyro-baseline
replay. Both fixed feedback methods improve mean normal/point error on all8
routes; the point-error maximum is still43 cm, and the transported-force variant
does not outperform the simpler body-frame mean overall. Preserve both and
their unqualified acceleration/flat-support assumptions. Next fixed dynamic
excitation and temporal exit/arrival integration must remain distinct from the
unchanged model-training data intervention.

The [fixed16-scene actual RGB/body active scan and full audit](docs/go2_active_exit_scan_development_v1_result_2026-09-05.md)
are COMPLETE/PASS:3,040 exact live decisions,3,104 RGB packets and166,322 native
physical samples. Selected views cover39/40 opening sides without closed-side
proposals, but0/16 scans meet all physical criteria. Four narrow dead-end/corner
scans contact calf links against walls; twelve complete without contact but miss
the fixed final-heading tolerance. No proposal is a trusted traversal or place.

The [corrected numerical diagnosis](docs/go2_active_scan_orientation_diagnostic_development_v2_result_2026-09-05.md)
also completes all16 traces, preserving its empty V1 accounting FAIL. Only four
unique physical-array trajectories underlie the sixteen rendered specimens.
At50 Hz the coning correction barely helps;500 Hz evaluation-only rates reduce
maximum full-scan midpoint heading error from about0.049 to0.0004 rad. Those
privileged replay rates are not runtime IMU inputs. Next implement a separate
causal high-rate measurement/preintegration interface and validate it physically,
address articulated clearance, then execute observation-to-arrival and a small
whole-maze prototype. Follow the [current integration next steps](docs/go2_scan_to_navigation_next_steps_2026-09-05.md)
before another large model sweep. All completed source/results remain bound.

The [eight-trial paired live50/500 Hz gyro scan and full audit](docs/go2_fast_gyro_scan_development_v1_result_2026-09-05.md)
are COMPLETE/PASS:80,355 live fast measurements,1,496 RGB packets and1,464 exact
decision replays. The separate causal channel leaves learned RGB/body tensors,
gait, ground estimator and scan rule unchanged. Both rates succeed3/4 at the
new heading0; both hit the narrow dead-end wall. High-rate maximum completed-scan
heading discrepancy falls from0.03851 to0.000452 rad, but success count does not
improve. Next articulated collision support/swept-volume evidence and actual
proposal-to-traversal-to-arrival integration; no unmotivated training sweep.

The [articulated geometry component and corrected24-scan diagnostic](docs/go2_articulated_scan_geometry_development_v2_result_2026-09-05.md)
complete all4,504 observed postures and six contact trials. All27 nominal
primitives are included; independent foot FK agrees within1.67e−16 m. A torso-only
outline would show20–22 cm of wall separation at actual contacts, while whole-body
support reaches the wall. V1's incorrect native-retention assumption is preserved
as FAIL; V2 resolves merged heads/feet through the actual robot-link roster.
Instantaneous extent is not future swept volume or environment clearance.
Next execute actual proposal-to-traversal-to-arrival integration with explicitly
provisional sensing/memory; narrow-wall and geometry-level ground-support limits
remain open. That diagnostic is terminal; no full-maze result is established.

| Milestone | Work and deliverable | Completion criterion | Current status |
|---|---|---|---|
| M0: coherent interfaces | Source contracts, causal histories, complete commands, proper camera frame, metric negative controls | Source tests distinguish valid and invalid inputs | Done: 116 focused tests; known frozen defects remain documented |
| M1: primitive physics/render agreement | Fresh Genesis obstacle/contact, material and projection assay | Measured checks pass, failures and repetitions retained | Done: 15 checks; not Go2 qualification |
| M2: physical handoff evidence | Establish correct contact measurement, then fresh physical handoff evidence | Trustworthy physical labels and an interpretable terminal result | Fresh eight-case diagnostic complete and raw-audited; original generator remains interrupted |
| M3: reusable local execution | Separate action feasibility, target semantics, visual transfer; then consecutive edge transitions | Correct crossing and viable arrival state under Go2 dynamics; all failures counted | Multi-junction oracle panel audited: 8/8 all crossings contact-free, 4/8 full task; remaining final-heading misses preserved; visual control open |
| M4: real multisensor state | Synchronized RGB, ordered IMU/joint history, command history, validity and relative odometry | Sensor excitation/calibration, lag/reset/dropout tests; no privileged inputs | Live ideal simulated RGB/body capture and policy-only loader audited: 414 packets; hardware sensing and relative odometry remain open |
| M5: decision-useful prediction | Matched persistence, kinematics, direct history-conditioned policy, one-step and rollout models | Executed decision endpoint identifies benefit or credible absence of benefit | Prior negative studies preserved; fresh144-trial successive panel audited, JEPA first-transition head4/24 contacts versus supervised12/24; safety, causal switch coverage and multi-step planning remain open |
| M6: online memory and integration | Build graph from observations, provisional associations, beacon discovery, exploration and return | Novel layouts navigated without oracle graph; localization and execution failures separated | Observed-frontier/discovery/return state and local-controller bridge implemented; RGB place/exit/beacon perception and physical arrival not qualified; no full-maze evidence |
| M7: generalization and transfer | Locked methods, fresh layouts/shifts, bounded real Go2 pilot | Independent task-level evidence and honest transfer limits | Not started |

Preparatory sensor and memory work can proceed alongside local execution, but
neither substitutes for M2/M3. Learned hierarchy and larger models are deferred
until a measured limitation motivates them.

## Current M2 recovery, 5 September

The original collection was stopped for a reproduced measurement defect, not
because of elapsed time or an unfavorable scientific result. The inherited
contact detector mixes the environment/contact force axes. See the
[integrity report and native-assay specification](docs/go2_contact_measurement_integrity_2026-09-05.md).
All ten development files were restored and independently hash-checked; no
material or frozen source was deleted or changed. Session 96850 is terminal.
Do not poll it as live, resume the partial stream, or rerun the old generator.

The new per-contact adapter has passed the specified native Genesis assay:
six checks, with raw packets preserved and identical across the documented
reference-calculation correction. The fresh eight-case contact-attributed Go2
experiment has now completed and passed raw-artifact recomputation: seven
contact-free crossings, but only two usable arrivals under the declared checks.
See the [results and next scientific step](docs/go2_contact_attributed_execution_development_v1_result_2026-09-05.md).
The original interruption remains an integrity failure, not its requested
scientific terminal; the new study supplies a separate interpretable development
result. Previous generator contact counts remain recorded flags, not verified
collision incidence.

Current M3 update: the 32-trial prealignment-by-arrival-feedback comparison is
complete and audited. Combined control scored 3/8 on two crossings plus final
instantaneous arrival; other arms scored 0/8. No arm produced sustained final
stopping or two usable arrivals. First-arrival proxies did not reliably determine
continuation. See the [result](docs/go2_local_control_factorial_development_v1_result_2026-09-05.md).
Native readback then confirmed a more fundamental adapter mismatch: joint gains
100/10 rather than the checkpoint's 20/0.5. The separately specified sixteen-trial
[gain comparison](docs/go2_actuator_gain_pair_development_v1_result_2026-09-05.md)
completed and passed raw audit: default 0/8 task successes, checkpoint gains 8/8,
with 8/8 sustained final arrivals and no contacts in the corrected arm. Six cases
meet both arrival checks; both right turns miss first heading but continue.
Use this corrected development reference for the
[next execution/sensor/JEPA packages](docs/go2_rgb_multisensor_execution_next_steps_2026-09-05.md).
The four motifs and correlated width pairs do not qualify unseen-maze navigation.
The subsequent [live RGB/body command-capture package](docs/go2_causal_rgb_body_capture_development_v1_result_2026-09-05.md)
also completed and passed audit: nine probes, 414 RGB/history packets and 2,700
ideal simulated sensor samples, all contact-free with successful release motion.
Command tracking is measured, including asymmetry and lateral drift. The next
collection should reuse this acquisition path on actual multi-junction
development routes and matched alternative actions, not repeat the simple arena.
The [multi-junction route study](docs/go2_multijunction_route_development_v1_result_2026-09-05.md)
has now completed: all eight routes crossed every planned opening without contact,
four passed the complete final-arrival contract, and the other four missed only
final heading. Its 1,498 causal RGB packets passed the full audit. The next
scientific step is matched alternative-action data on independent procedural
development layouts, followed by direct-action versus JEPA models; do not wait
for every heading proxy on the current motifs to become perfect.
RGB endpoint images are now real fixed-mount captures, but the teacher still
uses privileged pose; this is not a deployable visual controller.

Current M5 update: the 24-layout, five-action counterfactual corpus completed and
passed raw audit, with sixteen training and eight development-validation layouts fixed
before collection. The original attempt stopped after nineteen branches because
exact prefix RGB equality failed. Raw physical states and causal histories match
exactly; the observed mismatch was sparse pixel variation. The separately
documented [V2 recovery](docs/go2_counterfactual_maze_dataset_development_v2_recovery_2026-09-05.md)
retains those nineteen branches and executes only the 101 untouched pairs.
It verifies bounded image variation and selects one actual canonical model
context per layout; no physical branch is repeated or image edited. See the
[completed dataset result](docs/go2_counterfactual_maze_dataset_development_v2_result_2026-09-05.md):
120 branches, 24 contact stops, 8,966 causal RGB packets, and 891/960 valid motion
targets with explicit censoring. The observed local contexts still comprise four
junction motifs; unique unvisited topology does not establish maze-task coverage.

The [fixed learning comparison](docs/go2_rgb_body_learning_comparison_development_v1_2026-09-05.md)
completed after full audit and all-row loader checks: direct, supervised rollout without latent prediction, and JEPA,
with matched observation exposure and three seeds. The extra supervised-rollout
control distinguishes latent prediction from extra outcome supervision. Models
consume only RGB, ordered body/control history and prospective action plans;
future observations and privileged motion/contact labels remain training targets.
Full raw corpus audit passed before fitting. Neither source tests nor
offline prediction metrics complete the executed-choice or maze-navigation gates.

The [learning result](docs/go2_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
is now audited. An action-only average beats every learned model on motion;
JEPA improves motion versus neural controls but not contact prediction. In a
separately specified conditional analysis of actually executed branches,
supervised rollout without latent prediction has zero selected contacts and
mean regret 0.147, versus JEPA/direct always stopping (0.297) and JEPA/rollout
regret 1.926 with 16.7% selected contacts. RGB shuffle strongly degrades the
supervised-rollout choices. These are eight-layout offline conditional results,
not online policy execution. Proceed with the
[post-comparison plan](docs/go2_rgb_body_post_comparison_next_steps_2026-09-05.md):
strict sensor-to-command adapter, bounded fresh online conditional-choice pilot,
then broader temporal/state coverage and online memory. Preserve the negative
JEPA result; do not launch a coefficient search.

The strict sensor-to-command adapter now passes its
[72-call checkpoint/packet replay](docs/go2_online_local_choice_adapter_development_v1_result_2026-09-05.md),
with exactly matching per-seed predictions. It uses all three fixed seeds and
permits only one local decision per episode; repeated decision states were not
covered by the outcome-head training. The separately frozen
[fresh online pilot](docs/go2_online_choice_maze_pilot_development_v1_2026-09-05.md)
completed on eight topology-disjoint layouts with three intents and three
methods. It reads each trial's actual current RGB/body packet, not a canonical
reference image. Teacher initialization and privileged simulator emergency stops
remain explicitly experimental. Do not infer a final result from partial trials.

The [full online result](docs/go2_online_choice_maze_pilot_development_v1_result_2026-09-05.md)
has now passed audit: all72 prefixes available, 5,660 RGB/history packets,
supervised rollout0/24 contacts and cost0.7077, JEPA2/24 contacts and cost1.7151,
all-stop cost0.7994. Supervised-minus-stop interval [−0.2341,+0.0827] crosses zero.
Risk/readout errors sometimes cause motion away from a sideways intent; actual
packet-to-command replay verifies this is the chosen policy behavior. Preserve
this negative/limited evidence, and do not change completed model costs.

The [paired18-trial gyro turn assay](docs/go2_gyro_turn_assay_development_v1_result_2026-09-05.md)
has now completed and passed raw audit. Gyro feedback meets all physical checks
in9/9 arena trials versus timed turning6/9; timed half-turns undershoot. Feedback
is slower and not more accurate on every angle. These correlated ideal-sensor
arena trials do not qualify narrow-clearance turns or real IMUs.

The next [temporal data package](docs/go2_causal_subtrajectory_development_v1_result_2026-09-05.md)
has also been derived, raw-audited and fully tensor-loader checked:914 actual causal windows,3,852 valid motion
targets and4,236 valid contact targets, preserving16train/8validation layouts.
Four past RGB/body/control packets, remaining-plan masks and strict contact
censoring support a future history-conditioned training comparison. Later
contexts contain only their executed branch's outcome; never call them five-way
counterfactuals. No old model or completed scientific protocol was edited.
The [temporal model/loss interface](docs/go2_temporal_model_interface_development_v1_result_2026-09-05.md)
now passes48 actual-window/no-fit checks and its22 focused tests. The
[three-arm, three-seed design](docs/go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md)
fixes1,200 updates per model, layout-balanced schedules and final checkpoints
only. The [full study](docs/go2_temporal_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
has now completed and passed all nine checkpoint/metric audits. JEPA-direct has
worse contact Brier and initial-choice regret than supervised-direct;
JEPA-rollout is worse on contact, motion and initial choices. Its70/72 direct-head
stop selections and10/72 contacting rollout selections do not demonstrate useful
navigation. These are correlated offline selections, not new physical collisions.
Follow the [post-comparison steps](docs/go2_temporal_post_comparison_execution_next_steps_2026-09-05.md):
relative orientation and a repeated online adapter, then a fixed fresh physical
directional-continuation panel that records action-switch failures. Later-state
training windows do not supply all candidate outcomes after movement. A separate
[live RGB-history replay](docs/go2_online_rgb_history_development_v1_result_2026-09-05.md)
has passed on120 recorded streams. Follow with fresh successive sensor-only decisions and
observed exits/place memory. Feedback-generated future commands must not leak
into JEPA candidate conditioning.

## Historical runbook: interrupted M2 attempt

User approval permits temporary relocation of only our new development files
to an ignored staging directory and restoration afterward. Do not export or
copy the repository, move tracked source, change the frozen runner, alter its
scientific thresholds, or inspect protected benchmark material.

1. Record an explicit file inventory, byte hashes and current HEAD. Stage only
   those new files, verify their hashes, and check the remaining worktree is
   clean at the exact existing freeze. Preserve a recovery manifest outside the
   tracked source area. Restore each file without overwriting unexpected edits
   on completion, failure or interruption.
2. Validate the initialized runtime and the existing stream using the frozen
   validators. Preflight found a completion filename for straight-passage
   stratum 00, 63 untouched streams, and no partial directories. Filename
   presence is not scientific validation and must not be counted as such.
   Do not call `initialize` on the already existing output roots.
3. Execute untouched streams sequentially in their declared order. Each stops
   after four qualified states or its 64-attempt cap. Do not rerun completed
   streams or resume a partial stream. Monitor process output and preserve
   failures; a crash is not an unfavorable model result.
4. Run the frozen generator reduction only after all streams have valid
   completion evidence. If any stratum misses its target, publish the allowed
   generator terminal and stop downstream work. Do not compensate by reading
   ranker outcomes, adding attempts, or changing selection rules.
5. Only if its gate passes, the actual frozen dependency order is: selected-state
   reset qualification; panel freeze; canonical encoding; fanout for the 48
   development states only; development target selection; fanout for the 16
   development-held-out states; held-out scoring; prescribed held-out repeats;
   row-evidence assembly; independent reduction/publication validation.
   Obtain state IDs from the validated stage outputs, not guessed identities.
6. Record the final classification and decide the next experiment below.
   Restore staged files and update this plan. M2 completion is a terminal
   interpretable result, not necessarily a successful handoff system.

Physical collection may take substantial time. Report elapsed progress and
stream counts; do not invent an ETA before measuring throughput. Do not change
threading, physics, image preprocessing or model inputs to accelerate a frozen
attempt. No learned-model training belongs to this existing workflow.

## M2-to-M3 decision rule

- Inadequate panel: quantify rejection categories and feasibility conditional
  on the generator. Propose a separately versioned generator/controller study;
  model performance remains unmeasured.
- Adequate panel but no viable candidate: change the future local target/action
  bank or execution controller, not the JEPA loss. Include stopping, turning,
  reversing and successive maneuvers with explicit arrival constraints.
- Viable candidates but poor selection: separate image-domain shift, target
  support and input history mismatch using matched controls.
- Reliable single edge: qualify two consecutive edges, then longer routes.
  A reached waypoint with unusable heading/speed is not successful continuation.

Each new empirical experiment gets a bounded question, immutable data roles,
baseline matrix, primary endpoint, numerical acceptance criteria and stopping
rules before outcomes. Development iteration is allowed; retrospective changes
must not be relabeled as preregistered evidence.

## JEPA and sensor comparison after local execution works

Build a causal state from spatial RGB features, ordered measured body history
and separately represented control history. Use independent validity and time
information; future measured sensors are targets, never candidate inputs.
Compare mean pooling with an ordered temporal encoder under matched capacity
and data. Include range/depth only after actual availability is established.

Compare frozen-feature prediction with joint adaptation and an appropriate
anti-collapse mechanism. Measure relative motion, local reachability and viable
arrival state as well as feature fidelity. Separate predictive-pretraining
benefit from the benefit of running rollouts online. Use actual counterfactual
branches and executed action regret, not latent cosine alone, to select the
next model development question.

The integration experiment crosses rollout/no rollout with persistent
memory/no persistent memory, retaining comparable recent history in every cell.
Match sensors, action proposals, gait controller, safety layer and task budgets;
report unavoidable compute differences. Add sensor ablations and simpler place
descriptors. Test unknown-goal exploration separately from return to an observed
goal. Include wrong merges, missed revisits, falls, contact, intervention,
timeouts and latency in the report.

## Persistence, boundaries and handoff

Continue through safe in-scope implementation and validation; stop a scientific
attempt at its declared terminal condition. Do not run endless parameter searches
until a metric becomes favorable. Preserve negative and infrastructure results.
If a next action needs a physical robot, destructive recovery, a new experiment
authority or a material change to the agreed task, record the exact blocker and
request the needed decision instead of claiming completion.

Current final-goal status: not achieved. M0/M1 and the 470 combined plus9 live-history focused checks are
foundations only. Development files are restored; causal sensor history and
directed routing are integrated but not empirically qualified for navigation.
Native contact measurement and the fresh eight-case Go2 evidence are audited.
The effective development actuator contract and ideal simulated command/sensor
capture are measured. Action-diverse data and the nine-model comparison now have
audited results. The strict online local-choice adapter has passed replay; the
fresh conditional pilot is complete and fully audited. Gyro turning has bounded
arena evidence, and causal subtrajectory targets now broaden decision-state
coverage within the existing corpus. The temporal model interface and fixed
comparison have completed with a full audited negative/limited JEPA result.
Next are successive sensor-only local execution with explicit action-switch tests
and online memory. Corrected-gain actual multi-junction
execution is measured; deployable visual control and novel-maze task
generalization remain unestablished.
Do not rely on the interrupted detector's
labels, or equate a contact-free camera view with whole-body traversability.

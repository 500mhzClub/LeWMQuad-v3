# Future plan: RGB and multisensor JEPA navigation on a quadruped

Date: 4 September 2026. Planning baseline: repository commit `1f7dd8e`.

This plan preserves the conclusions of the repository and final SAINTS progression-document review and translates them into proposed work. The [full scientific review](docs/scientific_review_and_navigation_plan_2026-09-04.md) contains the detailed evidence, source references, and literature comparison. The superseded BEV direction is outside this plan.

Status: proposed research direction. Writing this plan does not launch experiments, change frozen protocols, or supersede repository custody instructions. Existing experiments retain their own scope and stopping rules.

Latest sensor execution: [actual RGBD V1 and full raw audit](docs/go2_rgbd_observation_development_v1_result_2026-09-05.md)
complete two cases/ten pairs. All fixed depth checks fail despite faithful capture:
the renderer's visual floor differs from collision ground and combined RGB/depth
uses multisample resolve. The separate causal range interface is implemented,
but metric navigation use is not qualified. Next a source-backed single-sample
acquisition and native visual/collision identity check, then observed geometry
within continuous discovery/return. No relaxed threshold, old-result rescoring,
new maze success or JEPA benefit is claimed; no run remains live.

Latest execution: the [corrected whole-task pilot](docs/go2_whole_task_navigation_sampling_correction_development_v1_result_2026-09-05.md)
completed4 trials with full400-decision replay. All4 stopped at the first local
visual-change gate: no beacon discovery, return, contact or memory comparison
actually exercised. One identical physical trace group is not4 independent results.
The original infrastructure failure stays preserved;1,026 tests across89 files
pass and no collection/audit runs. Next [measured geometry within the continuous
mission](docs/go2_observed_geometry_whole_task_next_steps_2026-09-05.md): explicit
RGB-plus-range observation, actual boundary/arrival state and turning clearance,
then discovery/return and matched JEPA comparisons. Do not fit the novelty gate
to these failures or claim that audit PASS establishes navigation success.

Latest physical interface: [actual beacon-marker acquisition](docs/go2_marker_beacon_development_v1_result_2026-09-05.md)
passes the fixed six-case stationary response check and corrected full raw audit:
30 RGB frames, five positive detections, no absent/occluded/distractor detections,
one distinct pattern discovery. Original audit failure is preserved; correction
only decodes native box padding.981 tests pass across84 files. No live process,
new navigation success or JEPA contribution is claimed. Next combine the actual
marker observer with episodic memory in continuous explore/discover/return trials.

Latest implementation: [episodic route hypotheses](docs/go2_episodic_route_memory_development_v1_result_2026-09-05.md)
support uncertain online visit history, acquired multiview context and freshly
observed return candidates, without trusted place/edge claims. Thirty new tests
and a read-only four-trajectory event replay pass; the full suite passes 946 tests
across 82 files. Identical featureless images at distinct arrival locations expose
single-frame association ambiguity. No physical return has been attempted by
this component. Next connect it to actual beacon perception and continuous
exploration/return, preserving independent outcome checks and memory ablations.

Current result: the [task-acquisition factorial and full audit](docs/go2_task_acquisition_continuation_development_v1_result_2026-09-05.md)
complete28 trials and6,502 exact decision replays after916 passing tests across81
files. Early scan stopping improves efficiency and avoids one contact; stopped
observation recovers a missed arrival. Yet each arm still achieves2/4 tasks,
with premature arrival and turning-clearance failures retained. Task and strict
full-scan metrics remain separate; no JEPA-specific separation. Sources/protocol
are bound, and no collection/audit runs. Next implement the
[uncertain-memory, actual-beacon and return prototype](docs/go2_whole_task_hypothesis_memory_next_steps_2026-09-05.md),
alongside improved observed arrival/clearance geometry, not another margin sweep.
The unchanged full goal still requires observation-based memory, beacon discovery
and return, matched predictive comparisons, independent mazes and hardware evidence.

Latest implementation update: the [384-trial moving-prefix collection](docs/go2_moving_prefix_counterfactual_development_v1_result_2026-09-05.md)
completed and passed full raw audit; all600 composite cells pass actual tensor
and policy-I/O checks. Predecessor studies are terminal, including the paired
fast-gyro scan and its full audit. The preceding suite passed803 tests across71 files,
including21 new articulated-geometry/grouping tests; actual metadata preflight also passes.
The [18-model comparison and full audit are complete](docs/go2_context_matched_coverage_learning_development_v1_result_2026-09-05.md).
Expanded action coverage improves three-second contact prediction/offline ranking,
but JEPA shows no general matched-baseline advantage and old-window retention is
mixed. All128 switched validation actions are contact-free at0.5 s: that primary
Brier endpoint measures false alarms, not hazard discrimination. Always-stop
beats every learned head on the three-second offline cost but cannot navigate.
Next prioritize actual observation/exit/arrival integration and a separately
specified physical comparison with progress/stall/task outcomes; no model promotion.
The [completed16-scene actual RGB/body scan and raw audit](docs/go2_active_exit_scan_development_v1_result_2026-09-05.md)
now cover39/40 opening sides at selected views, with no closed-side proposals.
However, four scans contact calf links against walls and twelve miss the fixed
final-heading tolerance:0/16 full scan successes. The
[completed corrected orientation diagnosis](docs/go2_active_scan_orientation_diagnostic_development_v2_result_2026-09-05.md)
finds sampling bandwidth, not the tested coning correction, dominates the ideal
gyro error. Its500 Hz physics-derived rates are evaluation-only, not policy inputs.
Next implement a causal high-rate IMU channel, address articulated clearance,
and build the observation-to-arrival/full-task prototype in the
[current integration plan](docs/go2_scan_to_navigation_next_steps_2026-09-05.md).
No memory edge will be created merely because an RGB floor extension is visible.
The [live fast-gyro comparison now completes with full raw audit](docs/go2_fast_gyro_scan_development_v1_result_2026-09-05.md):
both50/500 Hz arms succeed3/4 at a new heading, while500 Hz greatly reduces
orientation error. Both retain narrow-wall contact. The
[completed articulated geometry analysis](docs/go2_articulated_scan_geometry_development_v2_result_2026-09-05.md)
explains why torso-only clearance would miss those contacts and corrects native
fixed-link grouping. Current joint-derived extent is implemented; future sweep
and environment clearance are not qualified. Next actual observation-to-traversal-
to-arrival integration, preserving those limitations.

The [first observed-traversal panel and full audit](docs/go2_observed_traversal_development_v1_result_2026-09-05.md)
now complete20 trials and1,399 exact controller replays, with no contacts or sensor
faults. Each learned arm succeeds4/4 but all choose forward on every decision and
their physics are identical: no learned-method separation. The slower directional
baseline has three premature candidates and one missed physical arrival, so
command accumulation/image change remain unqualified arrival proxies. The828-test
suite passed before launch. This is a hybrid world-model/controller integration,
not an end-to-end learned navigation policy or trusted online map. Next actual
branch/second-traversal continuation with a speed/cadence-matched non-learned
control, then observed beacon/memory return. Independent maze comparisons,
multi-step planning and real hardware remain open. That traversal study is terminal.

The [continuous branch/second-traversal panel](docs/go2_observed_continuation_development_v1_result_2026-09-05.md)
now completes16 trials and corrected full audit of4,164 decisions. Fixed-forward
and all learned arms each succeed2/4, with identical physical traces; eight scans
contact a south wall after successful first arrivals. A useful arrival must
support the next maneuver, not just a crossing and stop. The original reader-cap
audit FAIL is preserved; the new reader only implements the already specified
806-frame budget, leaving physical evidence unchanged. The863-test/78-file suite
passes. Next observation-driven alignment/centering in the continuous task, then
uncertain online place/branch memory and actual beacon/return integration. Pure
in-place scanning is outside the frozen learned action bank and is hand-controlled;
its failures do not isolate JEPA prediction quality. No new model sweep is warranted
by this result alone. Full maze and hardware requirements remain unchanged.

This addresses measured missing action-switch coverage; it does not add new
independent layouts or demonstrate exploration/return. Preserve the bounded
positive successive-control result and the earlier negative comparisons below.

The [completed1,498-frame ground-projection diagnostic](docs/go2_ground_projection_envelope_development_v1_result_2026-09-05.md)
now exposes a perception limit: nominal point error averages4.9 cm but reaches
1.18 m, while broad conditional uncertainty intervals often span metres. The
sampled views supply no visible-floor rays within0.5 m optical depth. Improve
causal attitude estimation, temporal evidence and actual exit/arrival detection;
do not treat current-frame palette floor as a clearance certificate.

The [fixed gravity-feedback follow-up](docs/go2_gravity_feedback_ground_development_v1_result_2026-09-05.md)
now improves pooled mean projected point error to2.6–2.7 cm, with improvement
in every old route and
maximum error still about43 cm. Both feedback variants help; transported vectors
do not beat the simpler body-frame mean overall. Preserve both, test fresh
dynamic excitation, and keep this sensor-estimator change separate from the
completed JEPA data-coverage comparison. This is not qualified clearance or yaw.

[Observed exploration integration and translation evidence](docs/go2_observed_exploration_and_translation_development_2026-09-05.md)
now connect symbolic observed exits to the causal local controller. Completed
144-stream and26-stream no-refit replays establish a useful short-range command
odometry baseline but show accumulated route/turn drift. RGB place/exit/beacon
observations and actual arrival qualification remain missing; neither synthetic
memory tests nor centimetre-scale short-term error proves full-maze navigation.

The [fitted bridge and RGB/body observation evidence](docs/go2_rgb_body_observation_evidence_2026-09-05.md)
now covers1,114 matched choices,818 visible-floor observations and26 ground-plane
replays. Palette dependence and attitude/height errors are explicit. These do
not qualify actual place identity, footprint clearance, arrival or transfer.

Current autonomous implementation status, 5 September: local actuator correction,
causal RGB/body acquisition and multi-junction oracle execution have measured
development evidence. A fresh 24-layout counterfactual corpus has completed and
passed raw audit, and the fixed nine-model direct / supervised-rollout / JEPA
comparison has completed and passed checkpoint/prediction audit. Conditional
executed-branch choices favor supervised rollout without latent prediction;
JEPA navigation utility remains unsupported. See [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for the current sequence and
[result and next steps](docs/go2_rgb_body_learning_comparison_development_v1_result_2026-09-05.md).
The strict local-choice adapter subsequently passed 72 actual-checkpoint packet
replays, and the fresh 72-trial online conditional pilot completed with full
raw-data/selection audit. Supervised rollout was contact-free but its mean cost
advantage over stopping is uncertain across eight layouts; JEPA was worse with
two contacts. See the [online result](docs/go2_online_choice_maze_pilot_development_v1_result_2026-09-05.md).
The fixed sensor-only [turn assay](docs/go2_gyro_turn_assay_development_v1_result_2026-09-05.md)
now passes raw audit: gyro feedback succeeds9/9 versus timed turning6/9 in a large
arena, with speed/accuracy tradeoffs and no narrow-maze or hardware claim.
The [causal subtrajectory package](docs/go2_causal_subtrajectory_development_v1_result_2026-09-05.md)
derives914 windows from the same24 layouts; raw label/plan audit and full
policy-only tensor-loader check pass. This
broadens temporal supervision, not independent sample size or later-state
counterfactual coverage. Repeated replanning, memory and full exploration/return
remain open; their new model comparison is now complete and fully audited.
The [temporal model interface](docs/go2_temporal_model_interface_development_v1_result_2026-09-05.md)
now passes actual-input/no-fitting checks, with452 focused tests overall. Its
[matched training design](docs/go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md)
has now completed all nine models and passed full audit. The
[result](docs/go2_temporal_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
does not support JEPA utility: its direct head is more conservative, while its
rollout head has worse contact/motion/initial-choice outcomes than the matched
supervised arm. Next test actual action changes after movement using the
[post-comparison execution plan](docs/go2_temporal_post_comparison_execution_next_steps_2026-09-05.md).
A separate
[live history buffer/replay](docs/go2_online_rgb_history_development_v1_result_2026-09-05.md)
prepares the input path for later replanning without changing this experiment.
The [temporal adapter/orientation replay](docs/go2_temporal_online_adapter_replay_development_v1_result_2026-09-05.md)
has now passed: 4,080 comparable predictions match exactly and both fixed
orientation-error limits pass across 58 streams. The combined development suite
passed 519 tests before launch. The separately frozen [144-trial successive-choice panel](docs/go2_successive_choice_maze_development_v1_result_2026-09-05.md)
has completed and passed corrected full audit on eight fresh topology-disjoint
layouts. Its half-second JEPA latent head has4/24 contact trials versus12/24 for
the matched supervised head: a bounded positive local result, not safe maze
exploration or multi-step planning. Prior negative comparisons remain valid.
The [action-coverage diagnostic](docs/go2_successive_action_coverage_development_v1_result_2026-09-05.md)
confirms all530 later training windows continue their prior action;189 actual
online switches lack later training-pair support. Next implement the prepared
moving-prefix data intervention and continue toward observed maze memory.
The chronological updates below preserve previous findings; their old “next”
steps are superseded by the current execution plan. Final-goal success remains
unestablished: no learned closed-loop maze policy or hardware result yet.

Implementation update, 4 September: the first source-specification and synthetic-test package is complete. See the [active-interface specification and diagnostic results](docs/go2_active_interface_specification_2026-09-04.md), [prospective in-memory guards](lewm/interface_semantics.py), and [focused semantic tests](lewm/tests/test_active_interface_semantics.py). The explicit CPU suite passes 94 tests, including 11 existing renderer-wrapper checks. It reproduces several limitations; passing diagnostic tests does not mean those scientific interfaces are qualified. No frozen implementation, dataset, checkpoint, or experiment result was changed, and no physical experiment was launched.

Second implementation update: an [independent Genesis physical-interface assay](docs/go2_physical_semantics_development_v1_2026-09-04.md) is now implemented and executed. All 15 measured checks passed and repeated identically after software-renderer setup was resolved; the expanded source/synthetic suite passes 116 tests. Visible-wall insertion produces actual contact, changing surface color preserves dynamics, and two known-angle camera projections agree within 0.2 pixels. This uses primitive boxes, not a walking Go2. Actual sensor calibration, commanded stopping and handoffs remain open. The frozen handoff runner is blocked by its clean-worktree requirement; the new development files have been preserved in place, not hidden or relocated to bypass it.

## Scientific aim

Implementation update, 5 September: the stratified physical collection was
stopped after a reproducible batched contact-force indexing defect was found.
All partial evidence is preserved and all ten staged development files were
restored with matching hashes. This is a measurement-integrity interruption,
not a valid negative physical or JEPA result. Recorded contact counts are not
verified collision incidence. A corrected development contact adapter and the
prepared causal sensor/directed-routing components are now integrated; the
explicit combined suite passes 194 tests. The bounded native contact-packet
assay now passes all six checks, with its failed reference-calculation attempt
preserved; see the [integrity report](docs/go2_contact_measurement_integrity_2026-09-05.md).
Next is fresh contact-attributed Go2 execution evidence. This supersedes the earlier
clean-worktree blocker and waiting-only next step, without changing the final
research objective or any frozen scientific source.

Enable a Unitree Go2 to explore previously unseen mazes, discover initially hidden beacons, and return to remembered places using RGB and deployment-valid sensor history. The robot should construct its memory online and execute high-level decisions through a conventional low-level locomotion controller.

Latest execution evidence, 5 September: a fresh eight-case Go2 study completed
with raw contact/trajectory auditing. Seven cases cross without contact, but
only the two straight cases meet all arrival checks. One narrow left turn has
a measured calf/corner collision; other failures concern arrival motion, some
only marginally over the chosen development limit. At the collision the camera
is already beyond the opening while the base remains behind it. This gives a
concrete reason to distinguish RGB progress from whole-body clearance, not proof
that JEPA is necessary. Next: matched pre-turn/arrival-control mechanisms and
actual two-edge continuation. See the [result and follow-up plan](docs/go2_contact_attributed_execution_development_v1_result_2026-09-05.md).
This was followed by a completed, audited 32-trial controller factorial with
actual continuation: combined control met the two-crossing/final-arrival endpoint
in 3/8 cases, versus 0/8 for each other arm, but none demonstrated sustained final
stopping. Native readback then confirmed joint gains of 100/10 instead of the
pinned gait configuration's 20/0.5. The fixed paired gain-restoration comparison
then completed and passed raw audit: task success improved from 0/8 to 8/8, with
8/8 sustained final arrivals and no contact in the corrected arm. This is a
local execution-interface result on four correlated development motifs, not
visual or JEPA navigation. See the
[factorial result and actuator finding](docs/go2_local_control_factorial_development_v1_result_2026-09-05.md).
The [gain result](docs/go2_actuator_gain_pair_development_v1_result_2026-09-05.md)
and [next implementation packages](docs/go2_rgb_multisensor_execution_next_steps_2026-09-05.md)
now direct continued work. The explicit focused suite passes 279 tests.
The next live acquisition package has since completed: 414 time-aligned RGB/body
history packets and 2,700 ideal simulated sensor samples from nine command probes,
with a strict policy-only loader and full raw-data audit. The expanded suite
passes 309 tests. See the [capture result](docs/go2_causal_rgb_body_capture_development_v1_result_2026-09-05.md).
This is not hardware-valid sensing or a learned policy. Next: multi-junction
development routes and matched action outcomes, using the new causal recorder.
Those multi-junction routes have now been executed and audited: all eight complete
their crossings without contact, four meet the full final-arrival endpoint,
and four miss only final heading. The 1,498 additional causal RGB packets and
325-test suite support proceeding to matched alternative-action data and model
comparisons. See the [route result](docs/go2_multijunction_route_development_v1_result_2026-09-05.md).
Oracle-route execution is not a learned RGB policy or a final maze benchmark.
Novel-maze and real-platform evidence remain missing.

The main scientific question is whether action-conditioned prediction of a compact sensor state improves navigation beyond a strong controller with the same observations, recent history, persistent memory, action proposals, and compute budget.

Evaluate two distinct JEPA contributions:

1. Predictive training improves the representation used by a navigation policy, even when the predictor is absent at deployment.
2. Running the predictor during navigation improves action selection beyond a matched policy that does not roll out latent futures.

Also establish whether persistent memory helps, whether body sensing improves execution, and whether their combination adds value. Learned temporal hierarchy is a later hypothesis, conditional on a functioning flat predictor plus memory.

## What the review established

The research has useful foundations: separated gait and navigation control, counterfactual action branching, paired training-seed comparisons, explicit negative controls, and honest reporting of failed experiments. Preserve these strengths.

The current evidence supports a narrower conclusion than complete maze navigation:

| Finding | Consequence for the plan |
|---|---|
| Rollout training improves feature prediction on the evaluated panel; navigation benefit is unproved | Measure executed decisions and task outcomes before expanding model size or horizon |
| The August 31 current-visual ranker selects the best route in 16/16 challenge states; the true-future ranker does so in 15/16 | Include a strong current-history/action baseline in every predictive comparison |
| September 1 localization succeeds with a simple MAP filter, but conditional execution reaches 0/16 goals even with oracle identity | Prioritize the local target-to-action interface |
| The August 26 waypoint lineage renders only a floor because of a scene-schema mismatch | Validate image semantics and affected provenance before interpreting geometry claims |
| The earlier counterfactual builder supplies a different manifest | Do not generalize the rendering defect to all earlier experiments |
| The proprioception experiment omits several unusable channels and averages away within-slot sample order | Test meaningful temporal body-state fusion; retain the existing null as specific to its experiment |
| The September 3 physical handoff panel has 180 qualified candidates out of 256, but only 52/64 strata filled | Distinguish panel construction from model evaluation and conditional feasibility from natural task success |
| Existing intervals primarily quantify training-seed variability on fixed environments | Add independent maze replication and environment-level uncertainty |

The [full review](docs/scientific_review_and_navigation_plan_2026-09-04.md) links each finding to its evidence. These are source-and-report findings, not results independently reproduced for this plan.

## Working design

A novel maze is partially observable. A single image cannot identify which of two identical corridors the robot occupies, and short visual history cannot remember branches explored minutes earlier. Maintain a short-term physical state and a persistent place memory with distinct responsibilities.

Use RGB, applied-command history, temporally encoded IMU/joint measurements, and deployable relative odometry as the principal proposed input. Include contact or torque information only where the actual logging and hardware provide meaningful measurements. Evaluate range/depth as a separate sensor condition after establishing its actual availability and timing. Simulator pose and future observations remain training/evaluation information, not prospective planner inputs.

| Component | Responsibility |
|---|---|
| Spatial visual encoder | Preserve obstacles, openings, landmarks, and viewpoint information |
| Causal multisensor state | Retain motion history, body state, sensor validity, timestamps, and uncertainty |
| Action-conditioned JEPA | Predict local visual and physical consequences of candidate command sequences |
| Task readouts | Estimate reachability, progress, arrival state, and execution reliability |
| Online place graph | Store observed traversals, provisional place associations, explored branches, and beacon sightings |
| Global planner | Select a remembered destination or unexplored branch |
| Local controller | Execute a target region with an approach direction, arrival condition, timeout, and recovery behavior |

Begin with a frozen spatial encoder and a modest temporal predictor as a reference, then evaluate joint JEPA adaptation. Keep local-prediction and place-retrieval projections separate where their invariance requirements differ. Feature similarity should not automatically be treated as travel cost.

The strongest candidate uses for prediction are motion mismatch, turn completion, arrival states that permit the next maneuver, recovery, and actions that reveal useful observations. Unknown geometry must remain uncertain; a plausible imagined corridor is not evidence of the actual layout.

## Immediate next work package: validate observations and physical handoffs

The next implementation proposal should be a bounded development package connecting the observation contract to the local execution contract. It should reuse existing source and focused checks rather than introduce another general auditing framework.

### 1. Establish the active scene, sensor, and action contracts

Produce a concise source-based table identifying:

- Scene schema supplied to each active renderer and its relationship to collision geometry.
- Camera frame, intrinsics/extrinsics, image preprocessing, and timing.
- Actual sensor channels, units, sample order, validity flags, and deployment availability.
- Requested commands, post-slew applied commands, control period, and prediction horizon in seconds.
- Local-target fields and their coordinate frames, including whether heading means bearing or desired arrival orientation.
- Which quantities are observable online and which are privileged labels.

Trace the confirmed floor-only lineage separately from the earlier counterfactual lineage. Record unresolved provenance explicitly. Do not inspect protected datasets to perform this source review.

Deliverable: a short active-interface specification and a prioritized list of source defects or unresolved contracts. This is the first recommended concrete task.

Completed at source level in the [active-interface specification](docs/go2_active_interface_specification_2026-09-04.md). It separates all three RGB lineages, requested versus applied commands, sensor versus control history, and target bearing versus arrival heading. It also identifies the physical camera routine's reflected FRU basis and the historical verifier's truncated-trace acceptance without asserting that either invalidated historical results.

### 2. Specify and implement small semantic fixtures

Use explicitly synthetic development fixtures to check the contracts:

- Insert/remove a visible obstacle at a fixed camera pose: rendered content and collision labels must change consistently.
- Change texture while holding geometry fixed: geometry labels must remain unchanged.
- Apply a known camera rotation and verify projection and frame conventions.
- Verify temporal windows cannot cross episode resets and candidate actions match executed post-slew commands.
- Check each used sensor channel for meaningful values and distinguish missing data from zeros.
- Feed deliberately correct, incorrect, and constant predictions to metrics so a non-discriminating metric fails immediately.

Source inspection can identify missing checks; empirical claims about the real data require a separately scoped development evaluation. Exit when the small evaluation distinguishes correct from deliberately incorrect behavior and the active interfaces are explicit.

Implemented for in-memory source interfaces. Analytic obstacle interventions, camera conversion, command reconstruction, sensor validity, causal windows and endpoint/metric negative controls are now covered. Actual Genesis pixel/physics consistency, texture rendering, calibrated projection and hardware sensor characterization remain open. The legacy occupied-IoU metric fails the semantic counterexample; it must not support occupancy or navigation claims. Prospective guards are not wired into frozen pipelines.

### 3. Resolve the physical handoff question

The [September 4 stratified generator successor](docs/lewm_go2_physical_handoff_stratified_generator_successor_v1_preregistration_2026-09-04.md) is already frozen. It seeks four teacher-qualified states in every one of 64 strata, with at most 64 candidates per stratum. It changes candidate allocation, not the learned model. Its completed result is not established by this plan; only its committed specification and predecessor summary were reviewed.

Keep any execution and interpretation within that experiment's existing scope. Do not add a new controller, target definition, sensor, threshold, or training step to the frozen attempt.

Use its eventual permitted result to select the next development question:

| Outcome | Next step |
|---|---|
| Panel remains inadequate | Analyze candidate rejection categories and physical feasibility; downstream model performance remains unmeasured |
| Panel is adequate, but candidate actions cannot realize the intended edge | Propose a separate local-target/action-bank/controller-interface experiment |
| Feasible candidates exist, but the current visual ranker selects poorly | Separate observation/domain mismatch, target encoding, and scoring errors using matched baselines |
| Local edge execution succeeds | Test consecutive edges and viable arrival states, then connect the controller to online memory |

Report all candidate attempts and rejection reasons. A panel selected for teacher feasibility establishes a conditional execution test; later navigation evaluation must use independently sampled starts and count physical failures.

### 4. Qualify a reusable local controller

In a prospective follow-on experiment, compare an oracle-information reference, a deployable geometric/range baseline where available, and the current visual policy under the same low-level controller and action bank.

Cover straight passages, left/right turns, offset openings, dead-end exit, stopping, and consecutive edge transitions. Define success using the correct crossing, arrival region, heading, speed, timeout, and ability to continue. Record contact, falls, overshoot, oscillation, interventions, and latency separately.

Choose numerical acceptance criteria from the declared platform and task requirements before observing comparative outcomes. A waypoint reached in an unusable arrival state is not a successful handoff. If even the oracle reference fails, fix the execution interface before attributing failure to visual representation or JEPA prediction.

## Subsequent research sequence

| Stage | Deliverable | Evidence required to proceed |
|---|---|---|
| Data coverage | Synchronized RGB/body-state histories and counterfactual branches using actual Go2 physics | Meaningfully different outcomes, valid sensors, adequate action coverage, and consistent training/evaluation histories |
| Predictive utility | Frozen-feature reference and joint-adaptation comparison with direct action scoring, persistence, kinematics, one-step prediction, and rollout | Improvement in a predeclared decision endpoint or a demonstrated predictive-pretraining benefit |
| Online memory | Empty-map exploration, observed edges, provisional loop closures, beacon discovery, and return | Better coverage and revisitation without an oracle graph or hidden goal locations |
| Controlled integration | Rollout-by-memory factorial and sensor ablations | Component contributions identified under matched inputs, control, and budgets |
| Generalization | Independent layouts, motif combinations, larger mazes, appearance and dynamics shifts | Paired maze-level outcomes with uncertainty across environments and training seeds |
| Physical transfer | Calibrated sensor/control integration and a bounded Go2 pilot | Measured transfer limitations, runtime performance, interventions, and task outcomes |
| Strategic hierarchy | Learned edge/subgoal prediction beyond the flat model plus graph | Incremental benefit under matched memory and compute, with executable subgoals |

Preparatory memory work and authorized hardware sensor characterization can overlap with local-controller development. Integrated navigation claims depend on demonstrated local execution.

Collect different actions from matched states and the same action across different scenes and physical conditions. Include stopping, recovery, and failures encountered by the local policy. Keep all scene descendants and sibling branches in the same data role. Randomize appearance independently of topology and behavior.

For proprioception, compare the existing mean pooling with a small ordered temporal encoder or recurrent physical state. Judge its contribution using motion and decision outcomes as well as visual prediction. Propagate predicted physical state through rollouts; never supply future measured sensor values to choose actions.

For memory, include simple image descriptors and frozen pretrained features under the same filter. Test physical revisits with position, orientation, lighting, and gait variation. Retain a new-place hypothesis and provisional associations; delay irreversible merges. Compare MAP and multi-hypothesis belief where ambiguity actually persists.

## Evaluation and stopping decisions

The central navigation comparison crosses online rollout and persistent memory:

| | No persistent graph | Online persistent graph |
|---|---|---|
| Strong history-conditioned policy without rollout | Local baseline | Memory contribution |
| JEPA rollout | Predictive contribution | Combined benefit and interaction |

Retain comparable recent history in the no-rollout baseline. Separately test predictive pretraining versus matched nonpredictive training. Hold sensors, action proposals, low-level control, safety layer, and task budgets constant where possible; disclose unavoidable compute differences.

Primary task outcomes are completion and beacon discovery within fixed budgets. Report coverage before first sighting, return success, wrong turns, repeated visits, execution failures, interventions, and sensor-to-command latency. Separate unknown-goal exploration from navigation to an already observed goal.

Use independent mazes and paired method comparisons. Account for clustering within mazes and for training-seed variability; do not count frames or branches as independent navigation trials. Set final sample sizes using development variability and a meaningful task effect. Keep targeted challenge sets separate from ordinary generated-maze performance.

If prediction improves features but not decisions, retain that as a bounded result. If a direct local policy plus graph matches rollout, investigate only a specific unmet decision requirement supported by evidence. If full belief matches MAP, retain the simpler reference. Defer scaling and hierarchy until there is a measured limitation they can address.

## Documentation follow-through

Update the scientific narrative to distinguish predictive fidelity, representation utility, online rollout, memory, local execution, and complete navigation. Preserve negative results and their exact scope.

Correct the progression document's planned use of legacy sealed V4: repository instructions permanently exclude it from final evaluation. Future final evaluation must follow the designated custody process; this plan grants no access to protected material.

The source-level interface specification, synthetic fixtures, primitive-scene
assay, native contact correction and three fresh Go2 development studies are
complete. The old frozen generator remains integrity-interrupted; its former
clean-worktree blocker is not a current next step. The immediate priority is the
pinned corrected actuator contract, broader local execution and deployment-valid
sensor integration. Do not begin another JEPA architecture
iteration before separating plant/adapter integrity, action feasibility,
target semantics and observation-domain shift.

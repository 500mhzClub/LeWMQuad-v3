# Immediate research priority: core JEPA world-model claims

The user's PhD centers on JEPA world models. Sensor-error robustness,
gyro correction and sim-to-real development are deferred. The broader
navigation goal remains incomplete; immediate experiments should establish
what JEPA learns and whether its predictions provide useful information.
Infrastructure work should be limited to what these experiments require.

## Literature-grounded direction (latest user steering)

The user requested a literature review and then reminded us of the prior frozen
V-JEPA 2.1 experiments. See `go2_jepa_literature_review_2026-09-17.md`, especially
its recovered-evidence correction. The initial proposal for new LeWM/DINO
baselines is superseded: the surviving frozen dense screen already favors
V-JEPA 2.1 (occupied IoU 0.5103 versus DINO 0.4709 and project ViT 0.3724).
Selected dense action-conditioned predictors and an eight-seed rollout study
also already exist. Recover their downstream evidence and applicability to the
current camera/command/navigation interface before new training. Do not repeat
these comparisons or restore the superseded BEV programme by default.

The recent CNN/GRU experiments are a separate smaller model lineage; neither
their negative results nor the earlier six-epoch predictor failures supersede
the later positive dense-predictor development results. A newly started,
untested projected-LeWM draft was removed after this correction; no training
or simulation was launched. Frozen V-JEPA 2.1 is the starting candidate for
reconnection to current development, with DINO retained as its comparator.

The proposed four-tick matched-branch collection is deferred pending this
baseline/coverage assessment. Matched reset branches are useful diagnostics,
not an established prerequisite for learning dynamics from exploratory data.
The review itself launched no experiment. The subsequent native transfer check
is now complete: `go2_frozen_vjepa_native_branches_2026-09-17.md`. Existing dense
one-step and rollout predictors lose to persistence at 500 ms (MSE ratios 1.606
and 1.684) despite 11/18 correct-action wins each. This does not erase their
historical positive results; it shows that direct transfer to current native
observations/commands is insufficient. Next adapt the dense predictor on existing
native training trajectories with the V-JEPA encoder frozen, retaining an
action-independent control and the unchanged checkpoints. New bulk collection,
another encoder screen, and a pooled LeWM reset remain deferred. The adapter
must use applied candidate commands reconstructed from causal state, not nominal
requested commands.

Native predictor adaptation and its branch evaluation are complete: see
`go2_frozen_vjepa_native_adaptation_2026-09-17.md` and its plan JSON. It uses 3,518
existing training sequences, a frozen V-JEPA encoder and two matched predictor
arms (future actions versus no future actions), fixed 24 epochs. Both completed
5,280 updates; training PID 13975/session 33534 exited 0. The adapted action model
has transfer feature MSE 0.243141 versus no-action 0.328250 and persistence
0.387830, retrieves 18/18 action branches, and reduces centered action-effect
error by 17.6% versus zero effect. These are two previously exposed geometries
excluded from adaptation training, not untouched final tests. See
`go2_frozen_vjepa_native_adaptation_branch_2026-09-17.md` for the results and the
preserved evaluation-only floating-point assertion failure/correction.

A subsequent CPU-only decision diagnostic on the unchanged native forecasts
finds a useful qualification: choosing actions to reach recorded visual goals
gives transfer endpoint error 0.535/0.628 mm for one-step/rollout, versus 1.100 mm
for persistence with uniform tie-breaking. Both retrieve 11/18 goal actions.
This is a tiny-pulse retrospective task, not navigation, but means factual
feature MSE alone must not gate decision testing. The same diagnostic on the
adapted model now retrieves all 18 transfer goals with zero selected-branch
endpoint regret; this is not zero motion-prediction error or navigation success.
See `go2_frozen_vjepa_native_visual_goal_2026-09-17.md`
for the result and the concrete visual-token/physical-motion interface gap.

The complementary recorded-mission evaluator is complete:
`go2_dense_native_recordings_2026-09-17.md`. Its 384 fixed windows cover all four
return-memory recordings, including both failures, with a common 500-ms visual
target. It compares adapted action/no-action, unchanged historical rollout and
persistence. All input timing and candidate-command alignment checks passed.
PID 25170/session 70961 exited 0. Action-model MSE is 0.486834 versus no-action
0.551460, persistence 0.588290 and unchanged rollout 0.734123. It beats no-action
in all four recordings and all six action categories, and persistence in three
recordings and all moving-action categories; hold is worse than persistence.
Retain this failure and distinguish these retrospective predictions from
prospective navigation. No dense caches were written to disk.

Next test a common training-only motion readout on observed versus predicted
dense features, keeping no-action, persistence and the existing command-history
baseline. The scientific question is whether the now-improved visual forecasts
contain useful physical motion information for the planner. Do not assume zero
visual-goal branch regret means zero motion-prediction error, silently substitute
500-ms output into the eight-horizon controller, or launch another maze sweep
before establishing a usable decision interface.

That common motion readout is complete: see
`go2_dense_visual_motion_readout_2026-09-17.md`. It uses the same 3,518 training
sequences, observed current/future dense features, no direct command/body inputs,
and a fixed 24-epoch fit. Encoder and predictors remain frozen. Transfer XY RMSE
is 8.189 mm with action predictions, 10.921 mm without actions, 10.064 mm with
actual future images, and 5.113 mm for existing command history. The physical
point-goal decision diagnostic also favors command history. The oracle readout
failure prevents attributing this solely to the predictor. Physical branch
outcomes repeat exactly across the four geometries: this panel cannot test a
geometry-dependent dynamics advantage. Preserve the positive dense visual
prediction evidence and test direct visual-goal costs beyond exact 500-ms goals,
before spending more training on motion decoders. The subsequent local
prospective pilots below now supply dense-predictor control evidence; full
independent-maze navigation remains unproven.

The direct later-visual-goal diagnostic is now complete:
`go2_dense_distant_visual_goals_2026-09-17.md`. The unchanged 500-ms predictor
agrees with an actual-future-image oracle on 17/18 transfer action choices for
both 800-ms and 1,200-ms goal images. Primary 800-ms physical XY regret is
0.434 mm versus 0.655 mm for blind uniform choices; yaw regret is 0.115 versus
1.542 degrees. These are still dependent one-pulse retrospective tasks. The
next step is a bounded prospective native image-goal control pilot using this
cost and the actual 500-ms interface, with new observations after each action.
Do not add another motion decoder or treat this diagnostic as navigation.

The prospective dense image-goal pilot is complete:
`go2_dense_visual_goal_pilot_2026-09-17.md`. The unchanged dense predictor directly
selected all 20 actual 500-ms commands from newly acquired images in both
transfer geometries. It approached to 1.85/0.39 cm and avoided contact, but both
runs continued past the goal and ended over a metre away. One briefly met the
predeclared three-frame visit criterion; neither stayed at the goal. Both blind
controls hit the obstruction. Preserve the original command-precision interface
failures as well as all four corrected live outcomes. Physics paused during
inference. Next compare predicted versus actual candidate goal costs from a
matched near-goal state, separating predictor ranking from visual-cost failure
before another controller change or navigation sweep.

That matched near-goal diagnostic is complete:
`go2_dense_goal_overshoot_2026-09-17.md`. Twelve exact-prefix native branches
show that the raw dense goal cost chooses forward even with actual future
images in both tested states. Braking is physically better; in case 3, hold
alone satisfies both endpoint tolerances. Predictor error also affects cost
margins, but perfect forecasts would not fix these choices. Next fit a goal
distance on training-layout visual pairs and physical XY/heading relations,
keeping encoder and predictors frozen. Test actual-future versus predicted-
future costs before another live comparison. This new cost supervision must
not be described as an isolated benefit of JEPA training.

The goal-metric fit is complete: `go2_dense_goal_metric_2026-09-17.md`.
It uses 24,294 training-only visual pairs from 138 recordings, fixed 24 epochs,
and a shared 426,016-parameter embedding with squared-distance supervision.
Encoder/predictor weights stayed fixed. PID 36304/session 29941 exited 0 after
4,560 updates in 863.2 seconds. Final training log-distance MSE is 0.006400.
Near-goal evaluation completed (session 29163, exit 0). The learned metric
chooses the physically best action with actual future images at both states,
and with predicted futures at one of two. The four-case prospective native
cost-only comparison is complete: mean action-trial final XY error fell from
118.71 cm to 3.16 cm; both now visit the goal region but neither finishes
within tolerance. Both later turn while near the goal. The two-state hold-versus-
recorded-turn diagnosis is complete: actual future image costs and physical
outcomes both prefer hold, while predicted features reverse that ranking.
The matched predictor continuation is complete (session 70471, exit 0):
four arms crossed dense-only versus goal-embedding auxiliary loss with action
availability, each receiving eight extra epochs. Encoder and goal metric stayed
frozen. The auxiliary action arm has lower training goal-embedding error but
higher dense error. Component evaluation and all four prospective task cells are complete. Neither
continuation improves on the previous model: final XY means 8.112 cm dense
and 3.888 cm auxiliary versus 3.160 cm parent, with 0/2 final arrivals each.
The auxiliary objective is not promoted. See
`go2_dense_task_goal_pilot_2026-09-17.md`; see `go2_dense_task_predictor_2026-09-17.md`. See
`go2_dense_metric_goal_pilot_2026-09-17.md` and
`go2_dense_metric_late_turn_2026-09-17.md`.

## Current evidence

- Matched frozen-representation latent predictor refits improved JEPA's
  800-ms action-branch retrieval from 6/18 to 11/18 on the small exposed
  transfer set. Raw latent MSE across different representations is not a
  valid ranking. Strong training-only conditional means remain competitive.
- After training-only motion-readout adaptation, JEPA branch-transfer
  position error improved from 6.561 to 5.462 mm; supervised features still
  performed better at 4.819 mm.
- On every matched executed window from the latest four navigation
  recordings, JEPA position error improved from 9.201 to 8.631 mm. Command
  history achieved 7.950 mm and the original untrained representation with
  fitted readout achieved 7.556 mm. No JEPA superiority is established.
- Existing navigation results demonstrate working components and a benefit
  from explicit routing memory. They do not isolate a JEPA training benefit.
- The completed factorial target-modality assay finds that freezing future
  RGB changes the JEPA target by only 0.0254% of its full temporal-change energy
  at 800 ms on the exposed action-branch population. This is a non-additive
  encoder sensitivity measure. Future body/control dominate the mixed target;
  the next training comparison should test a visual-only future target while
  retaining deployment-valid multimodal context and known actions.
- That visual-only target comparison is now complete: its predictor loses to
  visual persistence, retrieves 6/18 transfer action branches, and its fitted
  motion readout scores 9.804 mm across the same 2,404 navigation windows versus
  original mixed-target JEPA's 9.201 mm. Removing the shortcut alone is not
  sufficient. Prioritize learning action-dependent changes from current visual
  state, retaining persistence and matched controls, before another navigation
  sweep. Details: `go2_visual_target_jepa_2026-09-17.md`.
- The frozen visual-state anchor experiment is also complete. Action conditioning
  reduces centered action-effect error on transfer branches by 16.2% and pooled
  recorded visual error by 1.7% versus a matched no-future-action predictor.
  However, persistence remains better overall: 700-ms visual MSE 0.027610 versus
  0.030777 for the action model. These are common-space latent errors, not motion
  errors or a JEPA representation-training advantage. Next isolate temporal
  visual history; details: `go2_anchored_visual_dynamics_2026-09-17.md`.
- Visual-history comparison is complete: pooled recorded visual error improves
  to 0.026704 with actions and 0.026547 without them, versus persistence 0.027610.
  The action branch retrieval increases to 8/18, but branch forecast error
  worsens. A coverage count finds only 90/7,200 training draws at the verified
  identical-history departures, all with one-tick pulses. Sustained trajectories
  exist. The initial proposal to collect matched four-tick alternatives is now
  deferred by the literature-grounded direction above; this count alone does
  not prove inadequate training coverage. See
  `go2_visual_history_dynamics_2026-09-17.md`.

Sources: `go2_jepa_latent_branch_science_2026-09-17.md`,
`go2_frozen_representation_dynamics_2026-09-17.md`,
`go2_refitted_dynamics_readout_2026-09-17.md`, and
`go2_return_routing_memory_2026-09-17.md` in this directory. The target-modality
diagnosis is in `go2_jepa_target_modalities_2026-09-17.md`.

## Questions that should drive the next experiments

1. Does the learned representation encode future visual/environment state
   beyond predictable body state and command history? Current mixed targets
   include future RGB, body and control; latent predictability alone cannot
   establish visual world modeling. Prioritize matched modality controls and
   an environment-dependent prediction target with stronger simple baselines.
2. Does action conditioning predict differences between possible outcomes?
   Use identical causal histories with distinct executed action branches,
   evaluate correct versus shuffled actions and action-independent forecasts,
   and avoid treating overlapping windows as independent evidence.
3. Does JEPA training improve transfer or data efficiency relative to matched
   supervised and untrained representations? Compare common downstream
   targets, capacity, training data and readout budgets; retain failures.
   Use additional independent layouts/seeds when a component result warrants
   confirmation. Current exposed development results are exploratory.
4. Do better predictions improve decisions? Once there is a useful predictor,
   run matched prospective planning comparisons. Replay forecast accuracy and
   overall navigation success alone do not establish this contribution.

No new experiment is launched by this priority note. The completed refit and
recorded-navigation evaluation are retained, without automatic promotion.
More candidate futures and non-maze environment tests remain deferred pending
the core representation/action-prediction evidence.

The direct visual-goal baseline component fit is complete (session 46814,
exit 0; 24 epochs, 4,560 updates, 862.36 seconds). Its fixed evaluation also
completed (session 9093, exit 0):
`go2_direct_visual_goal_readout_2026-09-17.md`. It uses the same training-only
100-3,000-ms pairs and update budget as the learned cost, adding both image
pair directions with exact planar inverse labels. This addresses the old
500-ms readout's measured range/direction mismatch. The preceding predictor
comparison and prospective trials completed without a final-arrival success.
The new readout reduced current-to-goal position/heading error from
7.66 cm / 9.04 degrees to 2.00 cm / 1.18 degrees, but missed seven of nine
sampled within-goal states. No strong reactive-baseline claim follows.

The subsequent visual-arrival pilot is complete: native sessions 38708/27621
and reader 6325 exited 0. Keeping the parent predictor and scalar goal cost,
the direct readout now latches hold on observed-image arrival. Both exposed
local tasks ended inside tolerance (2.62 cm / 3.66 degrees), without contact,
and stayed inside for all 8.5 seconds after detection. Their pre-latch planner
actions, costs and RGB prefixes match the parent. See
`go2_dense_visual_arrival_pilot_2026-09-17.md`. This is a local recognition
improvement, not new predictor evidence or an independent maze result.
Next establish the matched direct visual-feedback comparator and test frozen
controllers on fresh layouts/goals. Preserve these successes and preceding
failures; the complete unseen-maze navigation goal remains active.

The fixed direct visual-feedback comparison has now completed (native sessions
20013/69724 and reader 13653, all exit 0): 1/2 feedback final arrivals versus
the retained world-model 2/2, all without contact. Cluster 02 exactly reproduces
the successful world-model physical trajectory without prediction. Cluster 03
takes a different approach, then falsely latches arrival at 5.27 cm actual error
versus 1.63 cm estimated; final error is 6.04 cm. Its readout later recognizes
it is outside tolerance, but the permanent latch prevents recovery. This is
evidence against general reliability of the shared arrival rule, not grounds
to tune on these exposed tasks or claim general planning superiority. See
`go2_direct_visual_feedback_pilot_2026-09-17.md`. Preserve the full failure.
Fresh layouts/goals and separate reporting of approach versus arrival detection
are next; prospective RGB-only recording is needed to avoid writing unused
depth for every new scientific comparison.

The four fresh local layout/goal comparisons are now complete:
`go2_fresh_visual_goal_comparison_2026-09-17.md`. Both frozen controllers
achieve **0/4 final arrivals**. The planner has one transient visit and zero
contacts; feedback has zero visits, one contact and two false arrival latches.
One planner failure is a near miss (3.040 cm final versus 3 cm tolerance), but
both right-opening tasks show a larger failure: the planner initially turns
left, away from the supplied goals, and ends with 95–102-degree heading error.
The initial signed readout identifies the rightward goal direction in task 01,
so arrival recognition alone does not explain this planning failure.

Next run a small matched-action diagnosis at that first task-01 decision:
compare forecast rankings with actual successor-image goal costs and physical
progress, retaining the predictor and cost head unchanged. This will separate
goal-scoring transfer from action-conditioned forecast transfer before choosing
another scientific intervention. Do not retune the arrival threshold or claim
generalisation from the earlier exposed-task 2/2. Eight prospective outcomes
and every failure are retained. RGB-only recording now saves all control and
physical evidence without unused depth; the eleven-frame capture check exactly
matched the old RGB path. All native and reader processes exited successfully.

The matched-action diagnosis is complete:
`go2_fresh_goal_direction_2026-09-17.md`. Five new native branches plus the
recorded left-turn successor reproduce the initial state and applied commands.
The learned goal metric selects left-turn even on actual future images,
whereas physical cost prefers right-arc. Mean future-feature error still beats
persistence by 44.6%, with 5/6 correct-action retrieval. Thus this specific
wrong-direction choice is explained by goal scoring despite useful prediction;
the predictor is not established as error-free.

Using the existing signed goal readout as the planner cost corrects the actual-
image ranking and changes predicted choice to forward, but the completed
four-task cost-only pilot still yields 0/4 final arrivals and one contact.
See `go2_signed_pose_goal_pilot_2026-09-17.md`. No checkpoint or arrival rule
changed. Retain this failure, and do not promote either cost as reliable.

A training-only diagnostic now identifies a specific supervision gap:
`go2_goal_metric_turn_separation_2026-09-17.md`. All 24,294 scalar-goal pairs
are within recordings. The metric fits distances from the common start to
left/right turns accurately, but assigns opposing-turn endpoints only 3–4%
of their physical squared cost on the two right-opening training layouts.
Those cross-trajectory pairs were never supervised despite both images being
training inputs. Next test a matched-budget mix of within/cross-trajectory
pairs using only existing training RGB and native labels. Freeze the encoder
and predictor and preserve the old fit. This is a goal-grounding intervention,
not proof of JEPA training superiority; full closed-loop unseen-maze evidence,
memory/backtracking integration and deployment validation remain outstanding.

The matched-budget cross-trajectory fit and four online trials are complete:
`go2_cross_trajectory_goal_pilot_2026-09-17.md`. Opposing-turn distances on
right-opening training layouts improve from 3–4% to 97–100% of physical cost,
and actual-image goal ranking at the exposed right start is corrected.
Nevertheless, control yields 0/4 final arrivals, one transient visit, one
contact and one false arrival latch. The predictor and arrival rule were
unchanged. This is useful diagnosis of goal supervision, not a JEPA advantage
or a successful navigation system; do not continue fitting goal heads by
default.

Inspection of the actual predictor training samples rules out a complete
absence of quiet-to-motion transitions, but reveals severe imbalance:
524 quiet-to-hold windows, versus only two exact quiet-to-500-ms examples
per moving action, both on left-opening geometries. See
`go2_dense_start_action_coverage_2026-09-17.json`. A balanced training-layout
start-action collection and matched predictor comparison would test this
hypothesis; causal evidence and prospective navigation are still required.
Do not infer that better local distance calibration supplies collision
avoidance, long-horizon planning or maze exploration.

The balanced start collection is now complete: 48/48 training-only cases,
zero contacts, eight environment groups with matched starting RGB. A fixed
four-arm continuation has been prepared and launched: original/mixed data
crossed with action-conditioned/action-blind predictors, identical budgets
and paired initial model/optimizer states. See
`go2_balanced_start_predictor_2026-09-17.md`. No gain is established yet;
the next evidence must be prediction/action-ranking comparisons followed by
prospective online control, not training loss alone.

That continuation has now completed (session 21525, exit 0, 46.46 minutes,
1,760 updates per arm). The corrected fixed evaluation supports the coverage
hypothesis at the exposed right start: 21.3% lower dense prediction MSE than
matched original-data continuation, and right-arc selection instead of forward,
with zero local physical regret. The broader 18-branch transfer panel is mixed:
5.2% worse dense MSE but 18.4% better goal-embedding MSE. Both action arms
retain 18/18 action retrieval. Neither metric alone establishes navigation.
The 12 preselected online trials are now launched, using both action predictors
and the shared action-blind policy on four exposed tasks. No arrival rule,
goal metric or criterion has changed. See the balanced-start report for the
preserved offline numerical tie correction; no training was repeated.

All 12 control trials are now complete: supplemented action predictor **2/4
final arrivals, zero contacts**, original-data continuation **0/4, zero
contacts**, shared blind policy **0/4, three contacts**. This is a matched
training-budget local benefit, with improvements on one right and one left
task. Both action predictors retain the same two failures involving the
shared arrival latch. See `go2_balanced_start_goal_pilot_2026-09-17.md` for
the full table, interpretation and retained references. No full-maze or JEPA
encoder-objective claim follows; independent prospective validation and the
dense model's route/collision interface remain necessary.

The next integration measurement is complete:
`go2_balanced_endpoint_navigation_2026-09-18.md`. A causal 500-ms endpoint
adapter reproduces parent predictions, but the same frozen motion head gives
the supplemented model 9.067-mm transfer XY RMSE versus command history's
5.113 mm, and worse point-goal action regret. This does not invalidate the
visual-goal benefit; it rejects treating that benefit as an automatically
better physical-motion interface. The existing full-maze runtime consumes
eight 100-ms outputs and a 700-ms dispatch/commit window. Do not silently
interpolate or truncate the dense model into that interface. No new full-maze
or hardware execution has occurred.

September 18: the training-data audit supports native visual targets at all
eight 100--800-ms offsets. The 48 balanced training starts are being extended
to 800 ms with two independent simulation workers. A small horizon-conditioning
extension preserves the retained 500-ms predictor output exactly at
initialization and masks commands beyond each target. Its focused check on
actual training RGB passed without optimizer updates. The matched training
design is in `go2_horizon_dense_predictor_2026-09-18.md`; this addresses temporal
coverage, while the weak physical decoder and collision/timing interface
remain unresolved. Keep those limitations distinct from visual forecast gains.

The native-horizon fit and evaluation have now completed. At 700 ms, visual
prediction MSE is 0.2343 versus action-blind 0.3121 and persistence 0.4018;
action retrieval is 18/18 from 500 through 800 ms. The retained 500-ms dense
accuracy is preserved. Scene-by-action interaction error is 12.0% below the
zero-interaction reference at 700 ms in aggregate, but not improved for every
prefix. Physical motion decoding remains weak and the measured float32
six-action/eight-horizon inference takes 2.55 s, exceeding the old 300-ms
deadline. See `go2_horizon_dense_predictor_evaluation_2026-09-18.md`.
Four prospective maze layouts are fixed for the next comparison; no native
execution has occurred on them. Move toward explicit closed-loop integration
and honest timing treatment, retaining these negative readout results rather
than treating visual prediction scores as navigation success.

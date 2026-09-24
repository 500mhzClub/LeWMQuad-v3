# Next steps after the first actual RGB/body JEPA comparison

The final aim remains unseen-maze exploration, beacon discovery and return using
RGB and deployment-valid sensor history, online memory and an ordinary Go2 gait.
The [new result](go2_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
supports a visual supervised-rollout baseline, not a JEPA utility claim.
Continue toward the system and an honest answer about JEPA; do not require a
positive JEPA result to count as scientific success.

Current update: step1's adapter and72-trial fresh online pilot completed and
passed full audit. See the [result](go2_online_choice_maze_pilot_development_v1_result_2026-09-05.md).
Do not rerun that package. The missing sensor-based look/turn-back primitive has
now passed its [fixed paired arena assay](go2_gyro_turn_assay_development_v1_result_2026-09-05.md):
gyro9/9 versus timed6/9, with narrow-clearance/hardware qualification still open.
The [causal subtrajectory derivation](go2_causal_subtrajectory_development_v1_result_2026-09-05.md)
also has914 raw-audited and tensor-loader checked windows with unchanged layout roles and remaining-plan
masks; their new fitted-model comparison is now complete and fully audited. The
[temporal model interface](go2_temporal_model_interface_development_v1_result_2026-09-05.md)
has since passed actual-input and synthetic checks, and the
[next comparison design](go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md)
has completed all nine models with bound schedules, sources and inputs.
The [result](go2_temporal_rgb_body_learning_comparison_development_v1_result_2026-09-05.md)
does not establish JEPA utility. The [next execution plan](go2_temporal_post_comparison_execution_next_steps_2026-09-05.md)
now supersedes the chronological proposals below: test successive physical
decisions and action changes after movement before another fitting cycle. The
[live history buffer](go2_online_rgb_history_development_v1_result_2026-09-05.md)
also passed recorded-stream replay in preparation for later replanning.
Continue the temporal-model,
predictive-diagnosis and memory steps below;
the chronological step1 proposal is preserved as the completed package's context.

## 1. Make the useful reference an actual online local decision interface

Implement a strict adapter accepting only a causal RGB/body/control packet and
an explicit body-frame local intent. Reconstruct all five prospective action
plans from the preceding applied command; batch them through the bound model.
Return candidate costs, selected command tape, model identity and measured
inference timing. No simulator pose, graph, future image or labels in the API.
Test rejection of privileged fields, stale packets, bad checkpoint bindings,
plan timing, deterministic ties and reset behavior.

Use all three existing seeds as a fixed prediction ensemble rather than choosing
the best validation seed. Keep supervised-rollout and JEPA-rollout ensembles
architecturally matched. Their already fitted sources/checkpoints are immutable;
the adapter is a new development component. Preserve the all-stop comparator.

Then specify a fresh bounded **online conditional-choice pilot**, not a
maze-navigation claim: eight new independently seeded procedural layouts,
three supplied local intents, and three methods (supervised rollout, JEPA rollout,
all stop), giving 72 physical trials. Do not reuse or recollect completed corpus
branches. Freeze new seeds, topology-disjointness checks, conditions, action bank,
intent/cost, 4-s action plus release, native contact stop and raw reduction before
execution. Do not alter the bound 24-layout generator; use a new source module.

The teacher may establish the starting junction, with privileged initialization
explicitly declared; after that, model selection uses the actual current packet.
Verify matched physical/body prefixes across methods, preserve bounded rendering
variation and record real input bytes. Compare paired realized cost, contact and
stop/progress rates by layout, including every failure. Without executing all
alternative actions in those new states, call it realized-cost comparison—not
oracle action regret. Simulator contact/stability emergency stops remain a
privileged experimental safeguard, not proven deployment-valid sensing.

This is the next concrete implementation package. Its value is testing the
real sensor-to-command path and fresh local generalization, not repeating model
fitting or polishing offline metrics until favorable.

## 2. Diagnose the predictive objective without a coefficient search

Preserve the present negative result. Compare scene discrimination at a fixed
action/horizon, separately from variation across commands and time. Use a fixed
simple current-RGB nearest-neighbour or frozen-feature readout as a perception
diagnostic, with identical training layouts and no role leakage. Report motion,
contact and decisions; do not rely on embedding similarity alone.

If visual information is usable but joint prediction suppresses it, make one
mechanistically motivated comparison: retain a stable current visual path and
predict action-conditioned residual physical/visual changes separately. Keep the
supervised-rollout baseline, matched direct readouts and runtime rollouts. An
action-only dynamics residual is more informative than relearning the easy mean
motion. Conditional scene variation matters more than a global noncollapse score.

Only expand the training question after specifying the change and its failure
criterion. Do not automatically extend training, replace seeds or sweep latent
loss weights on these same eight validation layouts.

## 3. Broaden temporal/state coverage before receding-horizon claims

Current outcome readouts were supervised at only the initial branch context.
Future observations participated in regularization/JEPA targets, not equivalent
supervised decisions from every future state. Applying this readout repeatedly
after movement would therefore introduce a new state-distribution assumption.

Create explicitly causal RGB-history/subtrajectory training windows from the
recorded sequences, with past images/body samples, remaining known action tape,
relative labels and correct stop censoring. Keep every window from a layout in
its original role. Add turning-in-place, stopping/reorientation, braking, slip,
narrow clearance and recovery only in separately specified physical collections.
Use real measured candidate execution; do not impute unexecuted branches.

Data preparation update: the fixed development derivation now supplies four
past frames and actual body/control histories at eight specified branch-relative
offsets. Windows after time0 use their own branch state, not a shared canonical
sibling. All targets stop at the original known-plan endpoint; contact censors
motion, and unknown outcomes stay unknown. There are still only16 training and8
validation layouts. Sampling and uncertainty must account for repeated windows
within a layout, not treat914 windows as914 independent trials.

Before fitting, implement/test an explicit history-and-plan-mask model interface.
Do not pass padded unknown commands into the old unmasked model as executed
stops. Freeze the matched direct/supervised-rollout/JEPA conditions, equal
observation exposure, layout-balanced training schedule and final-checkpoint
rule. Keep temporal coverage as an explicit change; do not simultaneously sweep
latent weights, change action costs and select favorable seeds. Report fixed
action-only and visual diagnostics, later-state prediction and initial-context
executed action choices. Later-state action ranking still needs fresh executed
alternatives or actual closed-loop evaluation; the derived windows cannot supply
that missing evidence. Choose and preregister a new physical replanning protocol
before launching it, retaining contact/stopping failures and the all-stop control.

The next evaluation after the single-decision pilot is successive **sensor-only
replanning**, first with supplied local intents, then intents from online memory.
Measure body clearance, heading/speed on arrival and actual continuation. Passing
one local action or selecting stop is not sufficient.

## 4. Complete memory, exploration, generalization and transfer

Integrate place association and relative odometry with the existing directed
observed-edge graph. Keep provisional associations, uncertainty and wrong-merge
recovery; never infer reverse traversability from a forward traversal alone.
Represent observed beacons and unexplored exits, then compare persistent memory
against the same recent-history controller without persistent memory.

Supply global intents from observed frontiers/remembered targets, not an oracle
maze map or unseen goal image. Evaluate unknown-beacon discovery separately from
return to an already observed location. Fix task budgets, safety/intervention
criteria, latency and independent-layout sample sizes before a final evaluation.
Legacy V4 and all sealed material remain inaccessible and ineligible.

Finally establish actual camera/IMU/joint/command timestamp and calibration
contracts, delay/dropout/contact observability and relative-odometry drift on the
physical platform. A bounded real Go2 pilot needs actual hardware access and
appropriate human safety supervision. Until that is available, software/simulation
work can continue, but physical transfer cannot be declared complete.

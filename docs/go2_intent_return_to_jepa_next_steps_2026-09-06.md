# Next: robust execution and useful learned prediction in connected mazes

Preserve the full original objective: RGB plus deployment-valid sensors, JEPA
prediction, useful online memory, independent novel mazes/training seeds,
matched training/rollout/memory comparisons and bounded hardware when available.
The newest paired assay remains0/3 full returns with no home hold; an earlier
batch had one passing final-home hold but still0/3 complete returns. More unit
tests or repeated room versions do not replace the objective.

## Immediate engineering experiments, driven by the audited failures

Update: the consistent4cm implementation passed focused and recorded-state
checks, then failed the paired physical test. The balanced-feature replay also
loses coverage overall despite surviving the old right failure. Neither is a
demonstrated end-to-end improvement; preserve their negative results.
See [the completed comparison and reachability evidence](go2_balanced_features_and_inner_arrival_progress_2026-09-06.md).
The [paired inner-arrival audit confirms0/3 returns](go2_inner_arrival_collection_result_2026-09-06.md),
despite six passing early local holds. Both nominal routes lose tracking during
the half-turn, and low friction still fails execution. No further same-room
tolerance variation is justified by these results. A response diagnostic shows
missing short-forward coverage and substantial low-friction lateral/dynamic
error, motivating a prospectively balanced action/state collection.

1. Separate internal control targets from external success criteria. Design a
   distinct controller/planner using an inner arrival region (prospectively,
   e.g.4 cm rather than6 cm) while retaining the external6 cm native score.
   Apply the inner region consistently to planning, settling and clipped-leg
   completion, not only the final decision check: a planner that already stops
   at6 cm can otherwise return an empty path outside the controller's region.
   Derive feasible cases with translation-coupled turns and finite action
   resolution before launching. A2 cm reserve is an engineering hypothesis,
   not a certified pose bound. Estimate/calibrate error on development scenes
   and test fresh scenes without retuning their acceptance. Keep the failed
   6.1557 cm intermediate hold as failed.
2. Diagnose visual support loss before another physical batch. At right1809,
   both consensus fraction and grid support fail for the active reference;
   all eight candidates fail. Measure spatial feature/correspondence coverage
   through descriptor matching, LK refinement, depth lifting and rigid pruning.
   A distinct spatially balanced detector or a complementary RGB-D alignment
   method is worth testing under unchanged geometric/gyro/increment gates.
   Test entire recorded streams, not a selected failure pair only; preserve
   current/past-only causality, conflict rejection and uninterrupted pose.
   Do not simply relax the six-cell or0.6-fraction thresholds. A new recovery
   on old recordings still requires fresh closed-loop evidence.
3. Address the low-friction model mismatch explicitly. The unchanged nominal
   table steers the first leg far off course despite available tracking. Compare
   a state-conditioned dynamics baseline, carefully bounded online response
   correction and the learned RGB/body predictor on actual pulse responses.
   Collect support/state/action coverage prospectively; do not feed simulator
   friction labels to the controller. Characterize backward and arc commands
   before adding them to the action vocabulary. Enlarging search budgets cannot
   by itself repair wrong dynamics or absent controllable action coverage.

Freeze distinct successor sources, seeds and outputs. Engineering fixes can be
combined for an end-to-end development test, but attribution needs prospectively
paired ablations. Preserve budgets, failures, all local/winding/home native
holds and complete raw reconstruction. Do not repeat the same failed setup with
new names or quietly exclude the low-friction challenge.

## Learning must become an actual experiment

Update: [dataset integration now materializes all185 recorded windows](go2_pulse_dataset_integration_progress_2026-09-06.md)
with917 valid targets and matched schedules. The
[training pilot now completes nine fitted models and108 updates](go2_pulse_training_pilot_result_2026-09-06.md),
with checkpoint persistence verified, but every learned head has worse planar
error than zero motion on this short-budget development corpus. All three
trajectories share one layout and cannot become independent layout roles.

The pulse-timed adapter now has real observation/prefix indices, native targets,
binary-contact decoding, matched objectives and a working dataset/training
runner. The [matched action/time control and gradient diagnostic are now complete](go2_pulse_baseline_and_loss_scale_result_2026-09-06.md):
the empirical predictor scores0.859cm versus zero2.233cm and JEPA4.103–4.932cm,
but fails the low-friction comparison (4.073cm versus zero2.994cm). All30 cells
have training support; no scoring targets were dropped. Initial JEPA gradients
on the shared first batch are dominated by the latent objective, while combined
position loss contributes less than0.4% of total. This motivates one prospectively
fixed task-scale/budget comparison, not a claim that scaling will fix learning.
Keep architecture, paired seeds and sensor/data exposure unchanged, compare
original and scaled objectives at matching budgets, and report the empirical
baseline and condition-specific errors. Then prioritize independent collection
regardless of the same-room outcome. The old185
windows yield917 valid targets and zero positive contacts. These correlated
open-room samples are not independent maze layouts or balanced hazard data.

That [fixed scale/budget comparison is now complete](go2_pulse_position_scale_budget_result_2026-09-06.md):
18freshfits/2160updates,12and120snapshots,withall9raw12predecessoridentities reproduced.
At120updates scaled JEPA recursive position error is1.07–1.95cm versus raw2.14–3.34cm,
but all trained heads still lose to the0.859cm empirical baseline. Scaled standalone
direct prediction worsens in all three seeds; JEPA beats scaled supervised rollout
in only two. Yaw and cumulative-contact semantics remain weak. Do not extend this
run, search more scales or promote a same-room winner. The next concrete task is
the [independent pulse-context collection constructor and pairing pilot](go2_independent_pulse_context_collection_design_2026-09-06.md),
with tracker-independent bounded simulation excitation, entire-layout role splits,
verified common action histories and support observability rather than friction oracles.

Use them only for explicitly development-only plumbing/dynamics checks. Collect
new data with independent layouts, appearances, start/body histories and all
supported actions; freeze train/selection/evaluation roles before fitting.
Include observable obstacle/branch choices and properly censored failures.
Keep physical contact labels separate from missing observations and compute
contact-class coverage before any risk-learning claim. Use multiple training
seeds and identical sample schedules across direct, supervised-rollout and
JEPA arms; report action-only/kinematic and geometric baselines too.

First verify prediction of actual useful state changes and decisions, including
RGB/history/action ablations and transfer under changed support. Then integrate
the predictor into online candidate selection with the same sensor/local-control
substrate and budgets across arms. Predictive-training benefit, online-rollout
benefit and online-memory benefit are three distinct questions. Do not claim
JEPA utility from lower latent loss, an untrained objective test or a model with
extra sensor/data exposure unavailable to the baselines.

## Connected-maze and deployment requirements

After dependable continuous execution, replace scripted room stages with actual
observed branch/marker choices, persistent visits/attempts and physically
executed backtracking. Validate home independently; preserve UNKNOWN association
in aliased corridors. Compare memory-on/off in completed connected-maze tasks
on independent layouts/seeds, not waypoint bookkeeping tests.

The measured143–146 ms median observation/control compute exceeds100 ms while
physics is paused. Profile acquisition/rendering/registration/planning separately
and test wall-clock deadlines, latency/dropouts and calibrated/noisy sensors.
Restore robot self-occlusion and near-field/swept-body evidence; uncertainty
must not be inferred from point count alone. Obtain bounded real-platform
evidence when hardware access permits. No current result supplies that evidence.

# Follow-through from successive control to the actual maze goal

This is a chronological working implementation plan, not authority to alter the frozen
[144-trial protocol](go2_successive_choice_maze_development_v1_2026-09-05.md).
It was first written during collection; do not treat interim outcomes as the final
method comparison or choose a favorable checkpoint from them.

Completion update: the [full144 result](go2_successive_choice_maze_development_v1_result_2026-09-05.md)
and [action-coverage diagnostic](go2_successive_action_coverage_development_v1_result_2026-09-05.md)
are now complete. JEPA's half-second latent head has a bounded contact advantage
over the matched supervised head, with4/24 contacts still present. The original
full checker FAIL is preserved beside its tested boundary-corrected audit PASS.
Steps1–2 below retain their planning history; the immediate action is the
moving-prefix collector/protocol and observed-memory/translation integration,
not another audit or rerun of the completed panel.

Moving-prefix update: the [fixed384-trial collection, full audit and600-cell
tensor qualification are complete](go2_moving_prefix_counterfactual_development_v1_result_2026-09-05.md).
Do not rerun them. The selected600-cell view omits698 other old temporal windows;
the new augmented loader therefore retains all914 old windows plus384 switches,
and a paired sampler fixes current-context exposure while expanding future
action support. The [18-model matched comparison and full audit](go2_context_matched_coverage_learning_development_v1_result_2026-09-05.md)
are complete. Broader support helps three-second predictions/ranking, but the
half-second switch-contact endpoint contains zero hazards and no general JEPA
advantage is established. Next advance actual RGB observation/arrival integration
and fixed physical task comparisons with progress/stall outcomes; not polling,
another audit, or a best-model promotion from offline scores.

Step3 now has an [observed exploration/local-controller bridge and measured
translation baseline](go2_observed_exploration_and_translation_development_2026-09-05.md).
The symbolic component handles ambiguity, stale bearings, conflicts, directed
return and discovery counts. Its supplied perception events are not validated
RGB detections. Command odometry was tested on144 short streams and26 longer
routes/turns without refitting; the measured long-term and turning drift makes
uncalibrated point-estimate place merges inappropriate. Continue with actual
RGB association/exit/arrival integration. The fitted-checkpoint bridge replay
has now passed all1,114 actual selections; it supplied symbolic place/exit inputs
and certified no arrivals. See the [RGB/body evidence report](go2_rgb_body_observation_evidence_2026-09-05.md).

Latest integration update: the [actual RGB/body scan and full audit](go2_active_exit_scan_development_v1_result_2026-09-05.md)
are complete. Opening proposals now come from actual packets, but narrow-maze
calf contacts and accumulated heading error prevent scan qualification. The
[current sensor-to-navigation plan](go2_scan_to_navigation_next_steps_2026-09-05.md)
supersedes older immediate steps: causal high-rate IMU, articulated clearance,
actual traversal/arrival evidence, then a small discovery/return prototype.
The803-test suite passes. The [live fast-gyro comparison and full audit](go2_fast_gyro_scan_development_v1_result_2026-09-05.md)
now complete the ideal simulated high-rate orientation step. Both arms succeed3/4
at a new heading; high-rate error improves but narrow-wall contacts remain.
No study is running and no full-maze result exists yet. The
[instantaneous articulated geometry component and24-scan analysis](go2_articulated_scan_geometry_development_v2_result_2026-09-05.md)
are complete, including corrected native grouping. Next an actual observation-to-
arrival prototype; future-motion clearance remains unqualified.

## 1. Finish the current scientific decision

Keep the existing collection process and frozen 67-path source binding. Finish
all 144 trials or preserve its declared infrastructure-failure terminal. The
independent auditor now exists and its first36-trial audit passes:279 actual
packet-driven choices,2,710 images,160,423 physical samples and16,036 sensor
samples. Its two source/test paths are bound by that audit and must remain
unchanged. The11 native contact trials in this prefix are retained; this prefix
contains only two layouts and is not the eight-layout efficacy result.

When collection finishes, run the existing full auditor once. Publish the six
methods' all-trial contact/completion/release outcomes separately from progress,
with the fixed paired comparisons and missing-progress counts. Include actual
decision/action-change counts and inference versus acquisition timing. A method
that only stops is not a successful directional controller. A fast displacement
ending in contact is not safe navigation. Do not promote the JEPA arm merely
because one aggregate or one layout favors it.

## 2. Identify the missing causal coverage before another training run

The present training source samples one action branch per layout and derives
later windows along that same fixed-action branch. The extra temporal windows
do not provide all alternative actions after a moving prefix. Quantify the
training and executed past-action/future-action tables, including zero,
braking/reverse and changed curvature. Distinguish an initial teacher-stop
decision, a repeated constant action and a genuine action switch. Report the
actual observed half-second prediction errors and contact masks by these groups,
with within-layout counts. Such post-panel stratification is exploratory;
different groups visit different physical states, so their error difference is
not itself a causal estimate of switching harm.

If the gap is consequential, design a new bounded dataset of identical moving
prefixes followed by physically executed alternative suffixes. Keep exact
physical/body prefix matching, actual RGB and one role per entire layout. All
prefix failures and early contacts remain outcomes, not replacement invitations.
The cheapest useful first panel should cover each moving bank action followed
by stop, reverse, continuation and changed curvature. Fix its scene population,
prefix lengths, suffix horizons and total physical budget before launch. Do not
pretend an unexecuted alternative was observed or multiply the independent
layout count by deriving more windows.

Repeat the matched direct / supervised-rollout / JEPA comparison only after a
clearly stated data intervention. Use the same sensor history, image exposure,
seeds, schedule and downstream controller across arms. Keep a training-only
action/remaining-plan mean or residual-motion baseline: it previously beat all
neural motion heads. Inspect the units and separate motion/contact optimization
terms rather than assuming an unnormalized sum is well balanced. Changing those
terms is a separate preregistered intervention, not an unreported simultaneous
repair. Do not run a latent-loss coefficient or best-seed search until positive.

## 3. Give the controller an actual exploration task

The current .8-m cue is a direction transported by relative orientation. It is
not the remaining vector to a fixed goal: no relative translation estimator is
present. Its .5-s greedy ranking has neither an exploration objective nor a
demonstrated long-horizon benefit. Keep those limitations distinct from visual
prediction quality.

Qualify relative translation against evaluation-only physical traces with
reset, drift, stopping, turning and revisit cases; the runtime input must remain
deployment-valid RGB/body sensing. Command integration is a comparison, not
ground truth. Qualify look/reorientation primitives inside maze clearances,
including footprint drift and release motion; the existing large-arena gyro
assay does not certify narrow-junction turning. Feedback primitives require an
honest future-control contract: do not feed their future realized feedback
commands into a predictor at the current decision.

Add observed exit hypotheses, beacon observations and provisional place
associations, then route using demonstrated directed traversals. The existing
`DirectedTraversalGraph` is useful here: no assumed reverse edge and failed
attempts remain recorded. It cannot validate an upstream place identity.
The older `OnlineTopologicalMemory` commits edges from MAP changes/new nodes;
those appearance-filter transitions must not be silently reused as evidence of
a viable physical traversal. Separate observation association from execution
qualification. Test repeated-looking junctions, uncertain revisits, wrong merges,
missed revisits and reverse routes explicitly.

Only then compare the same local controller with and without persistent memory
on discovering initially hidden beacons and returning. A known route replay or
a prepopulated graph does not count as exploration. Use real observed beacon
identities, not a previously unseen target image or simulator beacon coordinates.

## 4. Close generalization and transfer honestly

Freeze methods and task endpoints after development, then evaluate new maze
layouts and declared sensing/appearance/dynamics shifts with maze-level
uncertainty. All currently inspected layouts are development material, not a
final test. Preserve protected benchmark custody.

Real-Go2 evidence still needs hardware access, actual camera/IMU/joint timing and
calibration, a deployable watchdog and human safety supervision. The simulated
native contact/attitude emergency stop is privileged experimental protection.
Software and simulation progress can continue now; the final transfer deliverable
cannot be claimed from them. The ultimate goal remains active and unachieved.

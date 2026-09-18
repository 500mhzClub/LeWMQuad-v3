# Native forecast horizons for the dense visual world model

Current status: collection, fit and evaluation complete. Both predictor arms
finished 1,760 updates in 31.1 minutes (session 66149, exit 0). The evaluation
also exited 0 (87270). See `go2_horizon_dense_predictor_evaluation_2026-09-18.md`
for all eight horizons, motion decoding, action-choice diagnostics and timing.
At 700 ms, visual error is 24.9% below action-blind and 41.7% below persistence,
with 18/18 action retrieval. Decoded motion remains worse than command history,
and the six-action/eight-horizon float32 computation takes 2.55 seconds.
No new navigation or real-time result is established.
Both collection workers (sessions 57658 and 89694) exited 0; the summary
(85343) confirms 48/48 eligible, zero contacts, eight matched context groups,
and exact reproduction of every original 16-frame RGB prefix. The new
records consumed approximately 68 MiB with all RGB/physics evidence retained.
Result: `go2_balanced_start_horizon_actions_result_2026-09-18.json`.

Fit preparation (63098) exited 0 with 29,008 context/horizon pairs and 6,170
frame paths. Each arm draws 3,520 examples per horizon, with 25,024 distinct
pairs used across its fixed 28,160 draws. The other available pairs are not
claimed to have been used. Identical-image reuse left 4,370 unique encoder
inputs, encoded in 919.7 seconds. Final 80-update mean training L1 was
0.2399 action / 0.2959 action-blind; these losses are not navigation evidence.
RAM feature allocation was 9.70 GB and was released at process completion.
The evaluator is `scripts/evaluate_go2_horizon_dense_predictor_development.py`.

The retained dense predictor forecasts only 500 ms ahead. The existing maze
planner consumes eight 100-ms trajectory samples, including a 700-ms
dispatch/commit endpoint. This experiment learns visual endpoints at each
requested time from actual recordings; it does not interpolate missing states.

The training-data audit found 3,998 / 3,878 / 3,758 / 3,638 / 3,518 / 3,398 /
3,278 / 3,158 available samples at 100 through 800 ms respectively. All are
from the original admitted training recordings. All required RGB files exist.
The recently collected balanced starts cover only the first five offsets, so
the same 48 training cells were collected with eight action ticks.
These are six physical motions across eight visual environments, not 48
independent robot dynamics. All failures and physical evidence are retained.
Completed-case mesh files share storage with byte-identical retained originals.

The predictor keeps the frozen pretrained V-JEPA 2.1 encoder and existing
three-frame context at -1000/-500/0 ms. Its action input is extended from five
to eight applied forward/yaw commands plus requested target time, adding
2,688 parameters. Commands after the target time are masked. The new channels
start at zero weights, preserving the parent's 500-ms output exactly.

The focused check passed on actual training RGB with the retained supplemented
action predictor: zero output difference at 500 ms, post-target command
invariance at every supported horizon, and finite nonzero gradients for all
new channels. Action-blind conditioning retains time while removing future
commands. This check made no optimizer updates and used no transfer examples.
Evidence: `go2_horizon_predictor_initial_check_2026-09-18.json`.

The planned fit compares action and action-blind predictors with matching
1,760 updates of batch 16 and fresh AdamW optimizers. Both inherit their
corresponding supplemented checkpoints and matched historical budgets;
their initial weights differ. Each batch contains two examples at each
horizon. Two batch positions use balanced starts, rotating across horizons.
All other examples come from the original training pool. The final step is
fixed in advance; no transfer result selects a checkpoint. The training
builder checks that original 500-ms controls and actions reconstruct exactly.
Features stay in host RAM; only final weights are written.

Evaluate all eight horizons on the retained matched-action branch panel
against action-blind prediction and persistence. Compare 500-ms performance
with the retained parent to detect regression. These exposed diagnostics
cannot provide independent maze generalisation evidence.

Before obtaining the new fit's evaluation results, the evaluator was extended
to measure scene-by-action interaction: subtract the mean over actions and
the mean over scenes, then add the grand mean. A static scene representation
plus a scene-independent action effect has zero interaction. The metric reports
prediction error relative to that zero-interaction reference, with raw target
energy retained and negligible-energy ratios left undefined. Exact small
examples verified ratios 0 / 1 / 0.25 for perfect, purely additive and
half-strength interaction predictions. This measures visual interactions;
it does not by itself establish geometry understanding or collision prediction.

The retained panel's matching physical motions are exactly equal across both
scenes in all six role/prefix pairs at all eight horizons, with no contacts.
Thus it cannot establish obstacle-dependent physical dynamics. This extends
the previously known 500-ms limitation across the entire target interval.
Evidence: `go2_horizon_evaluation_interaction_check_2026-09-18.json`.

All twelve branch groups have identical requested command prefixes through
300 ms and three distinct action sequences from 400 ms onward. The evaluator
therefore predicts and decodes each distinct applied prefix once per common
context, then broadcasts to matching candidates. This preserves exact ties
before actions diverge and avoids numerical batch differences masquerading
as action discrimination. There can be no action-discrimination claim before
400 ms on this panel. Evidence: `go2_horizon_branch_prefix_check_2026-09-18.json`.

The same evaluation will apply the existing frozen 500-ms motion decoder to
every predicted horizon and to actual future features. Scores away from
500 ms explicitly test temporal transfer of that head. No readout is refitted.
The command-history baseline is reused from its retained 36-by-8-by-3 causal
prediction array, with recorded hash and identical trial order checked; no
historical depth replay is needed. This measures the trajectory interface
needed by the maze planner without assuming that it must beat command history
before any closed-loop trial can be informative.

The runtime input boundary now has a small native-packet adapter:
`lewm/dense_native_observation_development.py`. It accepts only three actual
640x480 RGB observations spaced 500 ms apart and the current packet's complete
15-row causal applied-command history. It reproduces the existing PIL resize
and normalization directly from in-memory RGB, without writing temporary
images or upsampling the old 96x128 model tensors. On retained training hold
and right-turn sequences, its pixels and command ordering matched exactly.
Wrong spacing, missing frames, downsampled RGB and future command availability
were rejected. Check session 70225 exited 0; evidence is in
`go2_dense_native_observation_check_2026-09-18.json`. No GPU or fitting was used.

The old controller's acquisition packets contain native RGB, but its model
history keeps only four 100-ms frames. Integration must retain the native
frames needed at -1000/-500/0 ms and wait for that context before planning.
The adapter is not yet wired into a live runtime. Shared mapping, routing,
arrival and backtracking behavior must remain matched across comparisons.

This is an integration prerequisite, not a complete navigation solution. The
existing frozen motion decoder remains weak even with observed future images;
learning more visual horizons does not repair that decoder automatically.
Collision scoring, the visual-to-navigation interface and inference timing
still need explicit solutions before connecting this model to full-maze
exploration, persistent routing memory and physical backtracking. No real-time
or hardware claim, or JEPA encoder-objective advantage, follows from this fit.

Sources: `go2_native_horizon_training_support_2026-09-18.json`,
`go2_balanced_endpoint_navigation_2026-09-18.md`,
`scripts/collect_go2_balanced_start_horizon_actions_development.py`, and
`scripts/train_go2_horizon_dense_predictor_development.py`.

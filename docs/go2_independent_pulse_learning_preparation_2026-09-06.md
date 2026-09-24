# Independent-layout prediction study: preparation while fixed collection runs

The near-field correction and first l00 batch are already launch-frozen. This
work does not alter their sources, episode order, sensor inputs or stopping rules.
It prepares the next scientific comparison without training on partial results.
The full goal remains reliable RGB-plus-deployment-valid-sensor JEPA navigation,
online memory and independent novel mazes; prediction plumbing is not success.

## Completed source-level preparation

`lewm/independent_pulse_evaluation_development.py` joins an explicit pulse dataset
to the already validated prospective inventory. It rejects role/layout/action/
departure reassignment, reports every missing planned episode and missing history,
and keeps the0.5/1/1.5/2/2.2or2.5second target times distinct. It fits the existing
action/time baseline only on the exact eligible train-role draw indices used
for model exposure, retaining repeated draws rather than peeking at evaluation
labels. A missing prediction makes that head unavailable for the paired score;
it does not silently shrink the comparison to easy cases.

All named heads must supply identical ordered evaluation-row identities. The
existing layout-first metric reducer is reused, with action/context/history/
support and exact-horizon strata. Differences are paired per layout, then
averaged across layouts. Hundreds of correlated frames or horizons cannot
become hundreds of independent maze trials. The3development-evaluation layouts
are only3independent topology units; the interface does not manufacture a
frame-level confidence interval or final-generalization claim. The interface
does not verify source artifacts or physical visibility: the terminal batch
audit remains a prerequisite, and absent outcomes remain unknown.

Nineteen synthetic tests pass (35285,3.66s). They cover unequal layout populations,
train/evaluation leakage, duplicate/reordered indices, missing predictions,
partial-time boundaries, missing histories and mutation isolation. No recorded
layout was fitted or scored by this new evaluator during preparation.

## Cumulative-event semantics, not a risk-calibration claim

The completed one-room learning comparison exposed decreasing cumulative-contact
probabilities, despite all-negative labels. Correct this structural error before
a new independent-layout comparison, without claiming the correction solves
risk estimation. `lewm/cumulative_pulse_contact_development.py` defines a distinct
same-parameter model: each head's raw event scalar parameterizes a nonnegative
softplus conditional hazard rate over its actual known interval. Integrated
hazard H gives event probability1-exp(-H), which cannot decrease with time.
The final0.2second interval of a short pulse is not rounded to0.5seconds.

Computation is in log space for small rates, with stable low-rate asymptotes;
the20-log-unit branches incur below1.1e-9absolute log-rate error. Unknown plan
padding contributes no hazard or gradients. Finite extreme inputs near±1000
remain finite; actual cumulative overflow is rejected, never silently clipped.
All four motion components and all latent/action/history parameters are unchanged.
This parameterization alone is neither calibrated collision risk nor a safety
mechanism, and old contact checkpoints are not scientifically interchangeable.

`lewm/cumulative_pulse_learning_development.py` uses the same event transform in
direct-head training and inference, and exactly once in recursive decoding.
A forward-only wrapper would have left direct-head training on the wrong logits;
the new objective explicitly prevents that mismatch. All three direct/supervised-
rollout/JEPA arms retain matched online-encoder observation populations,6cm
position units, fixed regularization and identical EMA maintenance. Observed
cumulative labels may not revert from1to0, even across censored slots. New
checkpoints explicitly identify integrated-hazard semantics and position scale.

Twenty-two focused synthetic tests plus the19evaluation tests pass together
(77963,41passed4.87s). Tests verify the exponential law at exact times, monotone
probabilities, causal gradients, unchanged motion, same initial parameter hashes,
training-versus-inference loss equality, one synthetic optimizer step in each
arm, target-only separation, matched image exposure and latched invalid targets.
These synthetic steps are implementation tests, not new recorded-data fits.
The new sources are not yet a launch-frozen scientific experiment.

The explicit221-file regression subsequently passes2,836tests in226.58s
(35291,exit0). It includes both new evaluation and cumulative-event test files.

## Provisional collection finding that prevents an exact-pairing claim

Read-only verification17112 checks the first24committed l00 episodes:55,800native
samples,780RGB-D frames,120motion targets,zero positive contact targets and all
individual physical-visibility checks passing (max depth error0.367503mm).
However,0/20full-sensor sibling prefixes match exactly. The initial checked
pair has identical native/contact histories and non-RGB sensor fields; its first
RGB frame differs at3,718pixels with maximum channel difference114. A geometric
diagnostic finds3,560of those pixels on exactly coincident front faces of two
adjacent-wall pairs. The remaining158pixels are not explained by that centre-ray
test. Both images were visually inspected and show coplanar interference strips.

This is not repaired by lowering an RGB threshold or editing recorded pixels.
The fixed batch continues under its original per-episode checks and preserves
the mismatch. Its terminal audit must report actual prefix results and native
coverage before further data or learning decisions. A separate boundary-only
wall visual constructor and static repeatability bench are under investigation;
see the [frozen union-wall bench protocol](go2_union_wall_rgb_repeatability_probe_v1_2026-09-06.md).
No new learned-model fit has been run on these partial records.

## Required next experiment sequence

1. Complete and terminal-audit l00 before another layout decision. Then collect
   sufficient preassigned train/selection/development-evaluation layouts without
   redefining roles or dropping failed near-wall/support contexts. Preserve raw
   missing-data accounting, exact sibling-prefix checks and visibility failures.
2. Assemble terminal-audited, visibility-valid windows only. Record exclusion
   and native contact/motion/future-image coverage separately. If there are no
   actual positive contact outcomes, risk learning remains untested; do not
   manufacture positives or use all-negative accuracy as hazard evidence.
3. Freeze one bounded matched study before any fit: same train-only draw schedule,
   three paired model seeds, direct/supervised-rollout/JEPA arms, same sensors,
   auxiliary labels, architecture width and update budget. Include zero-motion
   and the exact train-exposure action/time baseline. Bind the cumulative-head
   semantics for all arms; do not retrofit them into a predecessor checkpoint.
4. Establish action/history/RGB utility with causal input ablations on the same
   independent populations. Report condition-specific failures and paired layout
   effects, not a selected best seed. Stronger results require more independent
   layouts and a later untouched confirmation, not just more correlated windows.
5. Only after useful prediction and dependable local execution, compare online
   rollout against no-rollout candidate selection under the same sensor/control
   substrate, then memory-on/off with actual branches and executed backtracking.
   Robust tracking/slip response, real-time deadlines, realistic/self-occluding
   sensing and bounded hardware evidence remain required. Latest room-return
   remains0/3 and previous fitted heads still lose the empirical motion baseline.

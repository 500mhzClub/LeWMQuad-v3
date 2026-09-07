# Fixed development comparison: context-matched action coverage

## Question and scope

Does observing alternative actions from previously moving RGB/body contexts
improve predictions and offline action ranking, while retaining performance on
the original temporal windows? Does JEPA training help relative to matched
supervised prediction under each coverage condition? This is development-only,
not held-out evaluation, probability calibration, executed navigation, discovery,
directed return or hardware qualification. No protected benchmark is accessed.

The original temporal comparison and the later bounded positive successive-choice
panel remain valid within their different scopes. This new context-balanced
schedule differs from the earlier action-balanced schedule; historical model
differences are not an isolated coverage effect. Compare the new paired arms.

## Fixed population and intervention

Require completed original914-window derivation/audits and new384-trial physical
collection/full raw audit, plus the completed600-cell actual tensor qualification.
Retain all914 original windows and add384 switch suffixes:866 train/432 validation
across the same16/8 development layouts. There are546/272 distinct current
contexts, of which64/32 gain four future actions. No new current state is created.
All contacts are retained with existing strictly precontact motion/future-image
masks. The new suffixes have only three seconds of known actions. No release or
unobserved horizon is relabeled as an executed future.

For each of1,200 updates, sample one current context uniformly within each of
the16 training layouts, then one uniform action quantile shared across data
conditions. `coverage_limited` chooses among that context's original actions;
`expanded` chooses among all recorded actions. Thus current context exposure
and update/layout weighting are exactly paired. Initial and other unchanged
contexts select identical examples. New future trajectory/action/mask exposure
is the intervention; it is not held fixed between data conditions. Sampling is
with replacement, not a promise of equal frequency for every window.

Actual metadata-only preflight found all256 new training suffixes visited for
each fixed seed, with1,837/1,805/1,795 expanded-only examples among19,200 paired
examples. This was not a fit or outcome-driven selection. At materialization,
require exact equality of all current history tensors within each context.

## Fixed fitting

Exactly18 models: two data conditions × three training conditions (`direct`,
`supervised_rollout`, `jepa`) × seeds2026092400,2026092401,2026092402. Use the
unchanged128-dimensional four-frame temporal model and its unchanged loss code.
All six arms share initialization within each seed. All three training conditions
within a data arm receive identical sampled windows and actual future-image
populations. Direct still gets the shared variance/covariance regularization;
supervised rollout adds outcome supervision on the recurrent head; JEPA adds
future-latent prediction against the EMA single-actual-packet target.

CPU, one thread, deterministic PyTorch algorithms,1,200 updates per model,
AdamW lr0.0003, weight decay0.0001, gradient norm clip5, EMA0.99. Fixed loss
weights: direct1, rollout1 when active, JEPA latent1 when active, variance0.1,
covariance0.01. Use final checkpoint only. No early stopping, best seed, validation
tuning, weight search, continuation, retry or hidden replacement. Direct has fewer
active parameters than the two transition models; the JEPA-vs-supervised-rollout
contrast is parameter matched. Equal updates do not imply equal compute cost.

## Fixed evaluation

Score all432 validation windows with identical targets across fitted arms.
Report original initial/later windows separately from one-second moving
continuations and new switched actions. Report first-half-second, three-second
and all-known horizons, with actual mask counts and layout-macro position error,
yaw error and contact Brier. First-half-second switched-action position error
and Brier are the primary predictive endpoints; three-second and original-window
retention are prespecified secondary endpoints. Report null for empty endpoints.
Do not promote a low average by deleting difficult/censored layouts or contacts.

For each of32 validation moving contexts, rank its five observed actions using
three fixed forward/left/right0.8-m direction cues at0.5 s and3 s. Cost is
10×predicted contact probability plus predicted endpoint distance to the cue,
with lowest action index breaking ties. Realized cost is10 on an observed
contact, otherwise actual endpoint distance. Record regret against the observed
five-action minimum, chosen contact, chosen stop, and the actual always-stop
alternative. Unknown contact horizons are explicitly unavailable; never infer
missing outcomes. These96 correlated rankings/horizon are not96 independent
mazes or closed-loop success trials, and the three-second score is not the
existing half-second online controller.

Use training-only action/offset/horizon empirical means (layout-weighted),
zero-motion/no-contact and fixed command kinematics controls, separately for each
data arm's available training population. The empirical control deliberately
does not condition on past action or match the stochastic training frequencies;
label it as this simple control, not an optimal scene-free dynamics model.
Record its training-only action/horizon fallback counts.

RGB-only and body-only shuffle controls use cyclic donors from different layouts
with identical past action, future action and offset. Only complete all-layout
cells are eligible; report intact on exactly that eligible subset as well as all
intact examples. Preserve command histories and plans. Store predictions and
intact context latents, but do not infer noncollapse merely from RGB sensitivity.

Coverage contrasts compare expanded minus limited for all five usable heads.
Within each data condition compare JEPA minus supervised for each head,
rollout minus direct head within each predictive model, and JEPA direct minus
direct-only. Average three paired seeds within each layout; bootstrap the eight
layouts10,000 times with fixed seed2026092499 for descriptive percentile intervals.
Report all contrasts/endpoints without multiplicity-adjusted significance claims.
Reused validation layouts are not fresh evidence for final generalization.

## Integrity, resources and terminal behavior

Exact fresh output:
`.generated/go2_context_matched_coverage_learning_development_v1_attempt_001`.
Runner: `scripts/run_go2_context_matched_coverage_learning_development_v1.py`.
No command-line overrides. Before creating output, verify prerequisite source,
input and gait bindings and bind the complete ignore-aware imported source/test
closure plus this protocol. No source export, GPU, physics or hardware execution.
Keep launch, schedules, validation order, update logs, final checkpoints, intact
and shuffled predictions, baseline outputs and per-model results. Bind checkpoint
launch/schedule/initial identity and hash artifacts. Reverify source/inputs/schedule
before declaring COMPLETE. Any integrity/nonfinite/runtime exception is a retained
INFRASTRUCTURE_FAILURE, not a scientific negative; do not edit executed source or
restart this root. No success claim until a separate full checkpoint/prediction,
schedule and metric replay audit passes on these same bytes.

Preflight machine had64 GiB available RAM and50 GiB free disk. Materialize only
the explicitly permitted development tensors;18 small checkpoints and compressed
predictions fit this remaining disk comfortably. Monitor resource/runtime errors;
do not delete existing artifacts or start competing physical studies to make room.

## Next decision

Regardless of sign, preserve all results. If coverage improves the intended
switched endpoints without severe original-window regression, test its fixed
final-seed ensemble in a separately specified fresh successive-choice panel,
keeping online cost/horizon unchanged to isolate coverage. If not, examine the
prespecified horizons, action means and sensor controls to distinguish data,
representation and objective limitations before changing architecture. Neither
outcome substitutes for real observed exit/place/beacon and arrival integration,
longer uncertainty-aware planning, independent maze task completion or bounded
real-Go2 transfer evidence.

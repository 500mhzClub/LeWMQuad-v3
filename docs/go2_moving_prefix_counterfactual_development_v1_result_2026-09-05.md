# Moving-prefix counterfactual data: completed and qualified

All384 physical trials completed and passed independent full raw audit. Every
moving conditioning prefix reproduced the corresponding original physical/body/
camera state, with RGB differences within the existing fixed tolerance. No trial
was retried, resampled or discarded. The [fixed protocol](go2_moving_prefix_counterfactual_development_v1_2026-09-05.md)
and all executed source remain unchanged.

There are103 native-contact trials:100 during the three-second alternative
suffix, plus three during the later zero-command release. All384 prefixes are
available; the other281 trials finish without native contact. Contact is a
retained training outcome, not an integrity failure or proof of policy quality.
There was no learned action selection in this collection.

Recorded totals:1,710,516 physics samples,171,005 ideal body-sensor samples and
28,883 actual RGB packets. The independent audit reconstructs native contacts,
clocks, requested/applied command stages and slew, teacher/moving endpoints,
prefix identity, RGB/body histories and separate three-second target censoring.
Release samples do not label an unexecuted suffix horizon.

Collection1894 and full audit13100 both terminated exit0. Root:
`.generated/go2_moving_prefix_counterfactual_development_v1_attempt_001`.

- Launch SHA: `f8f1c6e01017aa08363bd8ad6997145e9df1a4fe52f115d1a88536608a4f3ae7`.
- Physical result SHA: `c50e6943739b19a834a623b0d8c40f5ef7f9f368d4d340a49e375f8a60dbe2e0`.
- Full raw audit SHA: `9709c7105637ee7682952578962a149cd126eb42bcab9f8d52cc0629e75e0235`.

## Actual composite tensor check

The separate [full tensor check](go2_moving_prefix_tensor_check_development_v1_2026-09-05.md)
passes all600 cells:400 train/200 development-validation, zero unavailable.
It combines120 old initial branches,96 old one-second moving continuations and
384 new alternatives. Across all120 layout/past-action conditioning groups,
the current four-frame RGB/body/control tensor hashes are identical between
all five future actions. Only the prospective plan and actual future targets
change. Policy loading opened no prohibited raw-physics/camera-world/contact
files under the instrumented guard.

The600 cells contain3,840 known contact horizons,315 positive contact labels and
3,525 valid strictly precontact motion/future-image targets. Of4,800 tensor
horizon slots,960 are outside known plans and remain masked. All new suffixes
have only six known half-second horizons; their3.5/4 s slots are not filled
from release motion. Actual labels are not imputed from a model.

Session35152 terminated exit0. Root:
`.generated/go2_moving_prefix_tensor_check_development_v1_attempt_001`.
Result SHA: `73bf946af00a689c8abac2cce5a79eabf421c73cde3972b2073682bcda093b3e`.
Launch SHA: `b24efbb344a9b4b173d8635575c002343da2fe48163383e2d4350a797b704e00`.
The composite loader, its tests and this check/protocol are now bound by execution.

## Retain old temporal coverage in the matched comparison

Execution update: the [fixed18-model comparison](go2_context_matched_coverage_learning_development_v1_2026-09-05.md)
has now completed and passed full audit in its separate development root. See
the [scientific result](go2_context_matched_coverage_learning_development_v1_result_2026-09-05.md).
The runner, augmented sampler,
metrics, tests and protocol are launch-bound. The descriptions of proposed work
below are retained as design history; do not launch a second copy.

The600-cell counterfactual view omits698 other old temporal windows. The new
`AugmentedCausalDataset` in `lewm/context_matched_coverage_development.py` retains
all914 old windows and adds384 switches:1,298 total,866 train/432 validation.
The actual metadata preflight passed (session68273, exit0): training has546
distinct current contexts and validation272. Exactly64 training/32 validation
contexts acquire four extra observed actions; no new current context is created.

The paired sampler chooses the same current context and action quantile for
coverage-limited and expanded training conditions. At original contexts with
unchanged action support, the selected example is identical. At one-second
moving contexts, limited training receives the old continuation and expanded
training can receive any of the five observed actions. Every update includes
one context per training layout. This keeps current-image exposure fixed and
retains old time strata; new future trajectories remain the intended intervention.
It is a new context-balanced design, not a replay of the earlier row-sampling
training experiment.

For prepared seeds2026092400–2402,1,200 updates yield19,200 paired examples per
seed, with1,837/1,805/1,795 expanded-only action examples respectively. All256
new training windows appear in each prepared schedule. These counts came from
metadata only, not fitted outcomes or validation selection. The sampler's13
tests pass, but this source is not yet launch-frozen and no new model was fitted.

Next implement and freeze the matched comparison: two data conditions × direct,
supervised-predictive and JEPA training × three declared seeds. Keep architecture,
loss units/weights, update count and downstream decision rule unchanged within
this intervention; evaluate direct and latent heads without best-seed selection.
Include training-only action/plan baselines. A later analytical-motion plus
learned contact/residual design is a separate intervention: do not quietly mix
it into the data-coverage comparison.

These remain the original24 development layouts, not600 independent mazes or a
final test. Real RGB association/exit/arrival evidence, autonomous beacon
discovery/return, independent final layouts/shifts and supervised hardware
transfer remain unfinished. Dataset qualification is not final-goal success.

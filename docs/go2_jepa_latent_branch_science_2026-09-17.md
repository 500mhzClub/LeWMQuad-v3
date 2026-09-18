# JEPA latent prediction on recorded action branches

**Complete. The current JEPA beats latent persistence at 800 ms, but does not
establish useful action- or scene-specific foresight. Stronger simple baselines
outperform it on this exposed development population.** All 12 action-branch
groups had exactly identical causal RGB/body/control tensors before their plans
diverged. All predictions were saved before loading future observation packets.
The primary run exited zero in 3.66 seconds; no weights or commands changed.

Primary 800-ms results on all 18 transfer contexts:

| Model | Prediction MSE / own persistence MSE | Correct action-branch retrieval | Correct RGB-scene retrieval | Persistence scene retrieval | Motion XY RMSE (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEPA | 0.366 | 6/18 | 10/18 | 10/18 | 6.561 |
| Supervised rollout | 388.738 | 6/18 | 9/18 | 15/18 | 5.628 |
| Untrained features + fitted readout | 358.521 | 6/18 | 9/18 | 9/18 | 5.644 |

The common command-history motion reference has 8.232-mm XY RMSE on these
specific 18 contexts. This limited pulse result does not supersede its stronger
performance on the completed full-navigation recordings. The supervised and
untrained latent transitions were not trained to align with EMA target
embeddings; their large latent errors are expected incompatibility diagnostics,
not a fair standalone measure of JEPA's usefulness. Physical readout errors
do share common targets. JEPA is not the best readout on this population.

JEPA action retrieval is 8/18, 10/18, 8/18, 6/18 and 6/18 at 400–800 ms.
The first three horizons precede the differing pulse and cannot distinguish
the action branches; all three future targets tie there. Correct-action JEPA
forecast MSE is 0.021411 versus 0.021981 averaged over both wrong-action
forecasts at 800 ms. That small difference does not yield reliable branch
identification. Scene retrieval matches persistence at the primary horizon.
JEPA target effective rank over all transfer horizons is 2.77/32 and mean
coordinate standard deviation 0.0959. This small, repetitive population does
not suffice to declare representation collapse or general representation quality.

## Explicitly post-hoc stronger baselines

`scripts/read_go2_jepa_branch_decomposition_development.py` completed after the
primary results. It uses only training targets for two simple predictors,
preserving all primary results. No transfer labels fit either baseline.

| Predictor, in the JEPA target space | Transfer 800-ms latent MSE |
| --- | ---: |
| Current JEPA | 0.021411 |
| Current-target persistence | 0.058577 |
| Training target mean at this horizon | 0.002911 |
| Training target mean given preceding action and horizon | 0.000453 |

The latter two baselines receive neither current RGB nor future action. The
current JEPA is about 7.35 times worse than the horizon mean and 47.2 times
worse than the preceding-action mean. These means exploit this collection's
repeated motion structure and are not general world models, but rule out
treating the persistence improvement as sufficient evidence of useful foresight.

Separately center predictions and targets within each three-action group to
remove their common component. JEPA's centered action-effect MSE is 0.0003183,
versus 0.0001933 for predicting zero action effect, a ratio of 1.647. This
centering uses alternative-branch outcomes and is diagnostic only, not a
deployable predictor or revised primary result. It shows that common-state
misalignment is not the only deficiency on these branches.

Complete primary result: `docs/go2_jepa_latent_branch_science_result_2026-09-17.json`.
The artifact root retains `causal_predictions.npz`, `encoded_targets.npz`,
`result.json`, and `posthoc_decomposition.json`, including all train/transfer
horizons, input identities and model identities. No new depth archive exists.

## Consequence for the next experiment

Prioritize learning the latent dynamics, not another navigator repair or broad
navigation repetition. The next controlled model experiment should separate
representation quality from transition-fitting quality, retain the same
training-only data and matched supervised/untrained controls, and test against
the stronger training-mean and action-effect baselines. A motion-only readout
gain or low latent loss alone is insufficient. Prospective navigation should
then test a demonstrated predictive change with the shared perception/controller
held fixed. Multiple seeds and new independent scenes remain necessary before
claiming generalization or planning benefit. No JEPA advantage is established
by this diagnostic, and no model is promoted.

User priority: focus development on core JEPA scientific claims. Gyro-error
correction and sim-to-real sensing work are deferred; existing sensitivity
results remain. No further perception/controller variant is introduced here.

The current navigation pipeline predicts short-horizon motion around a fitted
command-history reference. Its strong baseline results make motion RMSE alone
an insufficient test of learned visual world dynamics. This diagnostic asks
whether the current action-conditioned latent predictor predicts measured
future representations beyond persistence, distinguishes actual action
branches, and preserves visual scene information under matched body/control.

Use the fixed frame-13 departures from all 36 existing short-pulse episodes:
18 training contexts and 18 previously exposed development-transfer contexts,
with two geometry clusters per role. Each geometry has three preceding-action
conditions and three alternative pulse actions. The eight-step plans share
three committed zero commands before their different one-tick pulses. Verify
that all RGB/body/control histories are exactly identical within each three-way
branch group before using it for action discrimination. No training, new
collection, model selection or navigation occurs.

Load the same frozen JEPA, supervised-rollout and untrained-feature readout
models. Report all 100–800-ms horizons, with 800 ms primary. Predictions are
saved before future RGB/body/control packets are loaded. Compare each model
against persistence of its own current EMA target embedding and both wrong
action forecasts. Raw latent errors across models do not share a representation
space and cannot establish superiority by their numerical size.

For visual scene discrimination, replace only future RGB with the matching
other geometry's RGB, preserving the true future body/control tensors. Compare
true-versus-substituted target distance, including a persistence comparator.
Also swap only causal history RGB between these matched scene pairs and
measure its effect on forecasts. Report target variance/effective rank and
the existing physical-motion readout/reference errors as complementary evidence.
Scene discrimination may rely on static appearance; it does not alone establish
prediction of scene change or useful navigation. The pulse families have
identical native motion across geometry repetitions. This is a small exposed
development diagnostic, with one training seed and no independent final result.

Plan: `docs/go2_jepa_latent_branch_science_plan_2026-09-17.json`.
Runner: `scripts/probe_go2_jepa_latent_branch_science_development.py`.
Output: `go2_jepa_latent_branch_science_v1_attempt_001` on the artifact volume.
Before execution, CPU was 96% idle, 72 GiB RAM available, GPUs idle and 4.4 GiB
artifact space free. One CPU process on core 8, one numerical-library thread,
shares the small loaded population across models; extra workers would duplicate
these inputs for little benefit. No depth data is read or generated.

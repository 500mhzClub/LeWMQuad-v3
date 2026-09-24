# Go2 world-model progression V1: fixed 2×2 mechanism screen

Date: 2026-08-02

Status: **PREREGISTERED DEVELOPMENT COMPARISON; NOT A WORLD-MODEL
USEFULNESS CLAIM**

## 1. Decision this experiment can make

This experiment asks a narrow question on the already-built ordinary H6 pack:

> At a fixed terminal update and across three training seeds, does either a
> frozen true-latent-displacement action objective or dense spatial target
> prediction improve the existing one-step action-conditioned JEPA head?

It is a controlled mechanism screen, not the repository's missing causal
world-model experiment. It uses one factual successor per history. It cannot
show that a prediction for an untaken action matches that action's executed
successor, that blind rollouts remain accurate, that predicted costs rank true
physical regret, or that a planner improves navigation.

The four cells are:

| spatial target | plain JEPA | JEPA + frozen displacement-action loss |
|---|---|---|
| registered 64-token mask | `masked_plain` | `masked_delta` |
| complete 256-token grid | `full_plain` | `full_delta` |

The run is row-, update-, initialization-, batch-order-, optimizer-, and
fixed-terminal-matched. It is not FLOP- or wall-time-matched: the full-grid
cells predict four times as many target queries and delta cells execute an
additional frozen decoder. Any factor attribution therefore includes that
mechanism's additional computation.

## 2. Frozen source and runtime

The run is admissible only if the following six-file runtime closure matches
byte-for-byte. The runner records these bindings at start and rechecks them
before publishing `result.json`.

| path | bytes | SHA-256 |
|---|---:|---|
| `scripts/dev_train_go2_world_model_progression_v1.py` | 44,968 | `0cb15c6414d7deeda6c206981457c72a45558905ea695cdae924a844702d49e0` |
| `lewm/models/go2_world_model_progression_v1.py` | 12,224 | `b7582059034a1af475595b33f1a369a61aafa933564ec9e8d25317022680c0fb` |
| `scripts/execute_go2_world_model_existing_pool_three_arm_v1.py` | 103,001 | `b0ca02d706b0108885e51b32353b8af6e440259a0108fa934ad8bd9e70366d7d` |
| `scripts/dev_train_temporal_jepa_scaled.py` | 35,478 | `97154b693d3ca2b96e7e0d88c378c07b349c1d77e0db05548e48823834a35037` |
| `lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py` | 17,480 | `324bd76eb0f8285dac01ccba0741e38546fe93506f5afd51facfaa71826346c3` |
| `lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py` | 44,605 | `cec018dade02e4c8217d74792f8fdc6afba84f414d094f4879e353d78fee4f84` |

The separate offline analyzer and its focused test are:

| path | bytes | SHA-256 |
|---|---:|---|
| `scripts/analyze_go2_world_model_progression_v1.py` | 18,632 | `37fb6a306bc2e1370581e968094cdf9453a0e9df2d8814079dc2a92368ca6f31` |
| `lewm/tests/test_analyze_go2_world_model_progression_v1.py` | 5,547 | `1d64601b2c4a5b957189a2fac883b143f55a57826aea125489324a10ad297e43` |

The exact interpreter is
`.generated/venvs/world_model_rocm_7_2_1_v1/bin/python`, with PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`, NumPy `1.26.4`,
and Pillow `11.3.0`. The runner requires ROCm `cuda:0`, enables strict
deterministic algorithms, treats a determinism failure as fatal, and performs
no fallback.

Before execution, these files and this document must be frozen in a narrow
source commit. A hash mismatch requires a new review and preregistration; it
must not be silently treated as the same run.

## 3. Frozen data and predecessor

The only data root is:

`/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/pack`

Its manifest is 5,297 bytes with SHA-256
`22364f911ab5d3e2956ea9a3fc2d92e2869830cd858ef2d2269379dfc6041bae`.
The exact role bindings are:

| role/artifact | rows or bytes | SHA-256 |
|---|---:|---|
| train row identity | 16,000 rows | `9bd2b1bb89d7290b4dcae8490e3188f14d0072b73e1ce0e67de503fe976b6809` |
| train frames | 2,408,448,000 | `df9a5982370f4ba7c5d1c492f080d44f9900d889877ddb73f08454ba151a5a74` |
| train actions | 384,128 | `11bfcd0724397be8fc84969a32c01b71d41fdedb34c75bbc7a9e4d481a934a78` |
| train metadata | 928,029 | `2f265eaa57979f2e9c49956ab7bf83df29bcbc75d6b2f274f4d9b7b5d9635265` |
| validation row identity | 2,048 rows | `2d1859118824a99b52027d97ef2a406f3571cdf349325ca3b6b7f646f7554963` |
| validation frames | 308,281,344 | `e457d244c07516947ffb8005e2477d9a7f48c5e6a03b8701cf994debb06f6d66` |
| validation actions | 49,280 | `ad1b33d6ff4839736e27d37114bb1c01ca1cae693b5317c055dc9e776a8be6a1` |
| validation metadata | 118,813 | `6ef0d194c45a60d9cc28806dd8158360ae4ea6da55caf8685bdcdda9cfeff2a4` |

The pack contains 1,000 training scenes and 150 validation scenes across eight
families. Each family contributes exactly 2,000 train rows and 256 validation
rows. The validation final-action counts for IDs 0–8 are respectively
`[395, 168, 130, 74, 548, 51, 77, 353, 252]`; balanced accuracy is therefore
required instead of ordinary accuracy.

The frozen encoder/target predecessor is
`.generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1/attempt_v1/snapshots/update_1000.pt`,
52,282,877 bytes, SHA-256
`f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873`.

No sealed, held-out, test, navigation, or production role is in scope. V4 and
every legacy sealed role remain inaccessible.

## 4. Frozen training design

The seeds are exactly `2026080201`, `2026080202`, and `2026080203`.

For each seed:

1. Construct one frozen spatial substrate from the bound update-1000
   predecessor.
2. Initialize four independently allocated predictor/memory cores from the
   same exact state. No trainable parameter identity may be shared across
   cells.
3. Initialize one action decoder at seed `seed + 17`.
4. Pretrain that decoder for exactly 300 updates on true current-to-future
   target-encoder displacement. Each update uses 256 rows in microbatches of
   16. Masked and full panels contribute equally to the decoder loss.
5. Freeze the decoder permanently. Its source and terminal state hashes must
   match; decoder parameters may not accumulate gradients during predictor
   training.
6. Train every core for exactly 700 updates, batch size 256, microbatch size
   16, using the same row order within a seed. There are 179,200 row
   presentations per cell. All cells use the historical AdamW predictor and
   memory groups, four-times learning-rate scale, 150-update warmup,
   3,000-update cosine schedule, weight decay `1e-4`, and gradient clip `1.0`.
7. `masked_plain` and `masked_delta` use paired stochastic seeds;
   `full_plain` and `full_delta` use a second paired stream. Spatial-factor
   comparisons are therefore not paired to the same dropout stream, although
   initialization, rows, and schedule are shared.
8. Evaluate all 2,048 validation rows at update zero and fixed update 700.
   Intermediate traces are diagnostics only. No validation metric selects a
   checkpoint.
9. Save exactly one CPU-only update-700 snapshot per cell. Do not resume,
   extend, or choose an earlier checkpoint.

The delta loss coefficient is exactly `0.1`. The target for the decoder is the
factual final action ID. The decoder never receives an action as input, but the
predictor does; a frozen decoder prevents decoder/predictor co-adaptation but
does not prevent the predictor from learning an adversarial code that exploits
the frozen decoder.

## 5. Decoder anchor gate

The expensive comparison is preceded by three independent engineering anchor
runs, one per registered seed. Each anchor uses the complete 2,048-row
validation set and the exact 300-update decoder pretraining. It runs only one
core update, writes no snapshots, and is not scientific evidence.

For both the masked and full true-latent panels, in every seed, the
scene-cluster bootstrap lower 95% endpoint of nine-way balanced accuracy must
be strictly greater than `1/9`. The bootstrap uses 2,000 draws, seed
`20260802`, and the 150 exact validation-scene clusters. The runner threshold
is deliberately set to zero only so each seed emits its complete engineering
receipt; the external gate remains strictly greater than `1/9`.

If any of the six lower endpoints fails, stop this action-grounding mechanism.
Do not increase decoder width, pretraining updates, coefficient, data prefix,
or seed count. An anchor failure says the proposed frozen decoder is not a
usable supervisory instrument on this pack; it says nothing about causal
dynamics.

The exact commands, each requiring a fresh absent output directory, are:

```bash
.generated/venvs/world_model_rocm_7_2_1_v1/bin/python scripts/dev_train_go2_world_model_progression_v1.py --output .generated/dev/world_model_progression_v1/anchor_gate_seed_2026080201_v1 --pack-root .generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/pack --seeds 2026080201 --updates 1 --batch-size 256 --microbatch-size 16 --eval-batch-size 64 --decoder-pretrain-updates 300 --decoder-trace-every 50 --minimum-decoder-anchor-lower-bound 0 --eval-rows 0 --skip-snapshots --trace-every 1
```

```bash
.generated/venvs/world_model_rocm_7_2_1_v1/bin/python scripts/dev_train_go2_world_model_progression_v1.py --output .generated/dev/world_model_progression_v1/anchor_gate_seed_2026080202_v1 --pack-root .generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/pack --seeds 2026080202 --updates 1 --batch-size 256 --microbatch-size 16 --eval-batch-size 64 --decoder-pretrain-updates 300 --decoder-trace-every 50 --minimum-decoder-anchor-lower-bound 0 --eval-rows 0 --skip-snapshots --trace-every 1
```

```bash
.generated/venvs/world_model_rocm_7_2_1_v1/bin/python scripts/dev_train_go2_world_model_progression_v1.py --output .generated/dev/world_model_progression_v1/anchor_gate_seed_2026080203_v1 --pack-root .generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/pack --seeds 2026080203 --updates 1 --batch-size 256 --microbatch-size 16 --eval-batch-size 64 --decoder-pretrain-updates 300 --decoder-trace-every 50 --minimum-decoder-anchor-lower-bound 0 --eval-rows 0 --skip-snapshots --trace-every 1
```

For each receipt, inspect
`seed_results.<seed>.decoder_anchor_balanced_accuracy.masked.lower_95` and
`.full.lower_95`. Do not average away a seed or panel failure.

## 6. Fixed comparison command

Only after all six anchor endpoints pass, execute exactly:

```bash
.generated/venvs/world_model_rocm_7_2_1_v1/bin/python scripts/dev_train_go2_world_model_progression_v1.py --output .generated/dev/world_model_progression_v1/comparison_20260802_v1 --pack-root .generated/dev/world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1/pack --seeds 2026080201 2026080202 2026080203 --updates 700 --batch-size 256 --microbatch-size 16 --eval-batch-size 64 --decoder-pretrain-updates 300 --decoder-trace-every 50 --minimum-decoder-anchor-lower-bound 0.1111111111111111 --eval-rows 0 --trace-every 100
```

Then run the offline analyzer once into an absent path:

```bash
.generated/venvs/world_model_rocm_7_2_1_v1/bin/python scripts/analyze_go2_world_model_progression_v1.py --result .generated/dev/world_model_progression_v1/comparison_20260802_v1/result.json --output .generated/dev/world_model_progression_v1/comparison_20260802_v1/analysis.json
```

The analyzer validates the complete configuration, inputs, six-file source
closure, anchors, four-cell terminal panel, unchanged frozen decoder, and all
twelve fixed snapshots. It hashes snapshot files without opening tensor
payloads so the downstream causal evaluator can bind exact artifacts.

## 7. Frozen hypotheses and contrasts

Let the oriented terminal metric in seed `s` be:

- `MP_s`: `masked_plain`;
- `MD_s`: `masked_delta`;
- `FP_s`: `full_plain`; and
- `FD_s`: `full_delta`.

Higher is favorable for hardest-wrong-action margin, persistence advantage,
nine-way balanced accuracy, and the recorded hardest-margin q05. Factual
energy is multiplied by `-1`, so higher oriented value means lower energy.

For every metric and seed, report:

```text
delta main effect   D_s = 0.5 * [(MD_s - MP_s) + (FD_s - FP_s)]
spatial main effect S_s = 0.5 * [(FP_s - MP_s) + (FD_s - MD_s)]
interaction         I_s = (FD_s - FP_s) - (MD_s - MP_s)
```

Also report all four simple effects. The primary directional hypothesis is
that `D_s` for `hardest_wrong_action_margin_mean` is positive: displacement
action supervision should make the factual action's successor prediction
more favorable than its hardest nominal alternative. The secondary delta
hypotheses are positive persistence advantage, action balanced accuracy, and
factual-energy utility.

The dense-spatial hypothesis is reported through `S_s` on the same registered
64-token comparison surface. A 256-token full-grid diagnostic is additionally
reported within the two full cells, but it is not mixed numerically with the
64-token cells. Structural re-entry is a tested interface property, not a
claim that blind rollouts are accurate.

The interaction is diagnostic: a positive value means delta supervision helps
more under the full grid. There is no multiplicity-adjusted confirmatory claim
from these proxy metrics.

## 8. Training-seed uncertainty and meaningful-proxy rule

For each factorial effect, report the three seed values, their arithmetic
mean, sample standard deviation, minimum, maximum, positive-seed count, and a
two-sided Student-t 95% interval with two degrees of freedom and critical
value `4.302652729911275`.

With only three seeds this interval will be wide. It describes optimization
variation conditional on one reused validation panel. It is not fresh-scene,
counterfactual-branch, or deployment uncertainty. The decoder anchor's
scene-cluster interval is the only scene-resampled interval in this runner;
the other terminal metrics are row-weighted aggregates.

The exact proxy-allocation flag for the delta mechanism is:

1. hardest-margin `D_s > 0` in all three seeds;
2. the mean hardest-margin delta main effect is at least
   `0.25 * max(0, -P)`, where `P` is the across-seed mean of the two plain
   cells' hardest-margin means; and
3. the mean persistence-advantage delta main effect is nonnegative.

The 25% rule demands closure of a nontrivial fraction of the concurrent plain
model's gap to the physically necessary zero-ordering boundary. It is a
resource-allocation heuristic, not proof of causal fidelity. The analyzer
emits either `DELTA_PROXY_MEANINGFUL` or `DELTA_PROXY_NOT_MEANINGFUL`.

## 9. Stop and advance rules

Apply these rules without coefficient or seed tuning:

1. A source/input/configuration/determinism/nonfinite/snapshot-integrity
   failure is `INCONCLUSIVE_CONTRACT_FAILURE`. It is not a model result and
   grants no automatic retry under changed source.
2. Any decoder-anchor failure stops this delta objective before the 700-update
   comparison. Do not retune it.
3. If the fixed comparison completes, analyze update 700 only. Update zero and
   traces explain training but cannot select a checkpoint.
4. `DELTA_PROXY_NOT_MEANINGFUL` forbids scaling this delta objective to a
   larger observational pack. It does not remove the already-produced
   snapshots from the predeclared causal branch comparison.
5. `DELTA_PROXY_MEANINGFUL` permits the delta cells to proceed to the same
   frozen matched-state causal branch evaluator as both plain cells. It does
   not by itself authorize more training or a usefulness claim.
6. All four completed cells should be scored on the same causal branches. This
   avoids choosing a model from the reused factual validation set. Direct
   branch truth, outcome-equivalence-aware action regret, and per-seed paired
   effects are the adjudicators.
7. A full-grid cell proceeds to blind 1/2/4/8-step rollout testing only after
   it is noninferior to its masked counterpart on the separately
   preregistered direct branch gate. Structural re-entry alone is insufficient.
8. Scaling the existing observational pool is permitted only if a treatment
   improves direct branch truth across training seeds by a practical threshold
   frozen before branch results are opened. Proxy improvement alone is
   insufficient.
9. If neither action grounding nor dense targets improves direct branch truth,
   stop tuning these observational mechanisms. The next information-changing
   intervention is matched branch training data; after two adequately powered
   non-improving mechanisms, compare a conventional state-space dynamics model
   and a task-coupled Dreamer-style baseline rather than continuing JEPA proxy
   tuning.

The counterfactual calibration/pilot has a separate lifecycle. A calibration
failure does not turn this factual screen into causal evidence, and this
document does not authorize a counterfactual retry or replacement.

## 10. Limitations that must accompany the result

- The pack contains requested/factual action IDs but no executed-command,
  clipping, or controller-state fields. Action association can reflect the
  behavior policy and cannot establish executed dynamics.
- Existing support is substantial but confounded: a local state-only kNN
  classifier achieved balanced accuracy about `0.556`. Scene-disjoint
  validation does not remove this policy signal.
- HOLD often has meaningful motion, so an RGB-only current latent omits
  proprioceptive and belief state needed to disambiguate inertia and contacts.
- The frozen RGB substrate is unchanged. This screen does not test a modern
  proprioceptive encoder, AdaLN conditioning, or a separately trained dense
  DINO/V-JEPA substrate.
- The delta decoder is trained on true factual transitions from the same pack.
  Its anchor establishes decodable action association, not causation or
  generalization to untaken actions.
- Freezing the decoder removes decoder/predictor co-adaptation but still lets
  the predictor exploit any fixed decoder shortcut.
- Full-grid outputs are structurally re-entrant, but no multi-step objective,
  uncertainty model, scheduled sampling, or rollout stability constraint is
  trained here.
- The runner's ordinary terminal metrics retain the nominal nine-action
  hardest-wrong assumption. Physically equivalent or clipped actions can make
  that proxy wrong; the causal evaluator must use outcome-equivalence classes.
- Three training seeds share one 150-scene validation panel that has already
  informed earlier repository work. This is development estimation, not fresh
  confirmation.
- The 16,000-row pack is about 0.9% of the roughly 1.81 million available H6
  candidates. A null result is a result for this fixed screen, not proof that
  the larger existing pool lacks learnable signal.
- Snapshot compatibility currently supports the one-step registered route.
  A full-grid blind-rollout consumer requires separate source review and a
  frozen rollout contract.
- No result from this run alone establishes representation health in physical
  coordinates, counterfactual action binding, rollout advantage over
  persistence, planning geometry, or same-sensor closed-loop navigation gain.

## 11. Pre-execution verification already completed

The focused CPU/source suite was run under the bound ROCm interpreter with
plugin autoload disabled:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .generated/venvs/world_model_rocm_7_2_1_v1/bin/python -m pytest -q lewm/tests/test_go2_world_model_progression_v1.py lewm/tests/test_counterfactual_current_arm_snapshot_loader_v1.py lewm/tests/test_analyze_go2_world_model_progression_v1.py
19 passed in 0.80s
```

Earlier strict-determinism smoke repeats produced bit-identical science
payloads and terminal state hashes. Those smokes are engineering evidence only
because the source changed before this final freeze. No 700-update comparison
result exists at preregistration time, and no GPU work was performed while
writing this document.

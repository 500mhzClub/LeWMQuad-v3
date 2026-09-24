# Go2 RGB fixed-teacher factorized conditional-increment trajectory H4 JEPA V1 result — 2026-07-28

## Terminal outcome

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_JEPA_V1`.
- The sole authorized probe completed normally at the exact cap: 1,000
  optimizer updates, 16,000 ordered training presentations, and 10,240
  validation presentations in `722.1597084140521` active GPU seconds.
- Update 750 / presentation 12,000 was selected by the frozen rule: minimum
  validation 50/50 joint-plus-marginal normalized energy among eligible
  noncollapsed trained observations.
- The result passed 28 of 32 gates and failed exactly four. This is a
  scientific STOP, not an execution, receipt, or custody failure.
- This mechanism is closed without retry, resume, longer training, second
  seed, threshold change, checkpoint opening, or post-hoc checkpoint choice.
- The result is bounded development evidence for an RGB-only learned JEPA
  world model. It does not authorize navigation, held-out/sealed access,
  promotion, production, or deployment.

## What was tested

- The fixed accepted N320 RGB encoder remained the sole target and never
  moved. The online encoder, temporal state, factorized transition, categorical
  action tower, particles, and prediction head were trained jointly in one
  summed loss and one backward pass.
- The transition was exactly:

  ```text
  v_hat = W0(d + B(z,h,D(d)) * (1+tanh(D(d))) * c_a)
  c_a   = A(E[a]) - mean_j A(E[j])
  ```

  followed by the inherited normalized local step. `W0` was shared,
  bias-free, and zero-initialized. No learned state-only successor bypass,
  separately trained predictor, numeric command semantics, or privileged
  geometry was present.
- Inputs were RGB `e0:e2`, requested past actions `p0:p1`, and proposed future
  actions `p2:p5`; fixed-teacher targets were future RGB `e3:e6`. The exact
  causal V2 requested-command schedules, objective, optimizer, evaluator,
  selection rule, 32 gates, and cap were reused.
- Training remained exactly half all-six factual local-innovation energy,
  half cumulative open-loop future-trajectory energy, plus weight-one
  online-to-fixed-teacher history alignment. Wrong-action, all-HOLD,
  reversed/reset-history, persistence, and collapse branches were evaluation-
  only.

## Selected aggregate result

| Measure | Update 750 value |
|---|---:|
| Combined normalized energy score | 0.777215502687 |
| Joint-trajectory normalized energy score | 0.774099870794 |
| Future `p2:p5` local combined score | 0.839000171304 |
| Pre-observation `p0:p1` local-prior combined score | 0.797648348812 |
| Pre-observation `p0:p1` persistence gap | +0.202351651351 |
| Pre-observation gap bootstrap lower 95% | +0.183523396333 |
| H4 marginal normalized energy score | 0.832867278058 |
| H4 persistence gap | +0.167132721209 |
| H4 persistence bootstrap lower 95% | +0.134181582874 |
| H4 cyclic-action gap | +0.108100214861 |
| H4 cyclic-action bootstrap lower 95% | +0.094631024632 |
| H4 ordered-history gap | -0.012313893646 |
| H4 ordered-history bootstrap lower 95% | -0.015885900101 |
| H4 all-HOLD gap | +0.005950964234 |
| Combined distribution-value gap | +0.251531520782 |
| Combined distribution-value bootstrap lower 95% | +0.247989161867 |
| H4 normalized pairwise spread | 1.400538124091 |
| H4 best-atom normalized squared error | 2.196066712563 |
| H4 centroid normalized squared error | 1.961557263519 |

- The factorization fixed the predecessor's central action-conditioning
  failure. The selected H4 action gap is above the `0.05` gate by more than a
  factor of two, its bootstrap lower bound is strongly positive, and all eight
  families are positive.
- Generic prediction remains useful and noncollapsed. Combined, joint, all
  four future marginal scores, and the factual `p0:p1` prior beat persistence.
  Persistence and factual-prior gaps are positive in all eight families.
- The four-particle distribution remains value-adding in all eight families,
  with target and online effective-rank ratios `0.176579435666` and
  `0.200267056624` and zero near-zero-variance fractions.
- The distribution is not yet planner-quality. H4 particle spread is large,
  and both best-atom and centroid squared errors exceed the persistence-
  normalized value `1.0` even though the proper energy score beats persistence.
- Correct ordered history is still counterproductive: its gap is negative,
  its lower bound is negative, and zero of eight families are positive. The
  predictor learned requested-action identity without learning a reliable
  temporal predictive state.
- The aggregate all-HOLD control is narrowly positive, but only three of eight
  families are positive. HOLD/inertial semantics are therefore not robust.

## Learning trajectory

| Update | Presentations | Combined | H4 score | H4 persistence | H4 action | H4 history | H4 HOLD | Action-positive families | History-positive families | HOLD-positive families |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 250 | 4,000 | 0.813056 | 0.882196 | +0.117804 | +0.141588 | -0.011740 | +0.097281 | 8 | 0 | 8 |
| 500 | 8,000 | 0.799478 | 0.863063 | +0.136937 | +0.121737 | -0.013636 | +0.033729 | 8 | 0 | 7 |
| 750 selected | 12,000 | 0.777216 | 0.832867 | +0.167133 | +0.108100 | -0.012314 | +0.005951 | 8 | 0 | 3 |
| 1,000 | 16,000 | 0.780334 | 0.831205 | +0.168795 | +0.108997 | -0.024716 | +0.017360 | 8 | 0 | 4 |

- Action conditioning appeared decisively by update 250 and remained broad
  and well above threshold through the cap. This is a robust mechanism result,
  not a single late fluctuation.
- Generic prediction improved through update 750 and then narrowly regressed,
  correctly making update 750 the selected checkpoint.
- HOLD discrimination was broad early but eroded as generic prediction
  improved. Ordered-history value was negative at every trained observation
  and worsened sharply at update 1,000.
- Update 250 cannot be selected post hoc: the registered selection rule uses
  the best combined score, and it would still fail all three history gates.

## Selected per-family findings

| Family | H4 score | H4 persistence | H4 action | H4 history | H4 HOLD | `p0:p1` persistence |
|---|---:|---:|---:|---:|---:|---:|
| `large_enclosed_maze` | 0.793108 | +0.206892 | +0.076565 | -0.005397 | +0.005757 | +0.214957 |
| `local_composite_motifs` | 0.843712 | +0.156288 | +0.040713 | -0.019402 | -0.002983 | +0.170108 |
| `loop_alias_stress` | 0.857077 | +0.142923 | +0.081894 | -0.013672 | -0.017874 | +0.192704 |
| `medium_enclosed_maze` | 0.870839 | +0.129161 | +0.092985 | -0.008755 | -0.002977 | +0.216364 |
| `open_obstacle_field` | 0.797785 | +0.202215 | +0.214220 | -0.016299 | +0.050778 | +0.231100 |
| `rough_local_dynamics` | 0.873215 | +0.126785 | +0.238692 | -0.015533 | +0.037781 | +0.200653 |
| `small_enclosed_maze` | 0.807717 | +0.192283 | +0.033831 | -0.011921 | -0.013867 | +0.186204 |
| `visual_sensor_stress` | 0.819485 | +0.180515 | +0.085902 | -0.007532 | -0.009007 | +0.206724 |

## Exact gate failures

The four failed gates were:

1. `h4_history_gap_at_least_point03`
2. `h4_history_gap_bootstrap_lower_positive`
3. `history_positive_in_six_families`
4. `hold_positive_in_six_families`

- All 28 other gates passed, including exact completion, fixed-target
  identity, noncollapse, every prediction/persistence gate, all particle-value
  gates, every factual `p0:p1` gate, every cyclic-action gate, overall positive
  H4 HOLD, and all per-family floors.
- The substantive remaining failure is temporal state: correct ordered context
  is worse than the better of reversed or reset context in every family. The
  HOLD breadth failure is consistent with the same missing motion/state
  estimate rather than missing categorical action sensitivity.

## Comparison with causal shared-transition V2

| Measure | V2 selected | Factorized V1 selected | Change |
|---|---:|---:|---:|
| Combined score | 0.742207 | 0.777216 | +0.035009 |
| Joint score | 0.737035 | 0.774100 | +0.037065 |
| Future local score | 0.806352 | 0.839000 | +0.032648 |
| H4 score | 0.782289 | 0.832867 | +0.050578 |
| H4 persistence gap | +0.217711 | +0.167133 | -0.050578 |
| H4 action gap | +0.000616 | +0.108100 | +0.107484 |
| H4 action lower 95% | +0.000227 | +0.094631 | +0.094404 |
| H4 history gap | -0.028849 | -0.012314 | +0.016535 |
| H4 HOLD gap | -0.000962 | +0.005951 | +0.006913 |
| Combined distribution value | +0.244954 | +0.251532 | +0.006578 |
| Failed gates | 6 | 4 | -2 |

- The new action gap is roughly 175 times V2's magnitude and is positive in
  all families. The action mechanism therefore worked as intended.
- That gain trades some raw prediction accuracy for controllability. This is
  acceptable evidence for the factorization, but the remaining negative
  history result prevents qualification as a useful predictive state.
- The exact stopped mechanism should not be trained longer or tuned. Its curve
  shows action sensitivity is already stable while history remains negative;
  more presentations would scale the wrong state pathway.

## Receipt and custody audit

| Receipt | Bytes | File SHA-256 | Content SHA-256 |
|---|---:|---|---|
| `reservation.json` | 8,901 | `941cb3d00786ba44ba015344119ea6eb568fec7a1ef5c53aea0b3fd2bc4dce8c` | `f00942012cabbfb4c7e861c3f4ea63e9fd422e3b48f930a0bf2ce456264e3a7f` |
| `metrics.json` | 60,230 | `90a4a1250414fff8648b16d6483fe25a0a5611180151427b699304a3756e9adf` | `4da11d3793f117ebb77bdbd96207b984d9367b76652d3e04cf0309b4b25bde2d` |
| `artifact.json` | 7,417 | `2863f78ebcb859beddca63a0460b796dbeec2ae3b13b73df4ac27e577cf2483c` | `cf6ee25f422a3531b266962c1af28a518e6ea6b7f235d77e3aa27de72e2c7dd6` |
| `access.json` | 1,572 | `9b1876f75adfc8274f9341e83819e5c9fbd87eba41e75432a47646953efe37af` | `48c6d6a6ed6eec3209838809d9f392942679c86dcae14e58ca98c79484e80e04` |
| `result.json` | 3,138 | `f3e68404effe4d0790062ad86d5c5d5cb1426345f96bcc7885568ea87ae56c81` | `2b9dd8c9e1339ee7c902a6faa2008733ebb432fd93bf73454e2cb6d06487cc03` |
| `completed.json` | 1,888 | `29acda7bf94eb304323e72491c4e9fc39c1ea7c72a0bc777f8820121405ce0ad` | `688a68b554261adb001c107d7c43d27fbcf6767a48e775f3db638acee42435bb` |

- Root and two independent receipt-only audits returned **CLEAR**. All six
  files are canonical, finite, self-bound JSON; `completed.json` exactly
  cross-binds the other five. All 32 gates and the selected update were
  independently recomputed.
- Access is complete: 1,000 optimizer updates, 16,000 train presentations,
  10,240 validation presentations, and exactly 183,680 RGB open attempts and
  successes. Active GPU time remained below the 5,400-second cap.
- The fixed target's initial and final state hashes are identical, with zero
  EMA updates. Exactly one accepted N320 initialization was opened; only 78
  `encoder.*` tensors were copied and zero non-encoder tensors were copied.
- Test/held-out, sealed, navigation, labels, arbitrary initialization,
  predecessor predictor checkpoints, retry, resume, wrong-action training,
  and auxiliary training controls all remained exactly zero.
- Audit read only the six canonical JSON receipts. No runtime checkpoint was
  listed, statted, hashed, opened, or reused.

## Scientific conclusion and ordered next step

- The repo has now crossed an important boundary: the RGB encoder plus one
  jointly trained JEPA graph can learn a broad, useful requested-action signal
  without action-ranking supervision, geometry, or a separately trained
  predictor.
- It has not yet learned a useful temporal predictive state. The current
  hidden path is redundant or harmful relative to current RGB plus the explicit
  incoming increment, so correct ordered context can be ignored while action
  identity still works.
- The next materially different candidate should retain centered categorical
  action conditioning but replace the stopped hidden/increment carrier with a
  **causal predictive-state bottleneck JEPA**. Ordered `e0,p0,e1,p1,e2` must
  form one recurrent state, and future prediction may receive only that state
  plus the requested action—no raw current-`z` or explicit `d`/inertial bypass.
- Keep the accepted fixed N320 target, proper trajectory objective, exact V2
  causal schedules/evaluator, and a fresh bounded falsification. Do not add a
  history hinge, relax the gates, reuse this run's checkpoint, or retry this
  exact factorization.

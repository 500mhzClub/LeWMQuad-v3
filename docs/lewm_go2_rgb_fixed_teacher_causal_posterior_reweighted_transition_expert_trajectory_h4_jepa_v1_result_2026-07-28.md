# Go2 RGB fixed-teacher causal posterior-reweighted transition-expert trajectory H4 JEPA V1 result — 2026-07-28

## Terminal outcome

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_CAUSAL_POSTERIOR_REWEIGHTED_TRANSITION_EXPERT_TRAJECTORY_H4_JEPA_V1`.
- The sole reserved attempt completed normally at the exact cap: 1,000
  optimizer updates, 16,000 ordered training presentations, 10,240 validation
  presentations, and `714.138882839994` active-GPU seconds.
- Update 1,000 / presentation 16,000 was selected by the frozen minimum
  posterior-weighted combined-energy rule. The result passed 26 of 32 gates.
- This is a scientific STOP, not an execution, receipt, source-binding,
  target-integrity, collapse, generic-prediction, persistence, action, or
  custody failure. The exact fixed posterior-reweighted K4 mechanism is closed
  without retry, resume, repair, expert-count/likelihood/epsilon variant,
  second seed, extension, or checkpoint opening.
- The existing navigation, G2--G8, held-out, and sealed authorities remain
  unopened and unchanged.

## What was tested

- The complete causal state was four normalized feature-lattice contents `q`
  and four strictly positive probabilities `w` summing to one. `q0` was four
  copies of online `z0`; `w0` was exactly uniform. There was no continuous
  recurrent state, writable matrix, momentum, incoming-increment carrier, or
  other hidden statistic.
- Four centered learned mode embeddings and one learned spatial table entered
  one inherited action-free belief context. One complete-table-centered
  categorical action tower and one shared bias-free, zero-initialized head
  produced all six action-conditioned priors.
- On `p0` and `p1`, each prior was emitted before its destination was visible.
  Its full-lattice squared error produced the fixed likelihood
  `exp(-d_k/(mean(d)+1e-6))`; prior mass times likelihood was normalized, and
  only then was factual `q` assimilated. Exactly two evidence updates occurred.
- After `e2`, the belief held only packed `(q2,w2)`. `w2` remained bitwise
  fixed while the same transition recursed over `p2:p5`; probabilities could
  change distribution mass but could not move expert content or increments.
- Future marginal, coherent joint, centroid, spread, real, wrong-action,
  all-HOLD, persistence, reverse, and reset metrics used the preregistered
  branch-specific posterior masses. The `p0:p1` diagnostic and all-six local
  training term remained equal-mass.
- The online encoder, modes/context, action path, and shared head trained
  jointly in one JEPA graph and one summed backward. The accepted N320 encoder
  remained the fixed no-grad target. No predictor, posterior, or system
  identifier was trained separately.

## Selected aggregate result

| Measure | Update 1,000 value |
|---|---:|
| Combined normalized energy score | 0.745903549914 |
| Joint-trajectory normalized energy score | 0.741525077445 |
| Future `p2:p5` local combined score | 0.816715436954 |
| Pre-observation `p0:p1` local-prior combined score | 0.773748001624 |
| Pre-observation `p0:p1` persistence gap | +0.226251998065 |
| Pre-observation gap bootstrap lower 95% | +0.210217168162 |
| H4 marginal normalized energy score | 0.785348837387 |
| H4 persistence gap | +0.214651163692 |
| H4 persistence bootstrap lower 95% | +0.187483302033 |
| H4 cyclic-action gap | +0.168639283958 |
| H4 cyclic-action bootstrap lower 95% | +0.148697918391 |
| H4 ordered-history gap | -0.007974303816 |
| H4 ordered-history bootstrap lower 95% | -0.010091289879 |
| H4 all-HOLD gap | -0.009095404244 |
| Combined distribution-value gap | +0.242693788126 |
| Combined distribution-value bootstrap lower 95% | +0.239548147277 |
| H4 normalized pairwise spread | 1.237510813305 |
| H4 best-atom normalized squared error | 2.012896610122 |
| H4 centroid normalized squared error | 1.513293575121 |

- Generic prediction, persistence, distribution value, action conditioning,
  noncollapse, fixed-target identity, and the pre-observation local-prior
  diagnostic all passed decisively. Action was positive in every family and
  substantially exceeded its threshold.
- Correct ordered history remained harmful in every family. Its aggregate H4
  gap and bootstrap lower bound were negative, and zero of eight families were
  positive. Fixed Bayesian-style expert mass therefore did not become useful
  temporal state.
- HOLD was negative in aggregate, positive in only two families, and reached
  `-0.054756648217` in `loop_alias_stress`. This was a material regression, not
  a near-threshold miss.
- Distribution readout remained valuable, but the best support atom's H4
  point error was `2.012897`, well above persistence-normalized parity. The
  posterior improved the weighted centroid relative to the immediate
  predecessor but did not identify a reliably better trajectory expert.

## Learning trajectory

| Update | Presentations | Combined | H4 score | H4 action | H4 history | H4 HOLD | Action-positive families | History-positive families | HOLD-positive families |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 250 | 4,000 | 0.789858 | 0.848352 | +0.124162 | -0.005190 | +0.108278 | 8 | 0 | 8 |
| 500 | 8,000 | 0.769146 | 0.821249 | +0.160774 | -0.008992 | +0.063257 | 8 | 0 | 8 |
| 750 | 12,000 | 0.745980 | 0.773925 | +0.133492 | -0.009165 | +0.014268 | 8 | 0 | 7 |
| 1,000 selected | 16,000 | 0.745904 | 0.785349 | +0.168639 | -0.007974 | -0.009095 | 8 | 0 | 2 |

- Combined prediction improved through the cap, and the frozen selector chose
  update 1,000 by only `0.000077` over update 750. H4 itself was better at
  update 750, but checkpoint selection was correctly not changed post hoc.
- Action support appeared by update 250 and remained positive in all eight
  families. Correct history was negative in all eight families at every
  trained observation; more optimization did not repair its sign.
- HOLD breadth fell from eight families at updates 250/500 to seven at 750 and
  two at 1,000. Extending or resuming this consumed run would optimize the
  wrong tradeoff and is scientifically unjustified.

## Selected per-family findings

| Family | H4 score | H4 persistence | H4 action | H4 history | H4 HOLD | `p0:p1` persistence |
|---|---:|---:|---:|---:|---:|---:|
| `large_enclosed_maze` | 0.762923 | +0.237077 | +0.141114 | -0.008188 | -0.002385 | +0.234857 |
| `local_composite_motifs` | 0.800083 | +0.199917 | +0.090596 | -0.016355 | -0.009631 | +0.186928 |
| `loop_alias_stress` | 0.838629 | +0.161371 | +0.103083 | -0.004628 | -0.054757 | +0.208995 |
| `medium_enclosed_maze` | 0.824505 | +0.175495 | +0.090343 | -0.012213 | -0.031682 | +0.229527 |
| `open_obstacle_field` | 0.729694 | +0.270306 | +0.334184 | -0.002514 | +0.028338 | +0.271095 |
| `rough_local_dynamics` | 0.764392 | +0.235608 | +0.368608 | -0.004713 | +0.030080 | +0.248801 |
| `small_enclosed_maze` | 0.772804 | +0.227196 | +0.084644 | -0.007254 | -0.018422 | +0.207827 |
| `visual_sensor_stress` | 0.789760 | +0.210240 | +0.136542 | -0.007930 | -0.014304 | +0.221987 |

## Exact gate failures

The six failed gates were:

1. `h4_history_gap_at_least_point03`
2. `h4_history_gap_bootstrap_lower_positive`
3. `h4_hold_gap_positive`
4. `history_positive_in_six_families`
5. `hold_positive_in_six_families`
6. `no_family_hold_gap_below_minus_point02`

- All 26 other gates passed, including exact completion, update-zero
  persistence, every fixed-target/noncollapse gate, every generic prediction,
  persistence, action, distribution-value, and `p0:p1` local-prior gate.
- The decisive failure is still ordered temporal state. The HOLD failures
  independently show that posterior reweighting did not preserve robust
  no-motion behavior across scene families.

## Comparison with the immediate predecessor

- Versus action-attributed causal system-ID V1, combined score worsened from
  `0.735384` to `0.745904`, joint from `0.731810` to `0.741525`, H4 from
  `0.772121` to `0.785349`, and best-atom H4 squared error from `1.151557` to
  `2.012897` (lower is better).
- Action improved strongly from `+0.087621` to `+0.168639`, with positive
  eight-family breadth and a much larger bootstrap lower bound.
- History became less negative (`-0.013267` to `-0.007974`) but remained
  harmful in every family and nowhere approached the registered `+0.03`
  threshold. This is not evidence that a longer run would cross the gate.
- HOLD regressed from `+0.010821` and four positive families to `-0.009095`
  and two. The worst-family floor fell from `-0.023626` to `-0.054757`.
- The `p0:p1` persistence diagnostic stayed essentially unchanged
  (`+0.229353` to `+0.226252`). The model could predict observed transitions,
  but the resulting error-derived posterior did not help the future sequence.
- The weighted centroid point error improved from `1.986106` to `1.513294`,
  while the best atom deteriorated sharply. Posterior mass produced a better
  mixture readout without learning a correct persistent response regime.

## Receipt and custody audit

| Receipt | Bytes | File SHA-256 | Content SHA-256 |
|---|---:|---|---|
| `reservation.json` | 11,230 | `a35bafa82646b2ae7ed08668da4df76e70f6ba133c9d007c1bf04c2227f5042a` | `cebff172ba35e52653cb295b5c3645ef6cd2497164295aa20f2dfcadd41dad2b` |
| `metrics.json` | 60,528 | `9cc47c3bb11121ee6f3dba0ae17d038dd79259ed88f38f62e9363d5d002ef6ff` | `ecc0d084d34d6b4c68126b103db98dbb8628455cf425c988f7e683f0e49470f5` |
| `artifact.json` | 9,685 | `c171b90fb82339df6e9089a9f40f9cd22ff1a04c9328426deaa061d47a3526f8` | `7ecefb394064a0eca258fb870bf520bfc6ccd04d582136cf0b1d94f7441892ca` |
| `access.json` | 1,585 | `47cd93dc7751c065129b91239a50aaa0d7a4a0ec4137c57f780792dac6cab560` | `ef226efba8aed63809c1fccb6a32d756aaac7a51b6c7c49d37e0ca238208741a` |
| `result.json` | 3,272 | `17dd942bb4324b6fa624ee81e37b6bbe51ac86b10213e9e8b2119fd7c14c9a83` | `1f8fd252d1037c0848e0032ef0213043ec1e807628733bea06c287a6e8374844` |
| `completed.json` | 1,915 | `9b4feaf482f3bd3a5add6ee300b80f55d3ce8011b0a6b2834924149e7b91734b` | `26f79bd11cae1d0fb06edb3d390a98e07a3ddd5d8a1302612c40a504775fac27` |

- Root and independent receipt-only audits returned CLEAR. All six files are
  strict, finite, canonical, self-bound JSON; `completed.json` exactly binds
  and cross-binds the other five. Its file SHA-256 matches the terminal stdout
  hash. The independent audit reconciled every cap and access count, confirmed
  that the six failed gates exactly equal the six false booleans, and
  independently recomputed all 30 gates whose numeric thresholds are present
  in the receipts.
- Access is complete: 1,000 optimizer updates, 16,000 train presentations,
  10,240 validation presentations, and exactly 183,680 RGB open attempts and
  successes. Active GPU time remained below the registered cap.
- The fixed target's initial and final state SHA-256 are both
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`;
  target EMA updates are zero.
- Test/held-out, sealed, navigation, labels, arbitrary initialization,
  predecessor predictor checkpoints, retry/resume, wrong-action training, and
  auxiliary training controls all remained exactly zero.
- Audit read only the six exact canonical JSON receipts. Runtime checkpoints
  and traces remain unopened write-only artifacts; none was discovered,
  listed, statted, hashed, opened, copied, or reused.

## Scientific conclusion and next boundary

- Retain the centered categorical-action pathway, generic trajectory
  distribution, fixed N320 teacher, and one jointly trained JEPA graph. These
  continue to provide broad action sensitivity, persistence improvement, and
  distribution value.
- Close this exact causal posterior-regime mechanism. The two observed
  full-lattice prior errors did not select a future-useful expert, and the
  mechanism may not be retried with a different expert count, likelihood,
  epsilon, temperature, seed, duration, score coefficient, or checkpoint.
- Together with recurrent, dense-history, factorized-increment, momentum,
  writable system-ID, and posterior-regime failures, another small temporal
  carrier or nearby gating variant is not justified. The next falsification
  must change the learned predictive mechanism materially rather than refine
  posterior bookkeeping.
- The canonical receipts do not contain realized posterior masses, entropy,
  maximum weight, effective-expert count, or calibration statistics. They
  prove that posterior-weighted scoring was used, but cannot support a claim
  that the posterior ever concentrated on a correct response regime. No
  inaccessible checkpoint may be opened to recover that unregistered
  diagnostic after the fact.
- This STOP grants no checkpoint access, navigation, G2, held-out/sealed
  access, promotion, production, or deployment authority.

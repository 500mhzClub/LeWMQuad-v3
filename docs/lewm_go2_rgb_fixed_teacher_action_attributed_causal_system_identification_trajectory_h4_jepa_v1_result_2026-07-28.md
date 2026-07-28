# Go2 RGB fixed-teacher action-attributed causal system-identification trajectory H4 JEPA V1 result — 2026-07-28

## Terminal outcome

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_IDENTIFICATION_TRAJECTORY_H4_JEPA_V1`.
- The sole reserved attempt completed normally at the exact cap: 1,000
  optimizer updates, 16,000 ordered training presentations, 10,240 validation
  presentations, and `709.8668034310103` active-GPU seconds.
- Update 1,000 / presentation 16,000 was selected by the frozen minimum
  combined-energy rule. The result passed 27 of 32 gates.
- This is a scientific STOP, not an execution, receipt, target-integrity,
  collapse, action, prediction, or custody failure. The exact mechanism is
  closed without retry, resume, extension, second seed, checkpoint opening,
  post-hoc checkpoint choice, or nearby matrix/writer repair.
- The existing held-out, sealed, navigation, and G2--G8 authorities remain
  unopened and unchanged.

## What was tested

- Four equal-mass causal states each contained normalized feature-lattice
  content `q` and one exactly `16 x 16` nonspatial action-response matrix `M`.
  `M` had no patch axis, nine-action slots, physical units, pose, geometry, or
  hand-written dynamics semantics.
- One shared prior was used on all six edges. Its only learned increment route
  multiplied the inherited action-free spatial context, the bounded memory
  read `1+tanh(P_M(vec(M)))`, and the complete-table-centered current action
  code before one shared bias-free zero-initialized head.
- On `p0` and `p1`, each prior was emitted before the destination RGB became
  available. The newly observed prior error and the centered key of the exact
  requested action that caused it made one fixed rank-one outer-product write.
  The factual destination then replaced `q`. Exactly two writes occurred, so
  `rank(M2) <= 2`.
- After `e2`, the belief contained only packed `(q2,M2)`. `M2` remained
  bitwise fixed while the same prior recursed over `p2:p5`; no future RGB, raw
  incoming increment, momentum, anchor, dense history, or generic memory
  update entered prediction.
- The online encoder, modes, response/key/memory projections, spatial context,
  action tower, and increment head trained jointly in one JEPA graph and one
  summed backward. The accepted N320 encoder remained the fixed no-grad target.
- The unchanged objective was weight-one online/fixed-teacher history
  alignment, half all-six local proper energy score, and half cumulative H4
  proper energy score. Every action, history, HOLD, persistence, and collapse
  control remained evaluation-only.

## Selected aggregate result

| Measure | Update 1,000 value |
|---|---:|
| Combined normalized energy score | 0.735384309778 |
| Joint-trajectory normalized energy score | 0.731810257117 |
| Future `p2:p5` local combined score | 0.810477390710 |
| Pre-observation `p0:p1` local-prior combined score | 0.770647336820 |
| Pre-observation `p0:p1` persistence gap | +0.229352663236 |
| Pre-observation gap bootstrap lower 95% | +0.213168272976 |
| H4 marginal normalized energy score | 0.772120826376 |
| H4 persistence gap | +0.227879172096 |
| H4 persistence bootstrap lower 95% | +0.202013057104 |
| H4 cyclic-action gap | +0.087620550889 |
| H4 cyclic-action bootstrap lower 95% | +0.078621517813 |
| H4 ordered-history gap | -0.013266784050 |
| H4 ordered-history bootstrap lower 95% | -0.015510476878 |
| H4 all-HOLD gap | +0.010821107055 |
| Combined distribution-value gap | +0.255714644476 |
| Combined distribution-value bootstrap lower 95% | +0.251314123341 |
| H4 normalized pairwise spread | 1.234437174617 |
| H4 best-atom normalized squared error | 1.151557217095 |
| H4 centroid normalized squared error | 1.986105853975 |

- Generic prediction was strong and noncollapsed. Combined, joint, every
  future marginal horizon, and both observed priors beat persistence. H4 and
  observed-prior persistence gaps had positive bootstrap bounds and eight-
  family breadth.
- Requested-action sensitivity remained decisive: the aggregate H4 action gap
  exceeded `0.05`, its lower bound was positive, and all eight families were
  positive.
- The four-atom distribution added value in all eight families. The best atom
  approached the persistence-normalized point-error baseline much more closely
  than either predecessor, although it remained above `1.0` and is not a
  planner qualification.
- Correct ordered history was counterproductive in every family. Its aggregate
  H4 gap and bootstrap lower bound were negative, and zero of eight families
  were positive. The fixed error-by-action matrix therefore improved the
  overall predictor without becoming useful temporal state.
- Aggregate HOLD value was positive, but only four families were positive.
  `medium_enclosed_maze` reached `-0.023626303512`, violating the registered
  `-0.02` family floor.

## Learning trajectory

| Update | Presentations | Combined | H4 score | H4 action | H4 history | H4 HOLD | Action-positive families | History-positive families | HOLD-positive families |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 250 | 4,000 | 0.776710 | 0.838611 | +0.097474 | -0.022621 | +0.048941 | 8 | 0 | 7 |
| 500 | 8,000 | 0.766529 | 0.819087 | +0.122012 | -0.013708 | +0.016307 | 8 | 0 | 7 |
| 750 | 12,000 | 0.741171 | 0.776256 | +0.086579 | -0.017138 | +0.000498 | 8 | 0 | 4 |
| 1,000 selected | 16,000 | 0.735384 | 0.772121 | +0.087621 | -0.013267 | +0.010821 | 8 | 0 | 4 |

- Combined prediction improved at every trained observation, and H4 prediction
  improved overall through the cap. The selection of update 1,000 is exact.
- Action support appeared by update 250 and stayed positive in every family.
  Ordered history was negative in every family at every trained observation;
  its sign did not improve with the main score.
- HOLD breadth eroded from seven families at updates 250/500 to four at
  updates 750/1000. Extending the consumed run would therefore optimize a
  useful generic/action predictor while leaving the decisive state failure
  unresolved.

## Selected per-family findings

| Family | H4 score | H4 persistence | H4 action | H4 history | H4 HOLD | `p0:p1` persistence |
|---|---:|---:|---:|---:|---:|---:|
| `large_enclosed_maze` | 0.750757 | +0.249243 | +0.095649 | -0.014309 | +0.009972 | +0.238508 |
| `local_composite_motifs` | 0.782200 | +0.217800 | +0.051493 | -0.013212 | +0.003974 | +0.193739 |
| `loop_alias_stress` | 0.811966 | +0.188034 | +0.060081 | -0.012910 | -0.018304 | +0.212296 |
| `medium_enclosed_maze` | 0.830563 | +0.169437 | +0.047017 | -0.019587 | -0.023626 | +0.232226 |
| `open_obstacle_field` | 0.721052 | +0.278948 | +0.143391 | -0.007093 | +0.057372 | +0.266312 |
| `rough_local_dynamics` | 0.753347 | +0.246653 | +0.173039 | -0.009287 | +0.063250 | +0.245686 |
| `small_enclosed_maze` | 0.757831 | +0.242169 | +0.055599 | -0.014114 | -0.004200 | +0.214094 |
| `visual_sensor_stress` | 0.769251 | +0.230749 | +0.074695 | -0.015623 | -0.001868 | +0.231961 |

## Exact gate failures

The five failed gates were:

1. `h4_history_gap_at_least_point03`
2. `h4_history_gap_bootstrap_lower_positive`
3. `history_positive_in_six_families`
4. `hold_positive_in_six_families`
5. `no_family_hold_gap_below_minus_point02`

- All 27 other gates passed, including exact completion, update-zero
  persistence, fixed-target identity, noncollapse, all generic prediction and
  persistence gates, all distribution-value gates, all observed-prior gates,
  and every requested-action gate.
- The primary blocker remains temporal state, not feature collapse or action
  conditioning. The two HOLD failures are consistent with the same inability
  to infer a robust sequence-specific response regime.

## Comparison with the immediate predecessors

- Versus factorized conditional-increment V1, combined and joint scores
  improved by `0.041831` and `0.042290` (lower is better), future-local by
  `0.028523`, and H4 by `0.060746`. Best-atom H4 error fell from `2.196067`
  to `1.151557`.
- Factorized action was stronger (`+0.108100` versus `+0.087621`), while the
  current action result still passed every action gate. Ordered history was
  effectively unchanged and harmful in every family (`-0.012314` versus
  `-0.013267`). HOLD breadth moved only from three to four families.
- Versus latent-momentum V1, combined and joint improved by `0.023022` and
  `0.023273`, H4 improved by `0.072012`, and action improved from `+0.067420`
  to `+0.087621`. History became substantially less harmful
  (`-0.039453` to `-0.013267`) but remained negative in all eight families.
- Latent momentum retained much stronger HOLD value (`+0.111125`, eight
  families). The system-ID statistic traded that inertial robustness for
  substantially better generic, action, and point-atom prediction.
- The current combined score remains slightly above the earlier equal-mass
  trajectory-distribution result (`0.735384` versus `0.725748`), but current
  action value is about eight times larger and its best-atom point error is
  much lower (`1.151557` versus `1.480741`). No existing result has yet joined
  broad prediction, action, ordered history, and HOLD in one passing state.

## Receipt and custody audit

| Receipt | Bytes | File SHA-256 | Content SHA-256 |
|---|---:|---|---|
| `reservation.json` | 11,544 | `782794bf29317e3106e0945a85b908a1df46875222e65e62f1e103152633a90a` | `11b6647ceb97ace226fc37fb48c60a109ef6ec36234fa9ced7a6567dcb86c25d` |
| `metrics.json` | 60,399 | `9903dcce6a0b3aca844bc45dd8fd2251256998219d683e60058e55116209564c` | `be74ed8f72059ba190a3be7954d7b8446a24f55b6aaf3aa7b246c34c90c0137e` |
| `artifact.json` | 9,042 | `14b4e717e1c1ed20a14be3e86e7a8e05f5a26861b85027c4e38ca939d59e1541` | `8ee7fffec3bac9c45eb9b5c9fecce4be87bb2c0d01ff94c16d543ce9b0895a42` |
| `access.json` | 1,586 | `709e241aeb02720b1f143f4e6ca57281aa19b37aa34b7082b36144d5dee5484d` | `d6bb64fe65308ff039827e8571288bc70c436827b6c17307ec3529c21d34ca0d` |
| `result.json` | 3,257 | `37c5956fbac69d49cf6cd84875f8a7496e5848edfb55323c5a5130e66411c6bc` | `5899b6aa729e2cb261d49409175b8289e3421fbebcb1b99e157fd8d917b99214` |
| `completed.json` | 1,917 | `e3f2b94c09442412a943c43daef900d4ae36339a9f8c912e9f15e00ceff8b98f` | `09be12df31a980e173a2c69e1187ccfa3a353697d54c0b928041fd8cbed13359` |

- Root and independent receipt-only audits returned CLEAR. All six files are
  canonical, finite, self-bound JSON; `completed.json` exactly binds and
  cross-binds the other five. The independent audit passed 106 checks.
- Access is complete: 1,000 optimizer updates, 16,000 train presentations,
  10,240 validation presentations, and exactly 183,680 RGB open attempts and
  successes. Active GPU time remained below the 5,400-second cap.
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

- Retain the centered categorical-action pathway and four coherent trajectory
  hypotheses. Across multiple mechanisms these are now robust sources of
  action sensitivity, persistence improvement, and distribution value.
- Close the continuous writable-history hypothesis represented by this exact
  global error-by-action matrix. Alongside recurrent, dense, incoming-delta,
  momentum, correspondence, inverse, compact-target, and whitening failures,
  another continuously encoded history state is not justified.
- The leading materially different falsification is one fresh **causal
  posterior-reweighted transition-expert JEPA**. Four jointly learned,
  centered-action transition experts begin with equal mass. On `p0` and `p1`,
  each predicts before observation; a fixed likelihood derived from its
  full-lattice prior error performs a Bayesian posterior update over expert
  mass. Factual `q` is then assimilated. History persists only through those
  probabilities, which cannot move content without the current action.
- This tests whether the missing temporal information is discrete regime
  identification—actual response, slip, or transient motion—rather than a
  continuous latent memory. Correct ordered evidence should concentrate mass
  on the matching action-response expert; reset or reversed evidence should
  choose a worse mixture.
- The successor must contain no RNN/Transformer history encoder, writable
  matrix, momentum, dense/token memory, raw-increment carrier, correspondence,
  warp, inverse classifier, per-action operator bank, trained history
  corruption, hinge repair, checkpoint reuse, or coefficient sweep. It starts
  fresh from accepted N320 and remains one jointly trained RGB/action JEPA.
- Use one separately preregistered 1,000-update / 16,000-presentation
  falsification with the same causal V2 schedules and selection rule. Require
  all existing 32 gates, particularly H4 history `>=0.03`, positive history
  bootstrap, at least six history-positive families, H4 action `>=0.05`, and
  broad HOLD/persistence support. A nonpositive history result closes this
  regime-selection mechanism without expert-count, temperature, seed, or
  duration retries.
- This STOP grants no checkpoint access, navigation, G2, held-out/sealed
  access, promotion, production, or deployment authority.

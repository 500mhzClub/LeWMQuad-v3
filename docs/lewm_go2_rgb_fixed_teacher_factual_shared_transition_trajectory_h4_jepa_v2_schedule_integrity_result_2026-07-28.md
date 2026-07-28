# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V2 schedule-integrity result — 2026-07-28

## Terminal outcome

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_JEPA_V2_SCHEDULE_INTEGRITY`.
- The sole authorized schedule-integrity replacement completed normally at the
  exact cap: 1,000 optimizer updates, 16,000 ordered training presentations,
  and 10,240 validation presentations in `722.020651` active GPU seconds.
- Update 1,000 / presentation 16,000 was selected by the unchanged rule: the
  minimum validation 50/50 joint-plus-marginal normalized energy score among
  eligible noncollapsed trained observations.
- The result passed 26 of 32 gates and failed exactly six. This is a
  scientific STOP, not an execution or custody failure.
- The one science-identical causal-schedule replacement is consumed. There is
  no retry, resume, longer run, second seed, schedule V3, threshold change,
  scalar tweak, generated-checkpoint opening/reuse, or same-mechanism scale-up.
- This remains a bounded development perception/world-model result. It does
  not establish a navigation policy or grant navigation, held-out, sealed,
  benchmark, promotion, production, or deployment authority.

## What was tested

- The exact V1 model and executable science were reused. Only each transition's
  schedule identity changed from the mixed boundary to the causal requested-
  action edge `F(i-1,5) --p_i--> F(i,5)`.
- One shared spatial Transformer transition and one shared residual head were
  trained jointly with the online RGB encoder in one backward pass. This was
  not a separately trained predictor.
- Inputs remained RGB `e0:e2`, requested past actions `p0:p1`, and requested
  future actions `p2:p5`; fixed-teacher targets remained future RGB `e3:e6`.
- The objective remained exactly half all-six factual local-innovation energy
  score, half cumulative open-loop future-trajectory energy score, plus the
  weight-one online-to-fixed-teacher alignment loss.
- Cyclic wrong action, all hold, reordered/reset history, persistence, and
  particle-collapse branches remained validation-only. No action margin,
  navigation loss, label, pose, depth, flow, BEV, reconstruction, or privileged
  state entered training.

## Selected aggregate result

| Measure | Update 1,000 value |
|---|---:|
| Combined normalized energy score | 0.742206799120 |
| Joint-trajectory normalized energy score | 0.737035297169 |
| Future `p2:p5` local combined score | 0.806352180925 |
| Pre-observation `p0:p1` local-prior combined score | 0.785016519953 |
| Pre-observation `p0:p1` persistence gap | +0.214983480134 |
| Pre-observation gap bootstrap lower 95% | +0.193886816321 |
| H4 marginal normalized energy score | 0.782288922931 |
| H4 persistence gap | +0.217711077696 |
| H4 persistence bootstrap lower 95% | +0.182847724686 |
| H4 cyclic-action gap | +0.000615848711 |
| H4 cyclic-action bootstrap lower 95% | +0.000226677750 |
| H4 ordered-history gap | -0.028848747242 |
| H4 ordered-history bootstrap lower 95% | -0.039597537605 |
| H4 all-hold gap | -0.000962185627 |
| Combined distribution-value gap | +0.244953636732 |
| Combined distribution-value bootstrap lower 95% | +0.239074013077 |
| H4 normalized pairwise spread | 1.220824440555 |
| H4 best-atom normalized squared error | 2.198116762786 |
| H4 centroid normalized squared error | 2.261630413124 |

- Generic prediction is strong and noncollapsed. Combined, joint, every
  future marginal, and the factual `p0:p1` prior beat persistence. H4
  persistence and `p0:p1` persistence are positive in all eight families.
- Particle support remains useful. Distribution value is positive in all
  eight families. Selected target effective-rank ratio is
  `0.176579435666`, online ratio is `0.204876800378`, and both near-zero-
  variance fractions are zero.
- Causal correction made the cyclic-action gap reliably positive in sign: its
  bootstrap lower bound is now above zero and seven families are positive.
  Its magnitude is nevertheless only `0.000616`, or 1.23% of the required
  `0.05`; it is not useful action conditioning.
- Ordered history remains harmful at every trained observation and in all
  eight families. The all-hold gap remains negative overall and is positive
  in only one family.

## Learning trajectory

| Update | Presentations | Combined | `p0:p1` prior | H4 score | H4 persistence | H4 action | H4 history | H4 hold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 250 | 4,000 | 0.793633 | 0.832652 | 0.847339 | +0.152661 | -0.000102 | -0.023077 | -0.000120 |
| 500 | 8,000 | 0.784748 | 0.826712 | 0.849191 | +0.150809 | +0.000145 | -0.024415 | +0.000457 |
| 750 | 12,000 | 0.746834 | 0.797136 | 0.784909 | +0.215091 | +0.000641 | -0.028286 | -0.000098 |
| 1,000 selected | 16,000 | 0.742207 | 0.785017 | 0.782289 | +0.217711 | +0.000616 | -0.028849 | -0.000962 |

- The registered selection score continued improving through the exact cap,
  so update 1,000 is the correct selected observation.
- The curve does not justify extending this mechanism. Prediction kept
  improving, but action sensitivity stayed about 81 times below the required
  useful margin, ordered-history value stayed negative, and hold sensitivity
  returned negative.

## Selected per-family findings

| Family | H4 score | H4 persistence | H4 action | H4 history | H4 hold | `p0:p1` persistence |
|---|---:|---:|---:|---:|---:|---:|
| `large_enclosed_maze` | 0.776696 | +0.223304 | +0.001311 | -0.030995 | +0.000040 | +0.228912 |
| `local_composite_motifs` | 0.799710 | +0.200290 | +0.000070 | -0.032261 | -0.000371 | +0.167999 |
| `loop_alias_stress` | 0.841756 | +0.158244 | +0.000778 | -0.047756 | -0.002260 | +0.187833 |
| `medium_enclosed_maze` | 0.851508 | +0.148492 | +0.001890 | -0.051184 | -0.002300 | +0.222950 |
| `open_obstacle_field` | 0.711776 | +0.288224 | -0.000152 | -0.012219 | -0.000198 | +0.257354 |
| `rough_local_dynamics` | 0.743365 | +0.256635 | +0.000004 | -0.012012 | -0.000953 | +0.238490 |
| `small_enclosed_maze` | 0.756988 | +0.243012 | +0.000085 | -0.023765 | -0.001290 | +0.198335 |
| `visual_sensor_stress` | 0.776514 | +0.223486 | +0.000940 | -0.020598 | -0.000367 | +0.217997 |

## Exact gate failures

The six failed gates were:

1. `h4_action_gap_at_least_point05`
2. `h4_history_gap_at_least_point03`
3. `h4_history_gap_bootstrap_lower_positive`
4. `h4_hold_gap_positive`
5. `history_positive_in_six_families`
6. `hold_positive_in_six_families`

- The other 26 gates passed, including exact completion, finite values, fixed
  teacher identity, noncollapse, every prediction/persistence gate, all
  particle-value gates, all four factual `p0:p1` gates, action bootstrap sign,
  action-family breadth/floor, and the all-hold family floor.

## Comparison with mixed-boundary V1

| Measure | V1 selected | V2 selected | V2 minus V1 |
|---|---:|---:|---:|
| Combined score | 0.761865 | 0.742207 | -0.019658 |
| Joint score | 0.756964 | 0.737035 | -0.019929 |
| Future local score | 0.817823 | 0.806352 | -0.011471 |
| H4 score | 0.807902 | 0.782289 | -0.025613 |
| H4 persistence gap | +0.192098 | +0.217711 | +0.025613 |
| H4 action gap | -0.000079 | +0.000616 | +0.000695 |
| H4 action lower 95% | -0.000463 | +0.000227 | +0.000690 |
| H4 history gap | -0.037439 | -0.028849 | +0.008590 |
| H4 hold gap | -0.001372 | -0.000962 | +0.000410 |

- Correcting the schedule materially improved generic future prediction and
  H4 persistence. It also changed action sensitivity from indistinguishable
  from zero to consistently positive in sign.
- Those improvements are not a usable controlled state. The action magnitude
  remained tiny, ordered visual history remained counterproductive in every
  family, and all-hold sensitivity remained absent.
- The V1 endpoint defect was therefore real and worth correcting, but it was
  not the primary cause of the mechanism's failure to learn controllable
  dynamics.

## Receipt and custody audit

| Receipt | Bytes | File SHA-256 |
|---|---:|---|
| `reservation.json` | 7,798 | `605372a50c3ddc8862eb13fa6b9551568eb18f6d3e371110fa61b528e3a2b979` |
| `metrics.json` | 60,798 | `52f36c0b819877cc8653bef74c49c62371136931a183edf84ff30e3d95d44b44` |
| `artifact.json` | 6,447 | `fd005d23d73fda3d4e793a46a0a97119568c4511ac2ad6cc8e1b63a7a0f2a4df` |
| `access.json` | 1,304 | `b143177a1567eee8f4016f9824cb6c34d77dbcb2966c567a081dd703261316d3` |
| `result.json` | 3,233 | `e199f5cd2177560e8b4762b26de30bda945ddba2273c99eaa7fbe1e7cb66ce82` |
| `completed.json` | 1,912 | `c49ee7ccd5509d73d56c2427d5245dd65db6c6b8638bee8af1b7d23cf24986d4` |

- Root receipt-only audit returned **CLEAR**. All six files are canonical,
  finite, self-bound JSON; `completed.json` exactly cross-binds the other five
  receipts; and source, index, census, N320, cap, loss-arithmetic, selection,
  result, and decision identities all reconcile.
- Two restricted independent receipt-only audits also returned **CLEAR** and
  independently reproduced the selected update, all 32 gate values, the six
  failures, exact counters, and terminal STOP identity.
- Access is complete: 1,000 optimizer updates, 16,000 train presentations,
  10,240 validation presentations, and 183,680 RGB open attempts/successes.
  Target EMA, wrong-action training controls, auxiliary training controls,
  retry/resume, arbitrary initialization, labels, test/held-out, and sealed
  access are all zero.
- The fixed target's initial and final state hashes are identical and its EMA
  count is zero. Exactly one accepted N320 initialization was opened; only 78
  `encoder.*` tensors were copied and no non-encoder tensor was copied.
- Runtime checkpoint files remain unopened and inaccessible. Their receipt
  inventory was read only as fields inside `artifact.json`; no generated `.pt`
  was listed, statted, hashed, or opened.

## Scientific conclusion and ordered next step

- This exact shared-transition mechanism is closed. The data can support
  noncollapsed generic latent prediction, but the objective lets the model
  explain average visual evolution while almost ignoring action and retaining
  no useful ordered-history evidence.
- More of the 2.896 TB pool, more updates, or another endpoint/schedule version
  would scale the wrong inductive bias rather than fix it.
- The ordered next candidate is a **factorized conditional-increment H4
  JEPA**: explicitly separate visual
  belief/history state from the action-conditioned latent increment so the
  predictor must represent what the requested action changes, while remaining
  RGB-only, jointly learned, perception-only, and free of navigation labels or
  privileged geometry.
- No factorized run is authorized merely by this STOP. Its minimal mechanism,
  source diff, falsification cap, and gates must be frozen before one new
  attempt.

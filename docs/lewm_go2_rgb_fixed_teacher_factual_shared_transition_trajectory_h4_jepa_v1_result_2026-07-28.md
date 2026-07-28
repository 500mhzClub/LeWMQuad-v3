# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V1 result — 2026-07-28

## Superseding causal-alignment correction

- This correction supersedes only the prior no-rebuild/factorized-next
  consequence recorded in commit
  `dab0e31dd25194b07efb9a542f3d677bb62201c9`. Every V1 observation, metric,
  receipt, failure gate, custody finding, and terminal STOP below is unchanged.
- The later causal adjudication distinguishes correlation from a valid
  candidate-planning transition. V1 labels `F(i,1) -> F(i+1,1)` with requested
  primitive `p_i`: its input is already one post-request tick into `p_i`, and
  its target is already one post-request tick into unseen `p_(i+1)`.
- The valid requested-action edge is same-episode
  `F(i-1,5) -> F(i,5)`, labeled `p_i`. One fresh, mechanism-identical,
  schedule-corrected V2 integrity replacement is therefore required before
  changing the model. The factorized conditional-increment mechanism is
  deferred and may be reconsidered only if corrected V2 still stops on the
  unchanged gates.

## Terminal outcome

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_JEPA_V1`.
- The sole authorized attempt completed normally at the exact cap. This was a
  scientific STOP, not an execution failure: 1,000 optimizer updates, exactly
  16,000 ordered training presentations, and 10,240 validation presentations
  completed in `731.364354` active GPU seconds.
- Update 750 / presentation 12,000 was selected by the preregistered minimum
  validation combined joint-plus-marginal normalized energy score among
  eligible noncollapsed trained observations.
- The selected result passed 25 of 32 gates and failed exactly seven.
- The exact one-block, K=4, factual 50/50 shared spatial-transition mechanism
  on the frozen V1 schedule is closed. There is no retry, resume, second seed,
  longer run, block-depth change, coefficient change, same-schedule nearby V2,
  checkpoint inspection, or checkpoint reuse. This does not prohibit the one
  fresh mechanism-identical, schedule-corrected V2 integrity replacement
  defined by the superseding correction: V2 starts again from the accepted
  N320 encoder and consumes no V1 runtime artifact.
- This result is a bounded development perception/world-model result. It does
  not establish a navigation policy and grants no navigation, held-out,
  sealed-benchmark, scaling, promotion, deployment, or checkpoint authority.

## What was tested

- One fresh spatial Transformer transition and one zero-initialized residual
  head were reused with exactly the same parameters on all six action edges.
- The model predicted `e1` and `e2` before those observations were visible,
  then inserted each factual online visual carrier while retaining its causal
  hidden state. It subsequently rolled the same transition open-loop through
  `p2:p5` with four coherent equal-mass particles.
- One backward pass jointly trained the online RGB encoder, hidden-state
  initializer, action/mode/spatial embeddings, transition, and residual head.
  The predictor was not trained as a separate stage.
- The loss was exactly half all-six-edge factual local-innovation energy score,
  plus half cumulative open-loop future-trajectory energy score, plus the
  weight-one three-frame online-to-fixed-teacher alignment loss.
- Cyclic wrong action, all hold, reordered/reset history, persistence, and
  particle-collapse controls were validation-only. No counterfactual ranking
  term, navigation label, pose, depth, flow, BEV, reconstruction target, or
  privileged state entered training.
- The only checkpoint input was the accepted N320 `encoder.*` prefix. All
  transition-side parameters were fresh, and the target encoder remained
  exactly fixed.

## Selected aggregate result

| Measure | Update 750 value |
|---|---:|
| Combined normalized energy score | 0.761864779262 |
| Joint-trajectory normalized energy score | 0.756963854170 |
| Future `p2:p5` local combined score | 0.817823032368 |
| Pre-observation `p0:p1` local-prior combined score | 0.777012986444 |
| Pre-observation `p0:p1` persistence gap | +0.222987013988 |
| Pre-observation gap bootstrap lower 95% | +0.205308728708 |
| H4 marginal normalized energy score | 0.807901859631 |
| H4 persistence gap | +0.192098141719 |
| H4 persistence bootstrap lower 95% | +0.147160076562 |
| H4 cyclic-action gap | -0.000079397548 |
| H4 cyclic-action bootstrap lower 95% | -0.000463193713 |
| H4 ordered-history gap | -0.037439019574 |
| H4 ordered-history bootstrap lower 95% | -0.055555247097 |
| H4 all-hold gap | -0.001372094824 |
| Combined distribution-value gap | +0.250435743625 |
| Combined distribution-value bootstrap lower 95% | +0.242493082294 |
| H4 normalized pairwise spread | 1.250721692077 |
| H4 best-atom normalized squared error | 2.987982367184 |
| H4 centroid normalized squared error | 2.582948173299 |

- Prediction and particle value were real. All four future marginal scores,
  the joint score, the combined score, and the dedicated factual `p0:p1`
  prior score beat persistence. H4 persistence was positive in all eight maze
  families, as was the `p0:p1` persistence gap.
- Four-particle support remained noncollapsed and useful. Combined
  distribution value was positive in all eight families. The selected target
  effective-rank ratio was `0.174980600675`, the online ratio was
  `0.204294462999`, and both near-zero-variance fractions were zero.
- The learned transition did not show meaningful action conditioning under the
  registered H4 cyclic-action test. The gap was slightly negative in aggregate,
  only six families were positive, and even those values were tiny.
- Ordered visual history was counterproductive at every horizon and in all
  eight families. The registered metric compares real ordered history with
  the better of the reset/reordered controls; it does not claim that both
  controls separately won.
- All-hold sensitivity was also slightly negative in aggregate and in every
  family. The effect was small and passed the `-0.02` family floor, but it
  failed the required aggregate and breadth gates.

## Learning trajectory

| Update | Presentations | Combined | `p0:p1` prior | H4 score | H4 persistence | H4 action | H4 history | H4 hold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 250 | 4,000 | 0.810820 | 0.822891 | 0.887985 | +0.112015 | +0.000403 | -0.017661 | +0.000060 |
| 500 | 8,000 | 0.784356 | 0.796099 | 0.853264 | +0.146736 | +0.000178 | -0.022980 | +0.000364 |
| 750 selected | 12,000 | 0.761865 | 0.777013 | 0.807902 | +0.192098 | -0.000079 | -0.037439 | -0.001372 |
| 1,000 | 16,000 | 0.768988 | 0.785885 | 0.826115 | +0.173885 | +0.000804 | -0.028372 | -0.000849 |

- The registered prediction criterion improved materially through update 750
  and then regressed, so selection was correct.
- Immediate factual priors and open-loop prediction learned materially through
  update 750, then regressed slightly. Action sensitivity stayed effectively
  zero, while ordered-history evidence stayed negative at every trained
  observation. More updates to this same mechanism are therefore not
  scientifically justified.

## Selected per-family findings

| Family | H4 score | H4 persistence | H4 action | H4 history | H4 hold | `p0:p1` persistence |
|---|---:|---:|---:|---:|---:|---:|
| `large_enclosed_maze` | 0.770570 | +0.229430 | +0.000549 | -0.040753 | -0.000840 | +0.248059 |
| `local_composite_motifs` | 0.938715 | +0.061285 | +0.000844 | -0.075261 | -0.001417 | +0.153004 |
| `loop_alias_stress` | 0.829729 | +0.170271 | +0.000074 | -0.056960 | -0.002363 | +0.228730 |
| `medium_enclosed_maze` | 0.803403 | +0.196597 | +0.000834 | -0.040213 | -0.002075 | +0.238247 |
| `open_obstacle_field` | 0.737058 | +0.262942 | -0.001685 | -0.014454 | -0.000713 | +0.249930 |
| `rough_local_dynamics` | 0.787358 | +0.212642 | -0.001856 | -0.010895 | -0.001056 | +0.230530 |
| `small_enclosed_maze` | 0.823123 | +0.176877 | +0.000281 | -0.044775 | -0.001319 | +0.220619 |
| `visual_sensor_stress` | 0.773259 | +0.226741 | +0.000324 | -0.016201 | -0.001194 | +0.214778 |

- The future-four all-hold string has sparse frozen-train support in
  `large_enclosed_maze` (one row) and none in `medium_enclosed_maze`; this
  limits fine family-level interpretation of that control. It does not explain
  the result away: all eight families were negative, factual hold transitions
  occur at every action position in every family, and the action and history
  controls independently failed.

## Exact gate failures

The seven failed gates were:

1. `h4_action_gap_at_least_point05`
2. `h4_action_gap_bootstrap_lower_positive`
3. `h4_history_gap_at_least_point03`
4. `h4_history_gap_bootstrap_lower_positive`
5. `h4_hold_gap_positive`
6. `history_positive_in_six_families`
7. `hold_positive_in_six_families`

The other 25 gates passed, including exact completion, finite values, fixed
teacher identity, noncollapse, all prediction/persistence gates, all particle
value gates, all four new `p0:p1` factual-prior gates, the action-family breadth
and floor gates, and the all-hold family floor.

## Comparison with the immediate predecessors

| Selected mechanism | Combined | H4 score | H4 persistence | H4 cyclic action | H4 history | H4 hold |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative K4 parallel trajectory | 0.725748 | 0.742969 | +0.257031 | +0.010815 | -0.013974 | -0.006572 |
| Local-only with trained controls | 0.852201 | 0.953251 | +0.046749 | +0.187365 | +0.000948 | +0.001280 |
| Dual-domain with trained controls | 0.778369 | 0.852012 | +0.147988 | +0.148459 | -0.000456 | -0.026661 |
| Factual shared transition | 0.761865 | 0.807902 | +0.192098 | -0.000079 | -0.037439 | -0.001372 |

- Relative to dual-domain V1, tying factual belief construction and open-loop
  rollout improved combined fit by `0.016504`, H4 fit by `0.044110`, and H4
  persistence margin by `0.044110`. It also made the reserved all-hold result
  much less negative.
- Those improvements did not transfer to causal conditioning. Removing trained
  corruptions exposed essentially zero action sensitivity, and ordered-history
  use became materially worse.
- Relative to cumulative K4, the new shared transition was worse on prediction,
  action, and ordered history. Its less-negative hold gap is not enough to
  qualify a controllable state.

## Scientific conclusion

- Weight sharing and factual one-step supervision were sufficient to learn a
  strong generic latent successor prior, including useful pre-observation
  `p0/p1` predictions and stable four-step open-loop prediction.
- On the V1 mixed-boundary schedule they were not sufficient to learn a
  controllable predictive state. The transition explained average visual
  evolution while effectively ignoring which indexed primitive was applied;
  its retained hidden state did not add useful ordered evidence under the
  registered controls.
- This makes encoder collapse, lack of particle support, and inability to roll
  open-loop poor explanations for the descriptive V1 failure. It does not
  establish that a causally aligned factual signal is insufficient: the V1
  input already contains one tick of the labeled action and its target contains
  one tick of an unseen destination action. A corrected integrity replacement
  is required to separate model failure from temporal input semantics.
- The 2.896 TB allowlisted pool contains far more unused RGB and rollout data,
  but this curve does not justify exposing more of it to the same objective.
  Data scale follows mechanism qualification; it does not replace it.

## Post-run schedule-semantic finding

- A subsequent read-only audit of all 108,288 action edges in the exact frozen
  train/validation schedules found a concrete endpoint-alignment defect in the
  V1 index adapter. Each command block is requested at time `t` and produces
  five frames at `t+0.1` through `t+0.5` seconds. V1 joins the first frame of
  one block to the first frame of the next block. Its nominal 0.5-second edge
  therefore contains 0.4 seconds of the labeled primitive plus 0.1 seconds of
  the destination primitive.
- All 108,288 action IDs matched the source command context and all timing,
  registry, context, and schema checks passed. This is a one-tick endpoint
  contamination, not a wholesale action-label shift.
- Realized body-frame motion confirms that the current primitive remains the
  strongest association. A train-fitted standardized nearest-centroid
  diagnostic on disjoint validation scenes obtained balanced accuracy
  `0.4793` for the indexed current primitive, versus `0.2899` for the
  destination primitive and `0.2806` for the previous primitive; chance is
  `0.1111`. Pure within-block 0.4-second motion retained `0.4680` balanced
  accuracy. The defect is therefore material and removable, but it is
  unlikely to be the sole explanation for every near-zero JEPA action result.
- `command_context` records the requested primitive arrays, not the
  subsequently executed/clipped arrays. Requested actions are the correct
  causal inputs for future navigation planning; controller response and
  clipping must be inferred from observation/action history rather than
  supplied as future privileged input.
- A direct corrected-schedule counterfactual was computed before any new model
  or GPU run. Using the same-episode pre-command boundary through the fifth
  current-command tick excluded 620 of 96,000 train edges and 75 of 12,288
  validation edges. Its held-back accuracy / balanced accuracy / balanced
  standardized eta-squared
  were `0.5057 / 0.4523 / 0.3403`, all worse than V1's
  `0.5327 / 0.4793 / 0.3591`. Pure within-block 0.4-second motion was also
  weaker at `0.5196 / 0.4680 / 0.3492`.
- This ordering is consistent with actuator/body inertia: the first tick under
  a new request still contains substantial response to the preceding command.
  It explains why V1's impure interval is more class-separable, but does not
  make that interval valid for planning. Separability cannot license an input
  observed after candidate `p_i` began or a target affected by unseen
  `p_(i+1)`.
- The evidence does not convert the completed V1 result into a PASS or
  authorize any V1 checkpoint. It authorizes no model change by itself. The
  next scientific comparison must first repeat the exact V1 model and science
  once on reset-safe `F(i-1,5) -> F(i,5)` requested-action edges.

## Ordered next category

- Run at most one fresh **factual shared-transition trajectory-H4 JEPA V2
  schedule-integrity replacement**. It must retain requested primitives and
  the exact V1 model, initialization, seed, losses, weights, optimizer,
  observations, selection rule, gates, thresholds, and
  1,000-update/16,000-presentation cap.
- The only scientific changes are the reset-safe endpoint/index schema and the
  deterministic same-seed backfill needed to restore the fixed family quotas.
  New schedule hashes, output namespace, receipt schema, and source bindings
  are operational consequences of that integrity repair, not model changes.
- Every edge labeled `p_i` must be same-episode
  `F(i-1,5) -> F(i,5)`. Six consecutive complete primitive blocks must yield
  seven shared boundaries; no destination-action tick may enter an edge and no
  missing pre-command boundary may be synthesized.
- Do not merge forward-speed classes, expose executed/clipped commands, change
  a model or loss scalar, relax a gate, reuse a V1 checkpoint, retry, resume,
  or scale data. V2 starts fresh from the accepted encoder only.
- The factorized conditional-increment H4 JEPA is deferred. It becomes an
  admissible fallback only after a clean corrected V2 STOP still demonstrates
  generic prediction without the unchanged required action/history value.

## Execution and custody audit

- Preregistration commit: `5c038f054f17d7d8928518723b12e1166db2d17a`.
- Frozen implementation commit: `065bae4069d53a4d2c87f781df5ae9e29d5027a2`.
- Independent source review and exact execution authority commit:
  `bbc78938c5754cd81a1b6fb4b031d2ec76fe5921`.
- Source closure at execution matched all nine frozen bindings, including the
  factual model SHA-256
  `38e264f8e18ffa3c3da4775fdd7d4a38549e8544f99cd863bfd2534999cd5b36`
  and wrapper SHA-256
  `693cbea45b2a49f0f3edfb7cabce347b852a67af78df1ecf5462c65be48cd977`.
- The fixed teacher's initial and final state SHA-256 were both
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`;
  EMA updates were zero.
- Access was exactly 183,680 successful RGB opens from 183,680 attempts,
  comprising 16,000 training and 10,240 validation sequence presentations and
  6,900,398,764 physical RGB bytes. Test/held-out, sealed, label, arbitrary
  checkpoint, retry/resume input, and retry/resume counts were all zero.
- The accepted N320 initialization was opened exactly once and copied exactly
  78 `encoder.*` tensors; no non-encoder tensor was copied and no predecessor
  predictor checkpoint was opened.
- No auxiliary counterfactual sequence entered training. The fixed target was
  used for all 1,000 optimizer updates and received zero updates.

## Canonical receipt bindings

| Receipt | File SHA-256 | Content SHA-256 | Bytes |
|---|---|---|---:|
| Reservation | `029558ba63e53f9c8c416941c36b5713c4246c3d1f3858f4e07a712e2b4907ed` | `ac0b93da592f4a346729183492a0a660e9239f9f78edf60c1857f4361f9b46b0` | 6,103 |
| Metrics | `2f04cb429b568b29cbdbf920bda830ac1e1ea0c61e9a3ad2949ee96c7ce399ad` | `55af006888c9e602187717b85a802e221f683091d08861f4cb29d3c3973459df` | 60,782 |
| Artifact | `228b6786f40133c38d6589b4d758af42bc7cb35104ef7c523783ea5564931062` | `b3b1998d0799e6f8ebb3a8fd47841b6161a4673815dbbf7c5939cc7338677115` | 5,734 |
| Access | `91cb072f83de546354aec421f1d9c81fea9f843fdfeda9eb747f936a11d6c4af` | `07a053fb1d659db34420aa6739b74d1006f7fe86d60989d818c3efbb02e84bb2` | 1,285 |
| Result | `209ece3c1ec1d2e082d8cea17563aa348ef8b0c8f6d4c0da9ad2b2b2908866e4` | `40585393d2283d5c412f41b41382b4f3e521770740568b997e8ac631ababbf74` | 3,146 |
| Completion | `223c800ace7bd6e84c080db4f53a8c6cc17c1a7a25c0dd39c538ddbda7e38b0f` | `75328291c222f43f589861dd3c27d07eab933f52e94e261432c2b67aca875a89` | 1,874 |

All six canonical content hashes, file hashes, byte counts, and completion
cross-bindings were independently recomputed and matched. Checkpoint filenames
were read only as metadata from `artifact.json`; no generated `.pt` file was
listed, statted, hashed, opened, loaded, copied, or reused. All stopped-branch
checkpoints remain inaccessible. The legacy V4 sealed benchmark remains
unopened, development-only, and permanently ineligible for final evaluation.

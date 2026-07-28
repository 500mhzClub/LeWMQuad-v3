# Go2 RGB fixed-teacher dense spatiotemporal cross-attention H4 JEPA V1 result — 2026-07-28

## Outcome and cap

- Terminal decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_DENSE_SPATIOTEMPORAL_CROSS_ATTENTION_H4_JEPA_V1`.
- This was a completed scientific falsification, not an execution failure.
  The run completed exactly 1,000 optimizer updates, 16,000 ordered training
  sequence presentations, and all five registered validation observations in
  640.3690703859902 active GPU seconds, below the 5,400-second cap.
- The preregistered minimum-mean-error rule selected update 750 / 12,000
  presentations. The checkpoint was finite, noncollapsed, and eligible for
  scientific selection, but the result failed 12 PASS gates.
- The fixed teacher was unchanged: initial and final state SHA-256 were both
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`,
  with zero EMA updates and 1,000 optimizer updates using the fixed target.
- Complete access receipts report 183,680 successful physical RGB opens,
  10,240 validation-sequence presentations, zero retry/resume inputs, and
  zero held-out, test, sealed, label, or arbitrary-checkpoint access.
- An independent read-only terminal audit recomputed every receipt hash,
  cross-binding, count, source binding, selection, and gate. It inspected the
  four checkpoint files by `lstat` only and opened or hashed no checkpoint
  bytes.

## What was tested

- Frozen source commit:
  `e770445e79e9f54997041d57514466831e8c1308`.
- The online path retained normalized patch tokens from `e0`, `e1`, and `e2`,
  interleaved explicit `p0` and `p1` tokens, and encoded the resulting 770
  tokens with two independently initialized pre-norm transformer blocks.
- Four independent horizon-query grids used ordered, fixed-slot future-action
  prefixes and shared two pre-norm decoder blocks to cross-attend to all 770
  history tokens. There was no recurrent state, BEV, warp, cost volume,
  retrieval, or hand-written transport mechanism.
- The prediction was one direct, nonrecursive, `e2`-relative latent delta per
  horizon. The sole final linear was initialized to exact zero, and update
  zero reproduced online persistence within the registered `1e-5` tolerance.
- One joint backward pass trained the online encoder, history encoder, shared
  action path, cross-attention predictor, and delta head. The complete
  gradient-bearing objective was exactly weight-1 fixed-teacher future-delta
  regression plus weight-1 three-frame online/teacher alignment. Absolute
  prediction was diagnostic-only; variance and wrong-action training terms
  contributed zero.
- The exact accepted N320 encoder prefix was the only checkpoint input. No
  predecessor predictor checkpoint, trace, or tensor was consumed.

## Aggregate metrics

The table reports scene-then-family macro values; displayed values are rounded
to six decimals. Full-precision selected arrays are recorded immediately
below it.

| Update | H1 real | H2 real | H3 real | H4 real | H4 action gap | H4 history gap | H4 persistence gap | H4 hold gap | Target / online rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 0.174981 / 0.207486 |
| 250 | 4.248666 | 6.360874 | 4.711182 | 5.857959 | 0.698651 | -0.134033 | -4.857959 | -1.021816 | 0.174981 / 0.203314 |
| 500 | 4.406493 | 6.582714 | 4.371808 | 5.725554 | 0.826790 | -0.053608 | -4.725554 | -2.242714 | 0.174981 / 0.204346 |
| 750 selected | 1.510296 | 1.538938 | 1.396999 | **1.447611** | **0.111635** | **-0.077081** | **-0.447611** | **-0.169507** | 0.174981 / 0.208168 |
| 1,000 | 2.689115 | 3.057900 | 2.403585 | 2.802254 | 0.286633 | -0.314801 | -1.802254 | -0.463550 | 0.174981 / 0.203202 |

- Selected mean H1-H4 real error:
  `1.4734610515005948`.
- Selected full-precision real error:
  `[1.5102962305837937, 1.5389376749113068, 1.3969992257093977, 1.447611074797881]`.
- Selected full-precision action gap:
  `[0.029963651116979313, 0.07754529427035402, 0.0978373232469633, 0.11163520369005084]`.
- Selected full-precision history gap:
  `[-0.06549029906341121, -0.07507037886224854, -0.05775495134825284, -0.07708073240645034]`.
- Selected full-precision persistence gap:
  `[-0.5102962226858762, -0.5389376836363184, -0.3969992294995777, -0.447611069579582]`.
- Selected full-precision hold gap:
  `[-0.04693104568091557, -0.12078570339179838, -0.13507381467788082, -0.16950704286092436]`.
- Selected H4 scene-bootstrap lower bounds were `+0.04839312078027176`
  for action, `-0.13175236987522068` for history, and
  `-0.7173164254545998` for persistence.
- Every registered value was finite. Target rank was exactly
  `0.17498056093851724` with zero drift at every observation; all target and
  online near-zero-variance fractions were zero.
- Mean training losses over 1,000 updates were
  `future_teacher_delta=0.49721785697340964`,
  `history_teacher_alignment=0.010524454678642228`, and
  `total=0.5077423110306263`. At update 1,000 they were respectively
  `0.4877476692199707`, `0.008316921070218086`, and
  `0.49606460332870483`; variance and wrong-action ranking were exactly zero.
- The selected result passed the aggregate H4 action threshold and bootstrap
  lower-bound gates, but failed the 10% H4 improvement, H1-H3 improvement,
  every persistence/family-coverage requirement, every history requirement,
  the H4 hold requirement, and the action-family `-0.02` floor.

## Selected H4 family findings

Values below are displayed to nine decimals; sign tests and family counts use
the full receipt precision.

| Family | Real / persistence | Action gap | History gap | Persistence gap | Hold gap |
|---|---:|---:|---:|---:|---:|
| `large_enclosed_maze` | 1.281449757 | 0.235626707 | -0.114050059 | -0.281449727 | -0.066950063 |
| `local_composite_motifs` | 2.181767395 | 0.316749196 | -0.118728665 | -1.181767394 | -0.353394163 |
| `loop_alias_stress` | 1.420055173 | 0.042508617 | -0.163224444 | -0.420055175 | -0.184920665 |
| `medium_enclosed_maze` | 1.608055219 | 0.153901535 | -0.046698288 | -0.608055214 | -0.248978442 |
| `open_obstacle_field` | 0.976462013 | 0.016060823 | -0.012064538 | +0.023537987 | +0.001683333 |
| `rough_local_dynamics` | 1.060250241 | 0.029365580 | -0.003434317 | -0.060250240 | -0.021071338 |
| `small_enclosed_maze` | 1.861994369 | -0.030342841 | -0.145659739 | -0.861994364 | -0.379574084 |
| `visual_sensor_stress` | 1.190854432 | 0.129212013 | -0.012785809 | -0.190854429 | -0.102850920 |

- Action sensitivity was positive in seven of eight families, but
  `small_enclosed_maze` was negative and below the allowed `-0.02` floor.
- Only `open_obstacle_field` beat persistence at H4. The other seven families
  were negative; `rough_local_dynamics` was closest to a tie.
- Ordered-history gap was negative in all eight families. The history mechanism
  therefore did not beat the registered reset/reordered controls in any family.
- Hold gap was positive only in `open_obstacle_field` and negative in the
  other seven families.

## Comparison with recurrent fixed-teacher V3

- Both branches selected update 750 and preserved the same fixed-teacher
  geometry without collapse.
- Dense V1 reduced selected mean H1-H4 error from V3's receipt value
  `1.546897490755602` to `1.4734610515005948`, 4.75% lower. It did not improve
  the decisive H4 result: dense V1 was `1.447611074797881` versus V3's
  `1.4387762256665138`, 0.61% worse.
- Dense V1 had a larger aggregate H4 action gap (`0.111635` versus `0.0718`)
  and a positive bootstrap lower bound, but action positivity became less
  uniform: seven of eight families versus all eight in V3.
- Dense V1's aggregate H4 history gap was somewhat less negative
  (`-0.077081` versus `-0.0951`), yet history remained negative in every
  family. H4 persistence was slightly worse (`-0.447611` versus `-0.4388`),
  and the hold gap was materially worse (`-0.169507` versus `-0.0854`).
- By update 1,000 the action gap had risen to `0.286633`, while H4 error
  worsened to `2.802254` and history to `-0.314801`. As in V3, additional
  optimization within the cap increased action separability without learning
  a successor prediction that beat persistence.

## Causal-cautious interpretation

- Direct access to the uncompressed three-frame token history was not enough
  to solve the successor-dynamics problem under this exact target, loss,
  initialization, data, seed, and cap. This weakens the hypothesis that GRU
  compression was the sole blocker.
- It does not establish that dense attention is generally harmful or that
  history is unnecessary. This was one seed and one preregistered dense
  mechanism; architecture, optimization, and target-state effects are not
  separately identified.
- Positive action gaps show that the learned predictor responded to proposed
  actions. They do not show that it mapped those actions to the correct future
  state. Stable rank and a fixed teacher also make representation collapse and
  moving-target drift poor explanations for this run's failure.
- The combination of action sensitivity, persistence dominance, and uniformly
  negative history is consistent with a deterministic squared-error predictor
  averaging view-dependent or partially observed future outcomes. That is a
  hypothesis supported by the pattern, not a demonstrated cause.

## Terminal branch rule and recommended successor category

- STOP permanently closes this dense cross-attention mechanism and further
  deterministic dense-H4 predictor-architecture variants. Do not open, reuse,
  resume, extend, or promote any checkpoint from this attempt, and do not run
  a same-mechanism V2 or another seed.
- This result grants no navigation, G2-G8, held-out, sealed, promotion,
  production, or deployment authority.
- The recommended next category is an **uncertainty-aware latent JEPA**, not
  another deterministic history encoder: predict a calibrated conditional
  distribution over future fixed-teacher state changes, for example a small
  learned mixture with bounded scale and a proper likelihood or energy score,
  so partially observed/multimodal futures are represented rather than
  collapsed to one L2 point estimate. It must remain RGB/action-only and fully
  learned, use fresh accepted initialization, consume no failed checkpoint,
  and receive a separate preregistration and capped authorization.
- If uncertainty reformulation is not technically defensible, the alternative
  category is a separately reviewed predictive-target reformulation that
  learns a temporally stable, action-controllable state target rather than raw
  per-patch future-view deltas. Either route changes the scientific category;
  neither authorizes another deterministic dense-H4 variant.

## Receipt and source bindings

- Reservation file SHA-256:
  `34b753c45b726ca1299bc7dde6859a1796333c645e64c076bfce6462e2a10b99`;
  content SHA-256:
  `0b4b676c5fb64abddf1aa69725d3742810e1fc197ed7824354ac01a538421c4e`.
- Metrics file SHA-256:
  `547fb41800b842cad4a3549a21e426e4ce017771b95a687a15f538c08612a209`;
  content SHA-256:
  `a6e46a1aede6da1b9bf6b417910dae25d638a19e0e445ced91b5575ed1354467`.
- Artifact file SHA-256:
  `c36b34ea5f7e27ffbdb9e1b770ae7659e7952b745f22ef8f618780cd3294ca2b`;
  content SHA-256:
  `ef800e07e4024d00dd1e713c5020875addf4521cc1156f943a5d739d46692a96`.
- Access file SHA-256:
  `dadb0df50ceba45fd01b98326662856178cede4d2fc87b55d13b2692fea51551`;
  content SHA-256:
  `75b0bc5e68e767f6e23ccbc4dc4254ded8e59bfca467c9a917538641ed74616c`.
- Result file SHA-256:
  `c9fda682cb773903d9edcc81174833fbb88471a0f44ab77596d139ba067fde9d`;
  content SHA-256:
  `d7c8698deb6d8892d5dde809d3fd3b322f629094e5faf0c519bb1e03a882811a`.
- Completion content SHA-256:
  `9769b8a22ec91e7087487799f024bbafdfe4acb9e795ebc93e87139a8bd8fe66`.
  Completion file SHA-256:
  `0642ca09bb1bafb3d207c1aa3b425d6ffb7a69efad48bae69910473c02c57d0f`.
- Independent read-only terminal audit: PASS. All six canonical self-hashes,
  completion links, source bindings, expected access counts, forbidden-zero
  fields, fixed-teacher identity, selected update, and 22 decision gates
  reconciled exactly. The four checkpoint files were regular non-symlinks of
  the receipt-declared 31,457,649 bytes each; their bytes remained unopened.
- Frozen dense model source SHA-256:
  `5c74675b93667e6035fc21c9fe497880ba4bff22641b3e735272e4cc1ede3d30`;
  wrapper source SHA-256:
  `30838feadc6e211df0d8b32638abdcab761f9e77132dc52b90aedace232db142`.
- This result draft used only the six JSON receipts and committed source,
  preregistration, review, authorization, and predecessor-result documents.
  No checkpoint, tensor, RGB, index, held-out, sealed, or protected bytes were
  opened or hashed while preparing it.

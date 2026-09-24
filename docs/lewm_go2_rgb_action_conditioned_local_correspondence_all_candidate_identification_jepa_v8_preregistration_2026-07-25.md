# RGB Action-Conditioned Local-Correspondence All-Candidate Identification JEPA V8

Date: 2026-07-25

## Decision and scientific question

V7 is a valid, sealed, terminal scientific failure. It learned finite,
action-specific local transports and a weak correct-versus-deranged
correspondence signal in all eight scene families, but it did not bind that
signal to the executed action:

- correct/deranged correspondence cross-entropy was
  `0.998866617679596`, short of the frozen `<0.99` gate;
- executed/hardest-wrong correspondence cross-entropy was
  `1.0049521923065186`;
- the executed action beat the hardest wrong action in `0/8` families;
- true/hardest-wrong forward MSE was `1.0080513069145072`;
- true/mean-target forward MSE was `1.0535559913881147`.

V8 asks one bounded question: can the existing V7 local-correspondence
signal identify the executed action when the training objective explicitly
compares all nine already-computed candidate transport distributions?

This is the one evidence-led successor allowed by the V7 terminal audit. It
is not a V7 retry, resume, checkpoint continuation, data change, encoder
change, transport change, threshold relaxation, or hyperparameter sweep. If
this single V8 attempt fails any scheduled continuation gate, the
local-correspondence transport branch is closed.

## Binding authority and inheritance

The complete V7 registration is inherited unchanged:

- path:
  `docs/lewm_go2_rgb_action_conditioned_local_correspondence_transport_jepa_v7_preregistration_2026-07-25.md`;
- commit:
  `094a478d219488a953b008bd7487b0a4a729a5bb`;
- file SHA-256:
  `b17b543bc63b4201e8a049c90dfa43eaaf78b6035ccf4d94235c5bbc110b8aa0`;
- byte count: `21190`.

The V7 terminal classification and sole allowed successor are bound by:

- path:
  `docs/lewm_go2_rgb_action_conditioned_local_correspondence_transport_jepa_v7_terminal_audit_2026-07-25.json`;
- commit:
  `cf21f4a3ed2caed103a765584bcadd29284c9282`;
- file SHA-256:
  `1e284375a5d1c79419aa21c553e48a5d396c1d33b27e3a56c0e58c4dae08e28f`;
- content SHA-256:
  `6b30ac4bb3784ea58822de7114197d184cd3a0a257ca29a60b858ab97b99c6f3`;
- byte count: `23123`;
- status:
  `PASS_EXACT_VALID_SCIENTIFIC_FAILURE_TERMINAL_NO_RETRY`.

Every V7 clause remains exact unless this amendment explicitly replaces it.
This amendment replaces only:

1. the mechanism name, schema prefix, output root, and fresh-attempt identity;
2. the Phase-A objective by adding the single loss below;
3. initialization observations for that loss;
4. two additional anti-collapse continuation conjuncts for that loss;
5. receipt fields needed to record those observations.

In particular, V8 preserves without reinterpretation every V7 model tensor,
initialization, input, role, endpoint byte, action identity, target,
neighborhood, border rule, normalization, detach rule, optimizer, learning
rate, weight decay, gradient clip, EMA setting, seed, schedule index and
prefix, batch size, checkpoint population, presentation count, GPU-active
cap, forward metric, correspondence metric, threshold, family-count gate,
Phase-B boundary, forbidden-access rule, sealing rule, and terminal failure
rule.

This preregistration grants no training, GPU, runtime-input, output-root, or
clean-export authority. Execution requires a frozen V8 source commit and
source manifest, independent exact-source review, separate execution
authorization, and proof that the fresh V8 output root is absent. The
repository-wide sealed-material and clean-export authorities remain
independently controlling; this document does not expand either boundary.

## The single scientific change

V8 adds no parameter, head, branch, bias, temperature, margin, target,
augmentation, input, or learned feature.

Reuse the exact V7 detached local-correspondence target:

`Q` with conceptual shape `[B,256,9]`.

Reuse the exact V7 candidate student logits and probabilities for all nine
actions:

`g` and `P` with conceptual shape `[B,9,256,9]`.

The forward values are the same values already used by V7 Energy-NLL and
correspondence diagnostics. They must not be produced by another encoder
call, predictor, or transport path.

For target row `Q` and candidate log-probability row `logP`, retain V7's
exact centered-log soft cross-entropy:

`Hc(Q,logP) = -logP_4 - sum_o Q_o*(logP_o-logP_4)`.

For sample `i` and candidate action `a`, define the all-token
correspondence cost:

`C_i,a = (1/256) * sum_token Hc(Q_i,token, log(P_i,a,token))`.

The exact source computation is:

`all_candidate_token_Hc =`
`    centered_log_soft_cross_entropy(Q[:,None,:,:], g)`;

`C = all_candidate_token_Hc.mean(dim=2)`.

The second helper argument is the V7 transport logits `g`, not `log(P)`;
the helper performs the one and only `log_softmax`. Aggregation order is
token mean, then rowwise nine-action cross-entropy, then arithmetic mean
across rows. The target is broadcast, not materialized as a new
candidate-specific tensor, and no mean cost is formed across samples before
classification.

Define the nine action scores:

`S_i,a = -C_i,a`.

Let `a_i` be the frozen executed-action index and reuse the exact V7 detached
row scale:

`m_i = stop_gradient(mean_a(E_i,a)).clamp_min(1e-8)`.

The only new training term is:

`L_CORR_ID = mean_i(m_i * cross_entropy(S_i, a_i))`.

Its coefficient is exactly `1.0`. The exact V8 Phase-A objective is:

`L_V8 = L_V7 + L_CORR_ID`,

where `L_V7` is the full registered V7 objective, including executed-action
JEPA MSE, Energy-NLL, executed-only `L_CORR`, raw/projected variance and
covariance regularization, with all original coefficients unchanged.

The implementation must pass the exact existing
`action_losses.row_scale` as `m`; it must not recompute a scale from `C`.
Classifier scores are raw `-C`, with no division by `m`, temperature, or
other scale.

For diagnostics, define the unscaled correspondence-action NLL:

`NLL_CORR_ID = mean_i(cross_entropy(S_i, a_i))`.

The gate applies to this unscaled diagnostic, not to `L_CORR_ID`, so changes
in JEPA energy scale cannot manufacture an apparent classifier improvement.

There is no loss normalization beyond the inherited detached `m_i`, no
class reweighting, label smoothing, logit scaling, temperature, margin,
entropy term, top-1 training term, auxiliary label, candidate subsampling,
or coefficient sweep.

## Gradient and detach topology

The forward values and detach topology remain V7-identical:

- `Q`, EMA-current tokens, EMA-next tokens, EMA states, and `m` are detached;
- every non-hold candidate `P` row remains live to the shared
  `transport_weight` and existing relative action embeddings;
- hold's relative embedding is exact zero, so its row contributes no
  transport-weight or action-embedding gradient;
- for a non-hold executed action, the online raw encoder, online geometry,
  and predictor trunk `h` are live through that candidate only;
- the predictor trunk is detached for all wrong candidates, and an executed
  hold row is structurally independent of `h`;
- `L_CORR_ID` consumes transport logits only: it has no shared-residual
  projector or online-target-projector path;
- hold retains exact `e_rel_hold=0`, uniform `P`, zero expected offset, and
  identity transport;
- no target or EMA state receives a gradient.

`L_CORR_ID` therefore supplies a direct comparative gradient to the same
transport/action mechanism that V7 activated weakly. It cannot train a
separate classifier or consume action-independent pooled features.

At exact-zero `transport_weight`, the source-only deterministic fixture must
prove:

- all nine candidate costs are bitwise equal;
- all nine scores are bitwise equal;
- the classifier posterior is uniform;
- `NLL_CORR_ID` is bitwise equal to a same-device, same-dtype standard
  cross-entropy reference evaluated on nine equal logits (the
  implementation-frozen float32 `log(9)` baseline);
- the all-candidate identification path and complete V8 objective are finite;
- `L_CORR_ID` itself has a finite nonzero `transport_weight` gradient;
- the complete objective has a finite nonzero `transport_weight` gradient;
- transport-path gradients to the predictor trunk, online encoder, and
  action embeddings remain exact zero;
- every target and EMA gradient is absent.

After installing the same deterministic bitwise-nonzero transport weight
used by the V7 routing fixture, it must prove:

- `L_CORR_ID` itself has finite nonzero gradients to `transport_weight`, the
  executed non-hold online raw encoder/geometry/predictor path, and non-hold
  action embeddings;
- wrong candidates do not open an online predictor/encoder gradient path;
- no target or EMA gradient exists;
- hold remains exact uniform/zero-offset/identity.

A separate executed-label value-invariance fixture must change only `a_i`
while holding the RGB-derived states and all model tensors fixed. The full
`g`, `P`, `C`, and `S` banks must remain bitwise identical; only the
cross-entropy target and resulting gradient may change. This prevents the
executed-label detach mask from becoming a forward label-leakage path.

These are source-acceptance tests, not additional runtime attempts.

## Frozen science and schedule

V8 preserves exactly the V7 frozen inputs and runtime science, including:

- raw V13 train and checkpoint-selection roles: `4262` and `495` pairs,
  `72` and `8` scenes;
- current RGB, next RGB, and executed action for every pair;
- exact nine-action vocabulary and order;
- qualified N320 online and EMA encoder initialization only;
- base seed `20260712` and schedule seed `20260713`;
- exact first `16000` pair presentations and prefix hashes:
  - update `100`:
    `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
  - update `400`:
    `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
  - update `1000`:
    `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`;
- ViT, projector, predictor, transport, optimizer groups, AdamW settings,
  float32, EMA `0.996`, global gradient clip `1.0`, no autocast, patch
  whitening, shared residual alpha, observations, and conditional Phase B;
- strict deterministic algorithms with `warn_only=False`;
- update decisions at `100`, `400`, and `1000`;
- maximum Phase A of `1000` updates, `16000` presentations, and `60`
  GPU-active minutes.

One scheduled current/next pair remains exactly one presentation regardless
of the nine candidate costs. The schedule adapter remains schema-only and
must not mutate, reorder, filter, regenerate, reseed, replace, or extend a
schedule index.

## Runtime observations and gates

Record on the exact same `495` checkpoint-selection pairs:

- `NLL_CORR_ID`;
- the canonical metric field
  `unscaled_correspondence_action_nll`;
- the frozen update-zero `NLL_CORR_ID` baseline;
- classifier posterior finiteness and row normalization;
- top-1 accuracy overall;
- per-executed-action row count, mean NLL, and recall;
- nine-action macro balanced accuracy, the arithmetic mean of the nine
  per-action recalls.

Top-1 accuracy and per-action NLL are observation-only. They are not
promotion gates and do not alter class weighting.

For receipt-only observations define:

- `R_i = softmax_a(S_i,a)`;
- NLL per row by standard
  `cross_entropy(S_i,a_i,reduction="none")`, then `NLL_CORR_ID` as its
  arithmetic mean over all exact `495` rows;
- top-1 as `argmax_a(S_i,a)` in frozen action-vocabulary order, retaining
  the standard lowest-index tie behavior;
- per-executed-action mappings in frozen `ACTION_VOCABULARY` order, each
  with exact row count, arithmetic mean row NLL, and recall; fail closed if
  any action has zero rows.

At update zero construct:

`Z = zeros_like(S)`;

`R_zero = softmax(Z,dim=-1)`;

`NLL_zero_rows = cross_entropy(Z,a_i,reduction="none")`.

This same-device, same-dtype reference freezes the exact float32 `log(9)`
bits without relying on a host-language logarithm or tolerance.

Independent preregistration science review identified one otherwise
available shortcut: an imbalanced action-prior model can reduce population
NLL without identifying minority actions. V8 therefore reuses V6's already
frozen, non-tuned macro-balanced-accuracy threshold of strictly above `2/9`.
This is an anti-collapse gate, not a new loss or mechanism. The inherited V7
correct/deranged and executed/hardest-wrong correspondence gates remain the
direct anti-shortcut controls; no duplicate classifier-derangement gate is
added.

At update zero, require:

- every inherited V7 update-zero check;
- all candidate costs and scores bitwise equal;
- classifier posterior `R` bitwise equal to `R_zero`;
- every row NLL bitwise equal to `NLL_zero_rows`, and finite
  `NLL_CORR_ID` bitwise equal to their arithmetic mean;
- all nine action populations nonempty and exact;
- first-index argmax predicts only action index zero, giving recall `1` for
  that class, recall `0` for the other eight, and macro balanced accuracy
  exactly `1/9`.

At updates `100`, `400`, and `1000`, require:

- every inherited V7 gate at that update, without relaxation;
- finite `NLL_CORR_ID` strictly below the frozen update-zero value;
- finite nine-action macro balanced accuracy strictly above `2/9`.

The two new gate keys are:

- `finite_unscaled_correspondence_action_nll_strictly_below_frozen_update_zero_log9`;
- `correspondence_action_identification_macro_balanced_accuracy_strictly_above_two_ninths`.

No V7 failed gate is removed, weakened, substituted, or made observational.
In particular, V8 must still satisfy:

- correct correspondence cross-entropy strictly below update zero;
- correct/deranged correspondence ratio `<0.99`;
- positive deranged-minus-correct margin in at least `6/8` families;
- executed/hardest-wrong correspondence ratio `<0.99`;
- positive hardest-wrong-minus-executed margin in at least `6/8` families;
- true/cyclic-wrong, true/hardest-wrong, non-hold/hold, and
  true/mean-target forward ordering at the exact registered thresholds;
- all rank, variance, spatial-diversity, family-count, finiteness, RNG,
  mutation, distribution, offset, and hold invariants.

The inherited terminal statuses remain:

- update-100 failure:
  `FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL`;
- update-400 failure:
  `FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL`;
- update-1000 failure:
  `FAIL_PHASE_A_TERMINAL_NO_PHASE_B_NO_RETRY`;
- full Phase-A pass:
  `PASS_PHASE_A_ENTER_FROZEN_PHYSICAL_PROBE`.

Phase B may begin only after the complete inherited final conjunction plus
both V8 anti-collapse gates pass. It may begin only in the same process and
may copy only the in-memory terminal online raw encoder state. It must not
copy or optimize the V8 transport tensor, predictor, online or target
projectors, correspondence targets, or optimizer state, and it must never
reopen a Phase-A checkpoint. All other Phase-B inputs, source, optimization,
caps, thresholds, and custody remain unchanged. Neither a Phase-A nor
Phase-B pass authorizes G2, navigation, held-out, sealed, promotion,
production, or deployment.

## Fresh namespace and custody

The exact V8 schema prefix is:

`lewm_go2_rgb_action_conditioned_local_correspondence_all_candidate_identification_jepa_v8`.

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_action_conditioned_local_correspondence_all_candidate_identification_jepa_probe_v8`.

It must be absent before reservation. It is a new
`attempt_index=1` of `maximum_attempts=1`; it is not V7 attempt 2. The root
must be reserved mode `0700` before importing Torch or opening runtime RGB,
schedule, N320, gate, or checkpoint bytes. Any post-reservation integration
or runtime failure consumes this sole attempt.

V1 through V7 documents are evidence-only. No V1 through V7 generated
runtime root, receipt, metrics payload, checkpoint, tensor, or trace may be
opened, hashed, copied, loaded, resumed, or used for initialization. In
particular, the V7 checkpoint and trace remain sealed. Record:

- `prior_runtime_output_open_count=0`;
- `rejected_checkpoint_open_count=0`.

V8 Phase A may open only the exact inherited V7 authorized development
inputs and source/authority metadata. Pose, depth, odometry, optical-flow
labels, occupancy, traversability, physical labels, navigation labels,
scene-family features, held-out inputs, sealed inputs, refinement,
backfill, rebalancing, filtering, resampling, and new renders remain
forbidden. The general raw V13 frame loader, camera-supervision arrays,
probability-calibration data, and every unauthorized role remain at zero
opens during Phase A.

Receipts must additionally bind the V8 preregistration, exact reviewed V8
source manifest/review/authorization, the new loss coefficient, update-zero
NLL baseline, each checkpoint NLL and gate result, and the observation-only
top-1/per-action metrics plus gated macro balanced accuracy. They must retain
every V7 operation count, consumed-input rehash, role, schedule prefix,
forbidden-access count, determinism, Phase-B entry, inventory,
canonicalization, publication, and sealing attestation.

Any newly written V8 checkpoint or trace is a sealed output. It may be
written once, receipt-bound, and sealed, but must never be reopened or
independently hashed by terminal auditors. Auditors may inspect only its
declared filename, byte count, and mode.

Terminal files must be sealed mode `0444` and terminal directories mode
`0555`, with the exact inventory receipt-bound as inherited from V7.

## Attempt and interpretation boundary

This mechanism receives exactly one fresh attempt:

- first decision: `100` updates / `1600` presentations;
- second decision: `400` updates / `6400` presentations;
- maximum Phase A: `1000` updates / `16000` presentations /
  `60` GPU-active minutes;
- conditional Phase B only after the exact final Phase-A conjunction;
- unchanged cumulative cap if Phase B is entered: `2000` updates /
  `32000` presentations / `120` GPU-active minutes.

There is no retry, resume, replacement, second seed, schedule extension,
observer rerun, loss-weight edit, temperature, margin, threshold relaxation,
new head, rejected-checkpoint access, or automatic V9. A V8 failure closes
this local-correspondence transport branch. Any later perception experiment
must use a materially different mechanism and receive a new preregistration.

A Phase-A pass would establish only that explicit all-candidate
correspondence identification made the RGB JEPA's local action-conditioned
transport satisfy the complete frozen development-maze perception
conjunction. It would not establish physical utility, navigation, or
held-out-maze generalization. Those claims retain the unchanged physical
evidence gate and separately authorized G2-to-G8 and sealed-held-out
sequence.

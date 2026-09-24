# Go2 dense V-JEPA 2.1 physical-interface ceiling V1

**Frozen:** 2026-08-03, before implementation, before evaluation-role V-JEPA
feature extraction, and before this mechanism opens either token cache.

## 1. Purpose and decision boundary

This is one development-only representation/interface qualification. It asks:

> Can a fixed low-capacity, spatially shared readout extract scene-disjoint
> physical action-ranking information from the **actual action-matched future**
> dense V-JEPA 2.1 tokens, beyond task/action, retained-physical, current-state,
> persistence, wrong-scene, and action-conditioned mean controls?

It is not a dynamics-model or navigation experiment. It trains no JEPA,
action-conditioned successor predictor, actor, critic, reward model, planner,
memory, or policy. Actual future tokens are a privileged ceiling input and are
never available to a deployed planner. A pass establishes only that this
specific frozen representation and readout interface has physical headroom
worth testing in a separately preregistered predictor experiment. It does not
authorize that experiment.

This branch follows new user direction after the terminal physical-outcome
screen. It is not authority inherited from any predecessor result. It does not
retry or relabel any stopped mechanism:

- the matched-successor frozen-feature and RSSM-style screen remains stopped;
- the dense V-JEPA horizon diagnostic remains stopped at update 1,600;
- the residual token-adapter and compact retained-input physical routes remain
  stopped;
- the learned-RGB V20--V22 and recurrent temporal JEPA routes remain terminal
  scientific failures; and
- the completed frozen-DINO physical-interface routes remain stopped and are
  report-only historical comparators.

There is exactly one attempt in a new namespace. No result-dependent change to
the representation, PCA, scorer, control, seed, schedule, threshold, role, or
data is permitted. A scientific failure closes this frozen single-frame
V-JEPA/readout route. An infrastructure failure produces no scientific
decision and grants no retry or replacement authority.

## 2. Development roles and access scope

Use the existing bounded matched-branch development bundle only:

- 128 train states from 16 scenes;
- 128 evaluation states from 16 disjoint scenes;
- exactly two scenes in each of eight families per role;
- exactly nine executed branches per state; and
- exactly three current-context RGB artifacts plus nine successor artifacts
  per state, or 1,536 artifacts per role.

The fixed role-plan identities are:

- train: `f6f94cf589ec44324fdefe0939aa7076e25543d984464d5b264a0b2f0ff9535b`;
- evaluation: `5dbf9733fd245caff27ce5c5c2b3dc90a3fe9ca9e1bc894dc10a97d64dad9231`;
- combined: `99e60638634eff6ac244cff023cd2ae8f7aa0c53326263ba7a36fa6847386375`.

The evaluation role is already development-exposed through the DINO and
physical-outcome screens. Scene disjointness remains mandatory, but no result
may be described as fresh confirmation, held-out evidence, or formal G2--G8
evidence. There is no cross-fitting: the readouts fit once on all 16 train
scenes and evaluate once on all 16 disjoint evaluation scenes.

The run may:

1. rehash and load the already-produced train V-JEPA cache;
2. open each bound evaluation RGB artifact exactly once through the reviewed
   decoded-pixel reader;
3. execute the bound frozen V-JEPA 2.1 encoder only to create one evaluation
   cache; and
4. train and evaluate the readouts described below.

It may not generate scenes or trajectories, open train RGB, execute any other
encoder, process the passive 3 TB pool, access protected/held-out/sealed roles,
reuse a predecessor predictor or physical checkpoint, retry, resume, extend,
or create a replacement attempt.

## 3. Fixed inputs and predecessor lineage

The scientific input set is closed exactly here. The execution authority must
bind every listed file by the same absolute path, SHA-256, and byte count; it
may not add another scientific input. Every table path beginning with `docs/`
or `.generated/` is resolved under the exact workspace root
`/home/andrewknowles/Workspace/LeWMQuad-v3`.

| input | exact path | SHA-256 | bytes |
|---|---|---|---:|
| Posthoc manifest | `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/manifest.json` | `87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e` | 11,964 |
| Posthoc terminal | `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/terminal.json` | `a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56` | 1,250 |
| Posthoc review | `docs/lewm_go2_world_model_bounded_branch_posthoc_join_admission_v1_terminal_review_2026-08-02.json` | `bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669` | 2,844 |
| RGB manifest | `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/rgb_manifest.json` | `5e03afa7665ffef54a1cab5e37135a18d42761bc844ecefacaa433f75a1b1f7e` | 1,880,307 |
| Train rows | `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/train.jsonl` | `edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447` | 30,432,624 |
| Evaluation rows | `.generated/dev/lewm-go2-wm-bounded-branch-posthoc-join-admission-v1/eval.jsonl` | `531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768` | 30,411,588 |
| Stored task-relevance result | `docs/lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_result_v1_2026-08-02.json` | `5094104ac29b4652cd577015c5fbf23b42f0768c78a205cbf07a77d992339ca7` | 94,165 |
| Stored task-relevance review | `docs/lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_independent_review_v1_2026-08-02.json` | `29eb00a486604824effb56502194855553f87c81a9691d4075a5810273c92ca9` | 2,080 |
| Physics result | `.generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1/physics_result.json` | `25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314` | 183,320 |
| Physics receipt check | `.generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1/physics_receipt_check.json` | `faeb50293bc684e35b6d725b027983ad0110e739db2d7b1aca1926e89a547dc6` | 892 |
| Consumed collection terminal | `.generated/dev/lewm-go2-wm-bounded-branch-experiment-integrity-replacement-v1/terminal_supervision.json` | `f7d2796139645892d22ad6bb99d26caffc2b5c3dcac2a655b1883b299d22bff4` | 12,949 |
| Authorized collection plan | `docs/lewm_go2_world_model_bounded_branch_integrity_replacement_v1_exact_plan_2026-08-02.json` | `8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef` | 343,973 |
| Calibration receipt | `.generated/dev/lewm-go2-wm-counterfactual-calibration-v3-textured-v03-posthoc-analysis-v1/calibration_receipt.json` | `58d1291ede7ee03a93d68eb7cec80c9322c47cd0b1d5fd1c41bf8f4b49ad484e` | 72,475 |
| Train V-JEPA cache | `.generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/features/vjepa2_1.pt` | `3549855ea857906dfe3a4b55fc817633b5114b2457f8facaa4fa87f9eddd798b` | 604,097,648 |
| Train V-JEPA receipt | `.generated/dev/go2_matched_branch_successor_screen_v1/attempt_v1/features/vjepa2_1.json` | `5d4f8a82d10a33c21b41f1543d6f56b3a230a38f67b02d3f8e7330a8d30180f5` | 1,822 |
| V-JEPA checkpoint | `/home/andrewknowles/.cache/vjepa2_1_vitb_dist_vitG_384.pt` | `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` | 1,664,223,428 |
| Physical-screen evaluation | `.generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1/evaluation.json` | `4320b80b20a1f347b1dbc6a7c026bb868820db21edbdcf1053470a400e19cec1` | 1,755,424 |
| Physical-screen result | `.generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1/result.json` | `a2ba2c3ca7881af54b3553b342b36ea72e3f7ca9b858a5eef4102ae9f7b643ee` | 1,769,042 |
| Physical-screen terminal | `.generated/dev/go2_matched_branch_physical_outcome_screen_v1/attempt_v2_integrity_replacement_v1/terminal.json` | `6eb16ea5fa3f9f1e6090eeddc47aace7dd5b9fee7807a56ed84bc7aa0fba2830` | 642 |
| Physical-screen terminal review | `docs/lewm_go2_matched_branch_physical_outcome_screen_integrity_replacement_v1_terminal_review_2026-08-03.json` | `d3f2d99c1a7f7d4e6d02215f04209732f326651e10bd06d040418cc7aafc5cbe` | 22,378 |

The only additional byte inputs are two closed sets already enumerated by the
bound documents above: (a) the exact 256 state-receipt files named by the fixed
train/evaluation rows and validated by the physics result/check, and (b) the
exact 1,536 evaluation RGB leaves named by the bound RGB manifest and fixed
evaluation role plan. Every member is rehashed and path-checked before use; no
other receipt or RGB leaf is admissible. The reviewed reader must reject
symlinks and path escapes.

The authority must also bind these immutable terminal predecessors:

| predecessor path | SHA-256 | bytes |
|---|---|---:|
| `docs/lewm_go2_matched_branch_successor_screen_v1_terminal_review_2026-08-03.json` | `c450baab14b50caed3469fa88f5812c92c02b04676059568e8dae3dc2e5bad83` | 4,991 |
| `docs/lewm_go2_dense_vjepa2_1_horizon_diagnostic_v1_terminal_review_2026-08-03.json` | `0751a9c2d6d2d7d7131ca32f3d3fdc5b4aa9740632fd9a84a51f5e87b82ee1cd` | 4,913 |
| `docs/lewm_go2_dinov2_physical_readout_calibration_integrity_replacement_v1_terminal_review_2026-08-03.json` | `7074779bdc506548d903c0319b74243f2b2934a1888325f813ee52f5a115c679` | 14,382 |
| `docs/lewm_go2_dinov2_dense_shared_spatial_readout_calibration_v1_terminal_review_2026-08-03.json` | `f6ed2d09a407a4cf70097eaa4b2dcffd223e598e4eb59cf8e751997459384020` | 27,120 |
| `docs/lewm_go2_dual_residual_token_adapter_jepa_v1_terminal_review_2026-08-03.json` | `365ab4057bfc51fe9d1b0bd3e7dd415bbddcde9adf89a3ac7674f34b2bc5f1fd` | 9,116 |
| `docs/lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v20_scientific_result_2026-07-30.json` | `d76fd16732d15b7637bbe8f68df65ba23990046812f4ec3d85297f7f8ea64956` | 17,166 |
| `docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21_scientific_result_2026-07-30.json` | `c9544055b11d162b5b5fc9b02d0a04f3961a61b4547411964812a9ae4c5da1e7` | 15,724 |
| `docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_scientific_result_2026-07-30.json` | `1f4896e8f0ae8cadbf09e6f6f34417f3fa6362f9321cfd5abd0aeb09735453d0` | 18,445 |
| `docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_scientific_result_2026-07-31.json` | `180b348449ef16326cd797087a85251037f1fbd6f722b141f35f72aa3f57821c` | 8,843 |
| `docs/lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1_scientific_result_independent_review_2026-07-31.json` | `fe630b86a3ba2b07224e44f4734f0d187294ef616bcda9d8224e8c5fe41ff473` | 3,099 |

These ten predecessor documents are lineage witnesses, not model inputs. No
predecessor checkpoint is a scientific input. The original failed physical
checkpoint remains permanently opaque and may be hash-checked only if required
by lineage validation.

## 4. Frozen V-JEPA extraction

The train receipt fixes the encoder interface. Evaluation extraction must be
identical:

- decode bound PNG as RGB, source size 224 by 224;
- resize with PIL bilinear to 438 by 438;
- center-crop to 384 by 384;
- apply ImageNet mean `(0.485,0.456,0.406)` and standard deviation
  `(0.229,0.224,0.225)`;
- present one image frame as `[3,1,384,384]`;
- use the frozen `vjepa2_1_vit_base_384` EMA encoder;
- area-resample the 24 by 24 output grid to 16 by 16;
- L2-normalize each token in float32; and
- store `[1536,256,768]` as float16 in exact evaluation artifact order.

The cache receipt must record binding, artifact order, encoder source,
preprocessing, decoded-pixel count, elapsed time, throughput, and peak GPU
allocation. No evaluation target, action, family, scene, or physical label may
enter encoder execution.

Independent source review must bind the encoder implementation as code, not
only by checkpoint: exact repository path
`/home/andrewknowles/.cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812`,
clean commit `204698b45b3712590f06245fbfba32d3be539812`, the exact transitive model
construction/configuration source paths and hashes, the reviewed `drop_path`
import shim, PNG reader, preprocessing implementation, 24-to-16 resampler,
normalizer, and extractor. The execution authority must bind that reviewed
source closure and the actual Python, Torch, HIP/ROCm, NumPy, Pillow, and timm
versions. A mismatch from the train receipt's preprocessing/tensor contract is
infrastructure failure.

## 5. Train-only PCA and exact scorer

Fit PCA only from the train role's 128 last-context grids and 1,152 actual
successor grids: 1,280 grids or 327,680 patch rows in fixed role/state/action/
patch order. Promote 768-wide float16 tokens to float64. Compute the population
covariance, use `numpy.linalg.eigh`, sort by descending eigenvalue with original
ascending index as exact-tie break, retain `K=8`, fix each component sign by its
largest-absolute smallest-index loading, and whiten by
`sqrt(max(eigenvalue,1e-12))`. Bind the mean, eigenvalues, signed basis, scales,
source order, and implementation in the PCA identity.

Reuse the unchanged 245-parameter
`DenseSharedSpatialReadoutV1`. For patch `i`, form
`r_i=[z_current,z_successor,z_successor-z_current]` in 24 dimensions and
condition on
`[goal_x/10,goal_y/10,requested_vx/0.30,requested_wz/0.45]`.
Every patch participates through the fixed shared attention/bilinear scorer.

Fit the exact task/action-only ridge first. Its identity must be
`69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a`
and its evaluation regret must be exactly `0.17441406250000002`. The dense
scorer predicts only residual normalized physical rank over that base.
Physical ranks are training targets and evaluation metrics; they never enter
V-JEPA or PCA inputs.

For true-future and capacity-matched current-state scorers use:

- seeds `2026080303`, `2026080304`, and `2026080305`;
- matched CPU initial states and state-order generators per seed;
- float32 parameters and inputs on the authorized ROCm device;
- AdamW, learning rate `1e-3`, weight decay `1e-2`, betas `(0.9,0.999)`,
  epsilon `1e-8`, no AMSGrad/foreach/fused path;
- gradient-norm clipping at `1.0`;
- 256 epochs, eight complete-state batches of 16 states, exactly 2,048 updates
  per member; and
- deterministic algorithms, no validation monitoring, early stopping,
  checkpoint selection, seed selection, or retry.

Ensemble scores are the arithmetic mean of the three member scores before
action selection.

## 6. Fixed arms and controls

1. `privileged_physical_oracle`: exact physical dense ranks.
2. `dense_vjepa_true_future`: task/action base plus the true-future ensemble
   using the actual action-matched successor grid.
3. `dense_vjepa_current_state`: independently fitted capacity-matched ensemble
   trained and evaluated with `[z_c,z_c,0]`.
4. `dense_vjepa_relational_persistence`: the true-future ensemble evaluated
   with `[z_c,z_c,0]`, without refitting.
5. `dense_vjepa_same_action_wrong_scene`: the true-future ensemble evaluated
   with the successor from the other evaluation scene in the same family,
   same state ordinal within each scene's fixed evaluation role-plan order,
   and same action. The source current state, goal, and requested action remain
   unchanged. Each scene must contain exactly eight states, each ordinal must
   have one unique counterpart, and the two scenes in each family swap;
   otherwise execution fails as infrastructure-invalid.
6. `dense_vjepa_train_action_mean_innovation`: the true-future ensemble
   evaluated with `z_s=z_c+mean_train_delta[action,patch]`, where the mean
   projected successor-minus-current delta is computed over all 128 train
   states for that action and patch only.
7. `task_action_only`: the exact fixed ridge control.
8. `retained_physical_predecessor`: the exact published per-state evaluation
   rows for the odometry-and-command-history ensemble; no checkpoint is loaded
   and no model is refit.
9. `hold_constant`: fixed action 6, report-only.
10. `random_expected`: exact uniform expectation, not a sample.

The two completed frozen-DINO physical-interface results are report-only
historical comparators. They cannot gate, select, or alter this mechanism.

## 7. Metrics, uncertainty, and gates

Select the lowest predicted score with action-ID tie-breaking. Report normalized
physical rank regret, oracle-equivalent selection, target progress, path length,
action histograms, per-seed scores, attention entropy, finiteness, and all
per-family/per-scene summaries. Zero fall/tip support remains
`NOT_TESTABLE_ZERO_EVENT_SUPPORT`, not a safety pass.

Every paired comparison uses state differences, scene as the resampling unit,
equal weight across the eight families, 10,000 resamples, seed `2026080314`,
and percentile 95% intervals. Nine actions, 128 states, and 256 patches are not
independent uncertainty units. Negative differences favor true-future V-JEPA.

Qualification requires every gate:

1. source, authority, role, cache, artifact-order, RGB-accounting, finiteness,
   and no-protected-access checks pass;
2. the privileged oracle has regret `0.0` and oracle-equivalent rate `1.0`;
3. upper95(`true_future - task_action_only`) is strictly below zero;
4. upper95(`true_future - retained_physical_predecessor`) is strictly below
   zero;
5. upper95(`true_future - current_state`) is strictly below zero;
6. upper95(`true_future - relational_persistence`) is strictly below zero;
7. upper95(`true_future - same_action_wrong_scene`) is strictly below zero;
8. upper95(`true_future - train_action_mean_innovation`) is strictly below
   zero;
9. true-future point regret is below exact random expected regret; and
10. a fresh-process cache-only replay rehashes both caches, rebuilds PCA,
    reinitializes and retrains all six networks, and exactly reproduces PCA,
    train-action means, task/model states, scores, selections, summaries,
    intervals, gates, and verdict.

These are zero-effect superiority boundaries tied to physical controls, not
new round-number thresholds. Loss, MSE, a favorable point estimate, one seed,
one family, or one attention map cannot override a failed gate.

## 8. Lifecycle and output contract

Implementation starts only after this preregistration is committed. The exact
new source/test closure then receives an independent review. A separate
post-review execution authority must bind the preregistration, reviewed commit,
every recursive source/test dependency, every input, runtime/hardware, fixed
configuration, and an absent root:

`.generated/dev/go2_dense_vjepa2_1_physical_interface_ceiling_v1/attempt_v1`

The authority may permit one complete extraction/fit/evaluation/replay only.
Its access contract must set evaluation RGB access and frozen V-JEPA 2.1
encoder execution to true for the primary process only, and require exactly
1,536 decoded and encoded evaluation frames: 384 current-context frames and
1,152 successor frames. It must set train RGB, every other encoder, a second
or repeated extraction, replay RGB/extraction/encoder execution, collection,
protected, held-out, sealed, retry, resume, extension, replacement, and
downstream-successor execution to false.

The absent attempt root must be atomically reserved and the attempt consumed
before the first RGB decode or either token-cache deserialization. The runner
must terminalize exact access counts. The reviewed manifest/RGB readers must
reject symlinks, path escapes, unexpected roles, and artifact-order changes.

The exact successful-attempt inventory is:

1. `reservation.json`;
2. `vjepa2_1_eval.pt`;
3. `vjepa2_1_eval.json`;
4. `ceiling_checkpoint.pt`;
5. `evaluation.json`;
6. `replay.json`;
7. `result.json`;
8. `terminal.json`.

The checkpoint must be written before evaluation results are published. Replay
must use only the two bound caches and may not reopen RGB or re-execute the
encoder. Any exception writes a failure terminal when possible, publishes no
partial scientific verdict, and grants no retry.

## 9. Terminal decisions

- `QUALIFY_VJEPA_DENSE_INTERFACE_FOR_SEPARATE_BACKBONE_LEVEL_MATCHED_BRANCH_JEPA_PREREGISTRATION`
  only when all ten gates pass;
- `STOP_FROZEN_VJEPA_PHYSICAL_INTERFACE_NOT_ESTABLISHED` when execution is
  scientifically complete and any scientific gate fails; or
- `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION` for any incomplete or invalid
  execution.

A qualification status authorizes only writing and reviewing a separate
proposal. It does not authorize model training, new data, planner integration,
navigation, G2--G8 work, or deployment. A STOP prohibits threshold relaxation,
readout variants, extra seeds/epochs, cache variants, or a 12-member V-JEPA/
DINO campaign on this interface.

# Shared V5 Camera V6 hard-raster diagnostic V1 preregistration

Date: 2026-07-23

Status: **source-free preregistration; implementation and execution not yet
authorized**

## User decision and narrow question

The user explicitly directed: **“Authorize the single zero-training raster
diagnostic.”**

This document freezes exactly one development-only diagnostic answering one
question: does the already-fixed soft evidence decoding and rasterization
stage discard a large amount of usable Camera V6 ray and ground evidence?

This is not training. It permits no optimizer, backward pass, gradient, EMA
update, parameter mutation, calibration, threshold search, data change,
checkpoint promotion, G2, navigation, runtime, production, or held-out access.
Even a diagnostic PASS cannot qualify Camera V6. It can justify only a
separate user decision about one structural evidence-decoding/rasterization
successor.

The output root is fixed as
`.generated/go2_shared_observable_camera_ray_jepa_v5/camera_v6_hard_raster_diagnostic_v1`.
It is presently absent. A later, separately reviewed implementation may
reserve it exactly once only after a separate one-attempt execution
authorization. Reservation consumes the attempt. The root may not be deleted,
reused, resumed, retried, or repaired.

## Immutable predecessor evidence

The rejected Camera V6 terminal audit is:

- path:
  `docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_terminal_audit_2026-07-23.json`;
- commit: `f1c4e2efe948165004512ccc1882e721d8626d0b`;
- file SHA-256:
  `367dd08f9a039710d61efd9ecb652134f6efbd056e126c4a51d67929f28b06b7`;
- canonical content SHA-256:
  `76727ada6442774412508b0ca96b1a50b5170bc75867235aecc132f28d1ac892`.

The bounded architecture postmortem is:

- path:
  `docs/lewm_go2_shared_jepa_v5_camera_v6_bounded_architecture_postmortem_2026-07-23.md`;
- commit: `6a67ad77905b44e8a40fa5eef3f8ca7656db349b`;
- file SHA-256:
  `7f5ca06e773c61b24fe792f210c38204066c35ee7ebd496e5a75174b9933d0b1`.

The sole model input is the rejected update-8000 checkpoint, bound through its
immutable V6 metric sidecar:

| Artifact | Bytes | File SHA-256 | Canonical content/state SHA-256 |
|---|---:|---|---|
| `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k/checkpoints/update_8000.metrics.json` | 15,365 | `c03bc02f5c45ad8b2de0042bdb4602fe03c88ad52c2ac5b77375d9e6f956d2dc` | content `7437bfee92f2b9fe9d77fd8acce1612c53ebf17c7d839786cff6f94f691bb3ee` |
| `.generated/go2_shared_observable_camera_ray_jepa_v5/protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k/checkpoints/update_8000.pt` | 29,466,305 | `01871a6495cd6ffa6cdcc97f1451014e887ac9a219360bb69ae0a866db3db20c` | content `4d20f50a688efd617f31ac092a5f7019084afb67e99a064029907222a61be120`; state `960854245db49a048e3a99e91b08d6746795f8c1abd52a267f592900259eee22` |

The checkpoint binding also records frozen-state SHA-256
`3f5cce294f840be4c6c8cfa43b2818bae68da739b13348fe45a3d5087fe2524e`
and trainable-state SHA-256
`6b01b16355d940133b6683b420f2e4f182d0535264aef595a97727d813919e96`.
No checkpoint may be opened during implementation or source review. The one
authorized execution may deserialize this checkpoint exactly once with
`weights_only=True`, validate all bindings, load it strictly into one
evaluation-mode model, and perform forward inference only.

## Exact development population

Only the existing `checkpoint_selection` role is permitted:

- 495 pairs from eight development scenes;
- 924 unique endpoints;
- the exact per-family sorted endpoint order;
- the exact cyclic-plus-one within-family wrong-RGB mapping;
- the target endpoint's calibration, ground plane, and supervision for both
  the matched- and wrong-RGB arms.

The development raw manifest is
`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1/manifest.json`,
with 311,598 bytes, file SHA-256
`e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360`,
and content SHA-256
`74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a`.
Its audit receipt is
`.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v13.json`,
with 26,975 bytes, file SHA-256
`0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76`,
and content SHA-256
`0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca`.

The `train` and `probability_calibration` roles are denied. Source raw,
simulator state, collision geometry, morphology, configuration labels,
navigation data, production data, and held-out data are denied. No input,
decoded RGB, raw model output, or per-frame prediction may be written to the
diagnostic root.

## One fixed hard-evidence adapter

Each matched or wrong RGB is forwarded once. The same unchanged raw output is
used for the existing soft raster, all direct metrics, and this single hard
adapter. There is no second inference pass and no alternative adapter.

For each raw output:

1. Compute the existing ordered 64-bin first-hit log probabilities from the
   unchanged hazard logits.
2. A ray has a finite hit exactly when
   `-expm1(ordered.no_hit) >= 0.5`.
3. Select `argmax(ordered.hit)` across the 64 finite-hit bins. A tie selects
   the lowest index, as in the existing tensor `argmax`.
4. For a finite hit, use depth
   `0.05 + (bin_index + 0.5) * 0.10 + selected_existing_offset`.
   Do not clamp, average, recalibrate, or modify the offset. A no-hit ray has
   canonical distance zero and contributes no occupied evidence.
5. A ground support is clear exactly when its unchanged query is in-frustum
   and its unchanged ground logit is `>= 0`.
6. A 0.05 m source cell is free only when all five supports are in-frustum
   and clear. A 0.10 m output cell is free-before-occupancy only when all four
   constituent source cells are free.
7. Construct the existing `ObservableCameraRayEvidenceV4` from those Boolean
   decisions, the predicted finite-hit depths, and the target endpoint's
   unchanged float32 camera origin, float32 camera basis, and ground-plane
   height.
8. Invoke the existing
   `rasterize_observable_camera_ray_evidence_v4` hard rasterizer. It performs
   calibrated float64 ray projection, uses the existing `2e-5 m`
   closed-boundary tolerance and boundary supercover, ignores hits outside the
   output extent, and Boolean-ORs all retained hits.
9. Class precedence is occupied, then free, then unknown.

Any nonfinite tensor, depth at or behind the registered near plane for a
declared hit, malformed calibration, source-contract mismatch, or unsupported
class is an integrity failure. The hard labels do not define a probabilistic
NLL; hard-raster NLL is excluded.

## Metrics and immutable baselines

For both matched and cyclic wrong RGB, publish the integer 3x3 confusion
matrix with target classes as rows and predicted classes as columns for all
nine scopes. Derive each present-class recall from that matrix and define
raster balanced accuracy as the arithmetic mean of the recalls for target
classes present in the scope, exactly matching the existing metric
convention. Publish matched balanced accuracy, wrong balanced accuracy, and
their difference. Use Python binary64 arithmetic and do not round before any
comparison.

The eight non-rough materiality scopes are intentionally the aggregate plus
the seven non-rough families. The aggregate overlaps the families and still
counts as one of the fixed eight:

| Scope | Immutable soft raster balanced accuracy | Immutable soft matched-minus-wrong raster balanced-accuracy drop |
|---|---:|---:|
| `aggregate` | 0.9009460724448773 | 0.2664251275669607 |
| `large_enclosed_maze` | 0.8187028299574806 | 0.3683291323380902 |
| `local_composite_motifs` | 0.8887728118230923 | 0.40648658282717515 |
| `loop_alias_stress` | 0.817520212817799 | 0.38180176978110736 |
| `medium_enclosed_maze` | 0.8671429422192141 | 0.40028310519068205 |
| `open_obstacle_field` | 0.9085505950468774 | 0.30614686430029825 |
| `small_enclosed_maze` | 0.8602078804715946 | 0.37484879826098155 |
| `visual_sensor_stress` | 0.8675719422139417 | 0.4418144494003426 |

`rough_local_dynamics` is evaluated and published but excluded from
materiality. The immutable aggregate soft recall baselines are free
`0.91637020862468` and occupied `0.8059679976935274`.

Before interpreting the hard result, the execution must reproduce the
immutable update-8000 sidecar. Require exact deep equality, with zero
tolerance, for every scope's existing physical metric projection, including
the soft raster metrics and:

- pixel first-hit balanced accuracy;
- depth median and p95 absolute error;
- ground-clear balanced accuracy and all six distance-group balanced
  accuracies;
- wrong-RGB pixel balanced-accuracy drop;
- wrong-RGB depth median and p95 error increases;
- wrong-RGB ground balanced-accuracy drop.

The same raw outputs must also yield exactly equal direct-metric projections
before and after hard adaptation. A mismatch is an integrity/reproduction
failure, not evidence for or against the scientific hypothesis.

## Fixed decision

Decision precedence is:

1. any binding, access, source-closure, model-state, finite-value,
   reproduction, direct-metric, or publication failure:
   `FAIL_INTEGRITY`;
2. otherwise the materiality conjunction below is either
   `PASS_MATERIAL_HARD_RASTER_LOCALIZATION` or
   `FAIL_HYPOTHESIS_REJECTED`.

The materiality conjunction passes only when all of the following are true:

- `hard_BA - immutable_soft_BA >= 0.05` in at least six of the eight
  enumerated non-rough scopes;
- aggregate hard free recall minus `0.91637020862468` is `>= 0.05`;
- aggregate hard occupied recall minus `0.8059679976935274` is `>= 0.05`;
- hard matched-minus-wrong raster balanced-accuracy drop is `>= 0.12` in
  every one of the eight non-rough scopes.

Equality passes. These are absolute metric-point differences. There is no
rounding, tolerance, confidence interval, averaging across scopes, relative
gain, normalized margin, “close enough,” post-result reinterpretation, or
fallback criterion.

## One-attempt lifecycle and denials

Implementation may begin only from this committed preregistration. It must be
covered by deterministic CPU/synthetic tests that open no experiment
checkpoint, dataset, or RGB. A source author handoff must bind every
executable source and test by SHA-256. A distinct reviewer must confirm exact
source closure and this contract without opening experiment inputs. A
distinct execution authorizer must then bind the committed preregistration,
review, sources, immutable inputs, one fixed command, one output root, and a
no-input visibility preflight.

The authorized process must reserve the absent root before any experiment
input is opened, record every permitted read, write only immutable canonical
reservation/access/result-or-failure artifacts, and terminate. Result
publication consumes the diagnostic regardless of PASS or failure.

No retry, resume, repair, threshold change, second hard decoder, training run,
successor implementation, checkpoint promotion, G2, navigation, runtime,
production, or held-out action follows automatically. This preregistration
grants none of those actions.

# RGB Swept-Progress Survival Joint-JEPA V4 — Physical-Evidence Calibration Preregistration

- Status: frozen before runner implementation and before any calibration/selection payload access.
- Purpose: determine whether the admitted, jointly trained V4 candidate can be converted into conservative current-camera physical evidence using one deterministic development-fitted probability calibration and fixed thresholds. This is inference-time calibration, not encoder/predictor retraining, checkpoint selection, or G2 evaluation.
- Candidate-admission result commit: `5f0697361edd81f9dea59be9ef5c635975011c84`.
- Calibration dependency closure: `9a60be5e2b557124c11654669fd884e50639ab76`.

## Exact candidate and source inputs

- Read only the admitted candidate receipt and candidate checkpoint under `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_candidate_admission/attempt_v1`.
- Candidate receipt file/content SHA-256: `7b21e9a908c05f56c344a74682ee0a3d912c449920d57ee9298619f53c9f66f1` / `247e9f1d81cb143631c4be4b85173f707516ff5cf32a0e9e08ca6d8100420f8f`.
- Candidate checkpoint byte count / SHA-256: `25,673,535` / `f8a330d1a4834e4cc61f7acae00069f866a37a5693464e6fbb93b998a971d37a`.
- Frozen adapter / calibrator / traversability-metrics SHA-256: `1ddbfd743d89614932823ae2247534ac6a76e2eaaf031911617a9311562b4b58` / `2a41a69d4bf981415f3c3ae6c437e78b3c07e781a603602f7ca58e4e6f785f2b` / `97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396`.
- Frozen raw contract / narrow loader / raw-input constructor / raw-input implementation SHA-256: `79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6` / `33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000` / `ce256dcb1ef67dff313855680365ce07d867aca986dfcad7b8e9493373fe099c` / `e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578`.
- Bound raw manifest: `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1/manifest.json`, 311,598 bytes, file/content SHA-256 `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360` / `74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a`.
- Bound raw audit: `.generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1.audit_v13.json`, 26,975 bytes, file/content SHA-256 `0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76` / `0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca`.
- The manifest binds the pair/endpoint indexes and every permitted RGB/shard/`raster_labels.u1` payload. No general frame loader or any other supervision array is allowed.

## Exact development populations

- The raw constructor may validate the shared development manifest, audit, pair index, and endpoint index. Payload requests are then restricted to `probability_calibration` and `checkpoint_selection`; no train-role RGB or raster payload is allowed.
- Preserve raw pair order and use only each pair's `next_endpoint_sha256`, with no deduplication or remapping.
- `probability_calibration`: exactly 415 rows from 8 scenes, producing 1,699,840 raster cells.
- `checkpoint_selection`: exactly 495 rows from 8 scenes, producing 2,027,520 raster cells.
- For every row, load only normalized RGB `[3,112,112]` and the complete physical UNKNOWN/FREE/OCCUPIED `raster_labels.u1` target `[64,64]`. All raster cells participate; UNKNOWN is a real target and the model anchor mask is not an evaluation mask.
- The fixed physical grid centers are forward `linspace(-0.95,5.35,64)` metres and left `linspace(-3.15,3.15,64)` metres. Euclidean cell distance is used for the fixed 2.0-metre obstacle range.

## Frozen calibration and threshold procedure

- Load the candidate once through the strict V4 adapter, keep it frozen, and batch only online RGB encoding plus the V4 semantic decoder. The predictor was already validated during admission and is not recomputed for this calibration-only operation.
- Fit exactly one global `lewm_go2_hierarchical_probability_calibration_v1` artifact on all 1,699,840 calibration cells in input order with `maximum_iterations=80`, ECE bins 15, CPU float64 deterministic LBFGS, natural empirical priors, and no weighting, balancing, subsampling, duplication, dropping, or class backfill. UNKNOWN, FREE, and OCCUPIED must all have support.
- Validate the fitted artifact and apply its same four parameters to calibration logits. Select thresholds only on `probability_calibration` using the existing deterministic selector and the full registered 2,016-tuple grid:
  - FREE minimum: `(0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99)`.
  - OCCUPIED maximum and UNKNOWN maximum: `(0.01,0.02,0.05,0.10,0.20,0.35)` each.
  - OCCUPIED detection minimum: `(0.01,0.02,0.05,0.10,0.20,0.35,0.50)`.
- Candidate tuples must have disjoint free-admission and occupied-detection thresholds, nonzero admitted cells, physical-free precision at least 0.99, obstacle exclusion within 2m at least 0.95, and occupied detection within 2m at least 0.95. The existing selector's highest-useful-recall tie-break is authoritative.
- Apply the unchanged calibration artifact and selected thresholds once to `checkpoint_selection`; do not refit, retune, or reselect there.

## Development gate and stopping rule

- Pass only if the calibration grid has at least one passing tuple and fixed `checkpoint_selection` metrics satisfy: admitted observable physical-free precision at least 0.99; directly observable occupied detection within 2m at least 0.95; useful observable physical-free recall at least 0.90; and physical obstacle exclusion within 2m at least 0.95.
- Record Brier score, ECE, unknown admission, class counts, calibration before/after metrics, chosen thresholds, all selection metrics, routing `NOT_APPLICABLE`, and exact access receipts regardless of pass/fail.
- A pass authorizes only preparation of a separately reviewed one-shot G2 binding. It does not itself open or pass G2, qualify navigation, promote, deploy, resume, or train anything.
- A scientific failure keeps G2 closed and triggers a mechanism decision, not threshold relaxation on `checkpoint_selection` and not repeated calibration variants.

## Execution and output boundary

- Fresh write-once root: `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence_calibration/attempt_v1`.
- On success or scientific failure, write canonical `calibration.json` and `result.json`; on operational failure, write only canonical `failure.json` and leave any partial outputs unqualified.
- No accelerator requirement, optimizer update to the model, backward through the model, EMA, dataset construction, training, G2, navigation, held-out, sealed, rejected-checkpoint, original-V4-runtime, N320, or production access is authorized.
- Freeze, test, independently review, and execution-bind the runner before its one scientific execution. There is no resume or scientific retry.

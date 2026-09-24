# RGB Swept-Progress Survival Joint-JEPA V9 — Physical-Evidence Calibration Preregistration

- Status: frozen before calibration adapter/runner implementation and before
  any V9 checkpoint or calibration/selection payload access.
- Purpose: determine whether the jointly trained V9 candidate can be converted
  into conservative current-camera physical evidence using the already
  reviewed V4 one-shot development calibration. This is inference-time
  calibration, not retraining, checkpoint selection, G2, or navigation.
- V9 terminal-result commit:
  `8a4f335de08884ec4dcc81325234ee69ce164e63`.
- Protocol authority: the V4 physical-evidence calibration preregistered at
  `e983e0abd9349426f69262563e12d90a4488180e`, with no scientific change.

## Exact candidate and source inputs

- Read only V9 `result.json` and, after that receipt passes every check below,
  its bound `checkpoint_update_1000.pt` under
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift/attempt_v1`.
- Result byte count / file SHA-256 / content SHA-256: `69,002` /
  `698acce34e9221e1660d243133937b621abc6742a5436a859c91b7ffbf55c7e5` /
  `344d10db882314fa3f227597dba4fc7e96747e3fdbe3f6d134e6c7f28c5c2c28`.
- Candidate checkpoint byte count / file SHA-256: `25,427,815` /
  `5456dc94136503543439e4bf691b8120c63c45a04e692f640c9c246f243c5ffd`.
- Before checkpoint access, require the exact V9 result schema, preregistration
  `47043472466e7a258ad0f0be854c05393e233db8`, amendment
  `04db6b26d46875297e3aa515fdf1d688bee2b755`, status
  `PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION`, 24/24 unchanged checks,
  no failed checks, exact terminal accounting, development-only/unqualified
  authority, explicit checkpoint-calibration authorization, and zero G2,
  held-out, sealed, promotion, or retry authority.
- Frozen V9 model source SHA-256:
  `eb5ac85cfe1394b946eddd5f56167066085bfa6598aaa364e15cf432c2228d0c`.
- Frozen V4 reference calibration runner SHA-256:
  `cee7c9c70e6bb9d2bacc6528ef77d009c80e2f484400de9f6445ebfd0c010313`.
  The V9 runner may adapt candidate validation/loading and receipt names only;
  role collection, calibration, threshold selection, gates, and access checks
  must call the exact reference procedures.
- Frozen probability calibrator / traversability metrics SHA-256:
  `2a41a69d4bf981415f3c3ae6c437e78b3c07e781a603602f7ca58e4e6f785f2b` /
  `97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396`.
- Frozen raw contract / narrow loader / raw-input constructor / raw-input
  implementation SHA-256:
  `79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6` /
  `33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000` /
  `ce256dcb1ef67dff313855680365ce07d867aca986dfcad7b8e9493373fe099c` /
  `e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578`.
- Preserve the V4-bound development manifest/audit and their exact permitted
  RGB, pair, endpoint, and `raster_labels.u1` identities. No general frame
  loader or other supervision array is allowed.

## Exact V9 reconstruction

- Load the checkpoint once on CPU with `weights_only=True`. Require the exact
  V9 checkpoint schema and exact top-level key inventory frozen by the V9
  executor; validate all seeds, commits, terminal accounting, development-only
  state, no-resume/no-qualification flags, inherited auxiliary objective,
  seven-tensor/16,576-parameter attention activity, zero target gradients, and
  initial V9 architecture/migration receipts.
- Reconstruct the exact V9 class from its online encoder state and swept masks,
  strict-load the complete state, freeze it in evaluation mode, and require
  bit-exact loaded state plus hard-sync count 1 and EMA count 1,000.
- Calibration inference may call only online RGB encoding and the inherited
  semantic decoder. It must not call the action predictor, mutate model state,
  run backward, update an optimizer, update EMA, or write a new checkpoint.

## Exact development populations

- Preserve raw pair order and use only each pair's `next_endpoint_sha256`, with
  no deduplication or remapping.
- `probability_calibration`: exactly 415 rows from 8 scenes, producing
  1,699,840 cells. `checkpoint_selection`: exactly 495 rows from 8 scenes,
  producing 2,027,520 cells.
- Load only normalized RGB `[3,112,112]` and complete physical
  UNKNOWN/FREE/OCCUPIED `raster_labels.u1` `[64,64]`. All cells participate;
  UNKNOWN remains a target and the anchor mask is not an evaluation mask.
- Preserve fixed grid centers forward `linspace(-0.95,5.35,64)` metres and
  left `linspace(-3.15,3.15,64)` metres, with Euclidean 2.0-metre range.

## Frozen calibration, threshold procedure, and gate

- Fit exactly one global hierarchical probability-calibration artifact on all
  calibration cells in order: maximum 80 iterations, ECE bins 15, CPU float64
  deterministic LBFGS, empirical priors, no weighting/balancing/subsampling/
  duplication/dropping/backfill, and support for all three classes.
- Validate and apply the same four fitted parameters. Select thresholds only
  on `probability_calibration` over the full registered 2,016-tuple grid:
  FREE minimum `(0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99)`;
  OCCUPIED and UNKNOWN maximum `(0.01,0.02,0.05,0.10,0.20,0.35)`;
  OCCUPIED detection minimum `(0.01,0.02,0.05,0.10,0.20,0.35,0.50)`.
- Preserve the V4 feasibility rules and highest-useful-recall tie-break. Apply
  the resulting artifact and thresholds once to `checkpoint_selection`; do not
  refit, retune, reselect, or inspect selection outcomes before the decision.
- Pass only if at least one calibration tuple passes and fixed selection
  metrics meet: physical FREE precision `>=0.99`; occupied detection within 2m
  `>=0.95`; useful physical FREE recall `>=0.90`; obstacle exclusion within 2m
  `>=0.95`.
- Record calibration/selection metrics, thresholds, routing `NOT_APPLICABLE`,
  and exact access receipts on pass or scientific failure.

## Stopping and output boundary

- Fresh write-once root:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_physical_evidence_calibration/attempt_v1`.
- Success or scientific failure writes canonical `calibration.json` and
  `result.json`; operational failure writes only canonical `failure.json` and
  leaves any partial artifact unqualified.
- A pass authorizes only preparation of a separately reviewed one-shot G2
  binding. It does not open or pass G2, qualify navigation, promote, deploy,
  resume, or train. A scientific failure closes V9 without threshold
  relaxation or a second calibration variant.
- No accelerator, training, dataset construction, G2, navigation, held-out,
  sealed, rejected-checkpoint, N320, production, or deployment access is
  authorized. Freeze, test, independently review, and execution-bind the
  adapter/runner before the sole scientific execution; there is no resume or
  scientific retry.

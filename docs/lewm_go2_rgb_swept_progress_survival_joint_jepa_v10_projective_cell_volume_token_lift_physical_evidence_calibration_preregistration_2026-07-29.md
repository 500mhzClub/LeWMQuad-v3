# RGB Swept-Progress Survival Joint-JEPA V10 — Physical-Evidence Calibration Preregistration

- Status: frozen before V10 calibration adapter/runner implementation and
  before V10 checkpoint or calibration/selection payload access.
- Purpose: test whether the jointly trained V10 candidate produces
  conservative current-camera FREE/OCCUPIED evidence under the already
  reviewed V4 one-shot protocol. This is deterministic inference-time
  calibration, not retraining, checkpoint selection, G2, or navigation.
- V10 terminal-result commit:
  `7ccb9cc88f1ddfa687a6d9b5cef847bbb3f11cfe`.
- Protocol authority: V4 physical-evidence calibration preregistration
  `e983e0abd9349426f69262563e12d90a4488180e`; no scientific change.

## Exact candidate

- Read only V10 `result.json` and, after that receipt passes every check, its
  bound `checkpoint_update_1000.pt` under
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift/attempt_v1`.
- Result byte count / file SHA-256 / content SHA-256: `70,550` /
  `f62fa6c908fe8cfb4ae838878d40b615e14ad343d5f123c1dd24e16f274bbb70` /
  `01ce5f55d3b2cc264b21a9924d27e64568873dfaf2a2364e1448991adda0b6b6`.
- Checkpoint byte count / file SHA-256: `29,741,203` /
  `f63a037868de1e4db465fb4f85af2b8e6eba9883880c19d908216db20d82faa0`.
- Before checkpoint access require the exact V10 result schema,
  preregistration `b9eaae6560c42e588c86fb8bf949cc95bd9e29e9`, status
  `PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION`, all 24 unchanged checks,
  exact terminal accounting and attention activity, development-only and
  unqualified authority, explicit calibration access, and zero G2, held-out,
  sealed, promotion, retry, or resume authority.
- Frozen V10 model source SHA-256:
  `68ade72ef4293bd23136ad739c269af360a962c901216c996c2247d494a88196`.

## Frozen source closure and reconstruction

- Preserve this exact reviewed source closure:

| Source | SHA-256 |
|---|---|
| `scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence.py` | `cee7c9c70e6bb9d2bacc6528ef77d009c80e2f484400de9f6445ebfd0c010313` |
| `lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter.py` | `1ddbfd743d89614932823ae2247534ac6a76e2eaaf031911617a9311562b4b58` |
| `lewm/hierarchical_probability_calibration.py` | `2a41a69d4bf981415f3c3ae6c437e78b3c07e781a603602f7ca58e4e6f785f2b` |
| `lewm/benchmarks/traversability_metrics.py` | `97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396` |
| `lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py` | `79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6` |
| `scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py` | `33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000` |
| `scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py` | `ce256dcb1ef67dff313855680365ce07d867aca986dfcad7b8e9493373fe099c` |
| `lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py` | `8c35f0cbafe78185ac74d4412914c177de20f899b0f009a9b9dc7aafdf7695a5` |
| `scripts/run_go2_shared_jepa_v5_matched_training_v1.py` | `e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578` |
| `lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py` | `53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a` |
- The V10 runner may change only candidate validation/loading and V10 receipt
  names. Role collection, calibration, threshold selection, metrics, gates,
  and access checks must call the exact reviewed V4 procedures.
- Load the checkpoint once on CPU with `weights_only=True`; validate its exact
  schema/key inventory, seeds, commits, accounting, objective, V10
  architecture/migration receipt, all-seven-tensor attention activity, target
  zero gradients, and development-only/no-resume/no-qualification flags.
- Reconstruct the exact V10 class from its online encoder state and swept
  masks, strict-load the full state, freeze evaluation mode, and require
  bit-exact loaded state, hard-sync count 1, and EMA count 1,000.
- Inference may call only online RGB encoding and semantic decoding. It must
  not call the predictor, run backward, mutate state, update an optimizer/EMA,
  or write a checkpoint.

## Unchanged population, calibration, and gate

- Preserve raw pair order and use only `next_endpoint_sha256` without
  deduplication/remapping: exactly 415 probability-calibration rows
  (1,699,840 cells) and 495 checkpoint-selection rows (2,027,520 cells), each
  spanning 8 scenes.
- Read only normalized RGB `[3,112,112]` and complete physical
  UNKNOWN/FREE/OCCUPIED `raster_labels.u1` `[64,64]`; all cells participate.
  Use fixed cell centres and the unchanged Euclidean 2 m range.
- Fit exactly one global hierarchical probability calibrator on all
  calibration cells: CPU float64 deterministic LBFGS, at most 80 iterations,
  15 ECE bins, empirical priors, and no balancing, weighting, sampling,
  dropping, duplication, or backfill.
- Select only on calibration over the full frozen 2,016 threshold tuples:
  FREE minimum `(0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99)`; OCCUPIED and
  UNKNOWN maximum `(0.01,0.02,0.05,0.10,0.20,0.35)`; OCCUPIED detection
  minimum `(0.01,0.02,0.05,0.10,0.20,0.35,0.50)`. Preserve V4 feasibility
  and highest-useful-recall tie-breaking.
- Apply the fitted artifact and selected tuple once to selection; do not
  refit, retune, reselect, or inspect selection outcomes before the decision.
- Pass only if calibration has at least one feasible tuple and fixed selection
  metrics meet FREE precision `>=0.99`, occupied detection within 2 m
  `>=0.95`, useful FREE recall `>=0.90`, and obstacle exclusion within 2 m
  `>=0.95`.

## One-shot boundary

- Fresh output root:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence_calibration/attempt_v1`.
- Success or scientific failure writes canonical `calibration.json` and
  `result.json`; operational failure writes `failure.json`. No accelerator,
  training, dataset construction, G2, navigation, held-out, sealed,
  predecessor checkpoint, N320, production, or deployment access is allowed.
- Freeze, test, independently review, and separately execution-bind the
  adapter/runner before exactly one run. A pass earns only preparation of G2;
  a scientific failure closes this V10 candidate without threshold relaxation,
  retry, or a second calibration variant.

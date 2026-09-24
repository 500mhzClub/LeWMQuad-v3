# RGB Swept-Progress Survival Joint-JEPA V4 — Physical-Evidence Calibration Result

- Terminal scientific status: `FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE`.
- Preregistration / source closure / execution binding: `e983e0abd9349426f69262563e12d90a4488180e` / `440ff2ac103025f8dc15c186737b63d1e2519ad8` / `35fc7315e88770a9f1c25b18d14fc93aeea55743`.
- The one scientific command completed once. There was no retry, resume, refit, threshold relaxation, or selection-role tuning.

## Artifact identities

- Result file/content SHA-256: `43d3f28b4ea8412e343799f79a0c880a2bbecddaedb461861089470e8697ae6c` / `84edf6a518b39e948f1c551fc128a16650232cabf55d71833d036c5f83c016b1` (362,462 bytes).
- Calibration file/content SHA-256: `f81771f9c397971094efeb06b4720a0cf3270f55f4e4f0535ef5e538d5a80a11` / `65cdbadb271ee88c9d5932239f097013ccc4cdbff17420f047cd3bc7630a0c17` (4,165 bytes).
- Calibration ID: `go2-hier-cal-65cdbadb271ee88c`.
- Candidate checkpoint remained the admitted 25,673,535-byte artifact with SHA-256 `f8a330d1a4834e4cc61f7acae00069f866a37a5693464e6fbb93b998a971d37a`.
- Independent exact-only artifact audit passed canonical JSON, file/content identities, calibration binding/ID, consumed-ledger digest, gate computation, access counts, and authority.

## Exact populations and procedure

- Calibration: 415 ordered next-frame rows, 26 batches, 1,699,840 cells; endpoint-order SHA-256 `d62cd668b2059b6b274a18becbc176622fa78ad97ae4bb3cf9514ee012cd7747`.
- Selection: 495 ordered next-frame rows, 31 batches, 2,027,520 cells; endpoint-order SHA-256 `eb274c8a5b206ee4d665a63a850141dcc12ad258e0b5e6fe0907274c1af300d4`.
- One deterministic CPU-float64 global calibration fit ran on every calibration cell with no weighting, balancing, subsampling, dropping, duplication, or backfill. All three classes had support.
- The exact 2,016-tuple conservative threshold grid was searched once on calibration. The fitted calibration and resulting thresholds were then applied unchanged to selection.
- The candidate model state did not mutate. The predictor was not recomputed; model backward, optimizer, and EMA counts remained zero.

## What calibration achieved

- Calibration was numerically valid and materially improved aggregate probability quality on its fit role: joint NLL `0.1507451 → 0.0949757`, multiclass Brier `0.0699432 → 0.0405266`, and accuracy `0.9512301 → 0.9737917`.
- The same fixed transform generalized in aggregate to selection: joint NLL `0.1835528 → 0.1067377`, multiclass Brier `0.0899364 → 0.0480697`, and accuracy `0.9365935 → 0.9700131`.
- Fitted unknown-vs-known scale/bias: `0.8322804 / -1.8006297`; fitted free-vs-occupied-given-known scale/bias: `0.7295854 / -1.8965150`.
- Therefore the failure was not an optimizer, artifact, class-support, or gross calibration-generalization failure.

## Why the physical gate failed

- No threshold tuple passed even on `probability_calibration`; `passing_candidate_count=0`.
- The selector consequently returned only its best compatible fallback (`free_min=0.50`, `occupied_max=0.35`, `unknown_max=0.35`, `occupied_detection_min=0.50`). These are diagnostic fallback thresholds, not qualified settings.
- Calibration-role fallback metrics: free precision `0.93154`, occupied detection within 2m `0.29065`, obstacle exclusion within 2m `0.79588`, and useful free recall `0.90160`.
- Selection fallback metrics: free precision `0.92923`, occupied detection within 2m `0.26853`, obstacle exclusion within 2m `0.72147`, and useful free recall `0.85067`.
- All five frozen development checks failed: a passing calibration tuple, selection free precision at least 0.99, selection occupied detection at least 0.95, selection useful free recall at least 0.90, and selection obstacle exclusion at least 0.95.
- Aggregate argmax/NLL quality is therefore not enough for conservative navigation. The current semantic logits do not separate safe-free evidence from occupied evidence strongly enough to obtain both high obstacle recall and high free-space utility at one disjoint operating point.

## Access and authority

- Exact candidate access: one receipt read, one checkpoint read, and one strict checkpoint load.
- Exact model-facing development access: 910 ordered endpoint RGB requests/physical reads and 910 raster-label requests; 16 physical `raster_labels.u1` array opens with 894 subsequent underlying-array cache hits.
- The access ledger binds 946 unique consumed files with canonical records SHA-256 `cd482a728b50af4cf7fd75c7641a59e8d0d27be46f7f5ec2d398a6e9aec3a4e0`.
- Every forbidden semantic counter and the general raw-frame-loader counter remained zero. No train-role payload, predictor, original V4 runtime, N320, G2, navigation, held-out, sealed, rejected-checkpoint, accelerator, model-training, or production operation occurred.
- G2 remains closed and unqualified. The fitted calibration and fallback thresholds are not authorized for G2 or navigation.

## Scientific stopping decision

- Do not retry global calibration, expand the grid, or tune thresholds on `checkpoint_selection`. The preregistered same-logit mechanism is falsified for the physical gate.
- The next step must be a materially different jointly trained perception mechanism that directly improves separability of conservative free admission and high-recall obstacle evidence while retaining the full joint-JEPA predictor path.

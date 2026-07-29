# V9 Dense Local-Token Lift — Physical-Evidence Calibration Result

- Terminal scientific status: `FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE`.
- Preregistration / source-closure amendment / source / execution binding:
  `2f561d26f0b6ca154b6f4eab00dba228f8bc8c9e` /
  `b2465b2148b999b216078d53fe9bd556e63703e0` /
  `2f978d3783223f7aed77355f510c5dead27f7627` /
  `0eb7cfc0cc66ccb8a5b13c10c88ff4fb4d695f86`.
- The one authorized command completed once with registered scientific-failure
  exit `2`. There was no retry, resume, refit, threshold relaxation, or
  selection-role tuning.

## Artifact identities and integrity

- Result: `364,104` bytes; file SHA-256
  `ded100d376abadcd5ce590f3b1d7243e14851cdf0182f42aee0fce4510044805`;
  canonical content SHA-256
  `471cd01f3c377fb436f87e725144fbb3efcf69504325921ce967d159c546e29a`.
- Calibration: `4,159` bytes; file SHA-256
  `09328b00e8773283160677e49cddc52519266e5ff098000dbabba8760b0a8c93`;
  validated semantic content SHA-256
  `0e285dcaedc1bb8e965bdc9ec2523a2fe3161aeb410a06a0e95f37f1050488cc`;
  ID `go2-hier-cal-0e285dcaedc1bb8e`.
- Three independent exact-output audits agreed on canonical structure,
  result-to-calibration binding, candidate bindings, 12-source closure, gate
  recomputation, access receipts, and terminal authority. No checkpoint or
  directory was reopened during result audit.

## Exact procedure

- The V9 terminal result was fully validated before the sole candidate
  checkpoint read. The checkpoint was loaded once on CPU with
  `weights_only=True`, reconstructed bit-exactly, frozen, and left unmutated.
- Calibration used 415 ordered next-frame rows from 8 scenes, 26 batches, and
  all 1,699,840 cells. Selection used 495 ordered rows from 8 scenes, 31
  batches, and all 2,027,520 cells. Endpoint-order hashes exactly match the V4
  protocol.
- One deterministic CPU-float64 hierarchical calibration fit used all cells in
  natural order and empirical proportions, with no balancing, weighting,
  subsampling, dropping, duplication, or class backfill. All classes had
  support.
- The unchanged full 2,016-tuple conservative threshold grid was searched once
  on calibration. The fitted transform and selected diagnostic fallback were
  applied unchanged to selection.
- The model predictor was not recomputed. Backward, optimizer, model EMA,
  training, G2, navigation, held-out, sealed, and train-role payload counts
  were zero.

## Calibration behavior

- Calibration was numerically healthy. On its fit role, joint NLL improved
  `0.143227 → 0.095899`, multiclass Brier `0.067165 → 0.042714`, and accuracy
  `0.953574 → 0.971327`.
- The same transform generalized to selection: joint NLL improved
  `0.183929 → 0.109011`, multiclass Brier `0.090686 → 0.050093`, and accuracy
  `0.936146 → 0.967368`.
- Selection FREE-vs-OCCUPIED-given-known ECE improved
  `0.129670 → 0.017453`; UNKNOWN-vs-known ECE improved
  `0.053167 → 0.012391`.
- Fitted UNKNOWN-vs-known scale/bias: `0.772298 / -1.443384`; fitted
  FREE-vs-OCCUPIED-given-known scale/bias: `0.822128 / -2.024429`. Both
  transforms are positive and monotone: they corrected confidence but could
  not create missing FREE/OCCUPIED rank separation.

## Why the physical gate failed

- No tuple passed even on `probability_calibration`;
  `passing_candidate_count=0`. Returned thresholds FREE `0.50`, OCCUPIED
  maximum `0.35`, UNKNOWN maximum `0.35`, and detection `0.50` are only the
  selector's registered diagnostic fallback, not qualified settings.

| Physical metric | Calibration | Selection | Gate | Selection |
|---|---:|---:|---:|---|
| Admitted FREE precision | `0.948554` | `0.928906` | `>=0.99` | FAIL |
| OCCUPIED detection within 2m | `0.341356` | `0.322811` | `>=0.95` | FAIL |
| Useful FREE recall | `0.867078` | `0.842727` | `>=0.90` | FAIL |
| Obstacle exclusion within 2m | `0.819217` | `0.744179` | `>=0.95` | FAIL |

- All five frozen checks failed, including the existence of any feasible
  calibration tuple. The large simultaneous precision/recall gaps rule out a
  useful rescue by a denser threshold grid or another monotone calibration.

## Comparison with V4 and scientific decision

- Relative to V4 selection, V9 improved near-obstacle detection from
  `0.26853` to `0.32281` (`+5.43` percentage points) and obstacle exclusion
  from `0.72147` to `0.74418` (`+2.27` points).
- FREE precision remained effectively unchanged (`0.92923 → 0.92891`) and
  useful FREE recall fell (`0.85067 → 0.84273`). Aggregate calibrated NLL,
  Brier, and accuracy were also slightly weaker than V4.
- Dense local attention therefore moved obstacle evidence modestly in the
  right direction but did not solve the central score-overlap problem. V9 is
  closed, unqualified, non-resumable, and unauthorized for G2.
- Do not retry calibration, tune its grid, vary the 5x5 support, or combine V9
  with another resolution variant. The next capped joint-JEPA mechanism should
  directly alter learned physical-evidence separability: independent
  high-precision FREE-admission and high-recall near-OCCUPIED evidence with an
  explicit abstain/UNKNOWN region, trained jointly with the retained JEPA
  predictor. History and feasibility must be checked before freezing it.

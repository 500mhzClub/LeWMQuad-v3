# RGB Swept-Progress Survival Joint-JEPA V5 — Near-Field Hazard Ranking Preregistration

- Status: frozen before implementation, training, or any new runtime/data access.
- Purpose: test one gate-aligned training mechanism for the obstacle/free discrimination failure established by V4 physical-evidence calibration. This is one loss-mechanism successor, not a coefficient sweep, calibration retry, threshold expansion, or detached perception/predictor stage.
- Authoritative V4 candidate-admission result: `5f0697361edd81f9dea59be9ef5c635975011c84`.
- Authoritative V4 physical-evidence calibration failure: `1f96caec54e5afa10882cd1e969518164f6dcf1e`.

## Falsified predecessor mechanism

- V4 calibration improved aggregate selection NLL `0.1835528 → 0.1067377` and multiclass Brier `0.0899364 → 0.0480697`, proving that the fitted confidence transform worked numerically and transferred.
- Nevertheless, zero of 2,016 conservative threshold tuples passed. Selection free precision was `0.92923`, useful-free recall `0.85067`, near-obstacle detection `0.26853`, and obstacle exclusion `0.72147`.
- Therefore global affine recalibration and threshold changes cannot supply the missing obstacle/free ordering. V4's generic present-class macro semantic loss plus half-weight all-range occupied-vs-rest auxiliary is misaligned with the extreme near-field safety operating point.

## Sole scientific change from V4

- Retain the exact V4 residual-local semantic decoder, RGB encoder, deformable BEV lift, action-conditioned predictor, survival head, EMA target, data, labels, action vocabulary, masks, optimizer groups/hyperparameters, clipping, seed, schedule, current losses `S+P+U+R+O`, controls, bootstrap, evaluation, and terminal checkpoint at 1,000 updates / 16,000 presentations.
- Initialize freshly through the same accepted N320 encoder-only path and the same V4 decoder constructor seeds. Do not initialize from or open the V4 candidate, V4 original runtime root, no-persistence control, or any rejected checkpoint.
- Add exactly one no-new-parameter loss `H`, active jointly from update one and backpropagated through the existing semantic decoder, BEV lift, and RGB encoder while the JEPA predictor remains trained jointly through its existing losses.

## Exact near-field hazard-ranking loss

- Fixed raster centers: forward `linspace(-0.95,5.35,64)` metres and left `linspace(-3.15,3.15,64)` metres. `near = Euclidean distance <= 2.0m`.
- Per-cell hazard score: `occupied_logit - logsumexp(unknown_logit, free_logit)`, identical in meaning to the existing occupied-vs-rest auxiliary score.
- For each current or next raster row, form the complete Cartesian set of pairs between true OCCUPIED cells inside `near` and true FREE cells inside `near`.
- An eligible row contains at least one cell of each set. Its loss is the arithmetic mean of `softplus(free_hazard_score - occupied_hazard_score) / log(2)` over every pair. There is no margin, sampling, mining, class backfill, distance weighting, or hand-selected subset beyond the frozen 2m mask and exact labels.
- Average eligible current rows and eligible next rows separately, then equally average the present view means. If neither view has an eligible row, return an exact graph-connected zero and record the inactive microbatch.
- Coefficient: exactly `1.0`. Total loss is `L_v4 + H`. No alternative coefficient, margin, range, score, or pair rule is authorized.

## Training and gate

- One fresh V5 run only, with the exact V4 cap: 1,000 optimizer/EMA updates, 4,000 microbatch graphs/backward calls, and 16,000 presentations. No retry, resume, intermediate-checkpoint selection, schedule extension, or early gate tuning.
- Record `H` by microbatch and 100-update window, eligible current/next row counts, ranked pair counts, gradient receipts, and the unchanged accounting for every inherited loss and model group.
- The terminal checkpoint must first pass the complete unchanged V4 24-check full-arm development gate, preserving semantic class recall, JEPA action utility, family metrics, and all controls. A failure closes V5 without calibration or G2.
- If and only if the unchanged V4 gate passes, package the terminal checkpoint as a development-only pre-calibration candidate and apply one separately source-frozen execution of the exact already-reviewed global calibration, 2,016-tuple threshold selection, and fixed selection-role physical gate. Calibration/selection parameters and thresholds remain unchanged from the V4 protocol.
- Success requires the physical development gate: at least one passing calibration tuple; selection free precision at least 0.99; near-obstacle detection at least 0.95; useful-free recall at least 0.90; and near-obstacle exclusion at least 0.95.

## Stopping and authority

- Meaningful improvement without a pass may justify a later material architecture decision, but does not authorize coefficient/margin/range variants of `H` or another calibration/threshold retry.
- A V5 physical pass authorizes only preparation of a separately reviewed one-shot G2 binding. It does not itself open G2, qualify navigation, promote, deploy, or access sealed/held-out material.
- No G2, navigation, held-out, sealed, production, rejected-checkpoint, no-persistence-checkpoint, or original-V4-runtime access is authorized by this preregistration.

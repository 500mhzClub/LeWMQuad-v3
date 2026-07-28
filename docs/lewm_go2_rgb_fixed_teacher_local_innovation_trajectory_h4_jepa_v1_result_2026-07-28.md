# Go2 RGB fixed-teacher local-innovation trajectory H4 JEPA V1 result — 2026-07-28

## Terminal status

- Decision:
  `STOP_MAIN_POOL_RGB_FIXED_TEACHER_LOCAL_INNOVATION_TRAJECTORY_H4_JEPA_V1`.
- The sole attempt completed normally at the exact cap. This was a scientific
  STOP, not an execution failure: 1,000 optimizer updates, 16,000 ordered
  training presentations, and 10,240 validation presentations completed in
  `2400.150838` active GPU seconds.
- Update 1,000 / presentation 16,000 was selected by the preregistered minimum
  combined joint-plus-marginal normalized energy score among eligible
  noncollapsed trained observations.
- The probe passed 21 of 26 gates. Its five failures were:
  - H4 marginal normalized energy score at most `0.90`;
  - H4 ordered-history gap at least `0.03`;
  - positive H4 ordered-history bootstrap lower bound;
  - positive H4 persistence-gap bootstrap lower bound;
  - no family H4 persistence gap below `-0.02`.
- This exact fixed-teacher local-innovation/cyclic-action/reverse-reset-history
  mechanism is closed. There is no retry, resume, second seed, longer run, or
  checkpoint access.

## What worked

The model learned a strong response to the exact trained cyclic wrong-action
intervention and recovered a modest aggregate persistence win by the cap:

| Selected metric | Value |
|---|---:|
| Combined normalized energy score | 0.852201 |
| Joint-trajectory normalized energy score | 0.846956 |
| H1 marginal normalized energy score | 0.896672 |
| H2 marginal normalized energy score | 0.943366 |
| H3 marginal normalized energy score | 0.926697 |
| H4 marginal normalized energy score | 0.953251 |
| H4 persistence gap | +0.046749 |
| H4 cyclic wrong-action gap | +0.187365 |
| H4 cyclic-action bootstrap lower 95% | +0.170547 |
| Combined distribution-value gap | +0.287941 |
| Combined distribution-value bootstrap lower 95% | +0.273951 |
| H4 normalized pairwise spread | 1.470570 |

- Every aggregate marginal horizon beat exact-zero innovation/persistence at
  update 1,000. The joint and combined trajectory scores also beat
  persistence.
- The exact trained cyclic wrong-action gap was positive at every horizon and
  in all eight maze families at H4. This is a large improvement over the
  cumulative-target trajectory predecessor's selected H4 action gap of
  `+0.010815`.
- Four-atom support retained clear distributional value. The combined
  distribution-value gap was positive in all eight families.
- Representation geometry stayed healthy and stable:
  - online effective-rank ratio `0.205248`;
  - fixed-target effective-rank ratio `0.174981`;
  - online and target near-zero-variance fractions `0.0`;
  - fixed-target rank and near-zero-variance drift exactly `0.0`;
  - fixed-target initial and final state hashes identical.
- Prediction improved sharply over the final 250 updates: combined score fell
  from `0.897599` to `0.852201`, while H4 crossed from slightly worse than
  persistence to an aggregate `+0.046749` persistence gap.

## What failed

The combined local-innovation and counterfactual-ranking mechanism did not
preserve the predecessor's broad prediction quality, and the predictor still
did not use ordered history in a meaningful way:

| Selected H1--H4 metric | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| Marginal normalized energy score | 0.896672 | 0.943366 | 0.926697 | 0.953251 |
| Persistence gap | +0.103328 | +0.056634 | +0.073303 | +0.046749 |
| Cyclic wrong-action gap | +0.015999 | +0.071338 | +0.106890 | +0.187365 |
| Ordered-history gap | +0.000781 | +0.003748 | +0.001590 | +0.000948 |
| All-hold gap | -0.001902 | -0.008031 | +0.001408 | +0.001280 |

- H4 score was `0.953251`, missing the registered `0.90` ceiling by
  `0.053251`. The cumulative-target trajectory predecessor was substantially
  better at its selected observation: combined score `0.725748`, H4 score
  `0.742969`, and H4 persistence gap `+0.257031`.
- The aggregate H4 persistence win was not robust under the registered
  bootstrap. Its lower bound was `-0.017683`.
- Only six of eight families beat persistence at H4. Local-composite motifs
  were materially worse at `-0.123326`; rough local dynamics narrowly failed
  the family floor at `-0.020247`.
- Ordered history remained effectively unused. H4 history gap was only
  `+0.000948`, its bootstrap lower bound was `-0.001676`, and large-enclosed
  and loop-alias families were negative. The training history-ranking loss was
  `0.030582` on average and still `0.026591` on the final update, remaining
  near the `0.03` hinge scale.
- The apparent action success did not generalize to the untrained all-hold
  corruption, which had only a `+0.001280` H4 aggregate gap and was positive
  in just two of eight families. Other untrained mappings were not tested, so
  the cyclic result may be intervention-specific and does not establish
  broadly correct action counterfactuals.
- Point predictions were poor despite the proper distribution score:
  H4 best-atom normalized squared error was `4.989570`, and spherical-centroid
  normalized squared error was `5.602796`.

## Learning trajectory

| Update | Combined score | H4 score | H4 persistence gap | H4 cyclic-action gap | H4 history gap |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 1.000000 | -0.000000 | 0.000000 | 0.000000 |
| 250 | 0.910233 | 1.040373 | -0.040373 | +0.230004 | -0.001552 |
| 500 | 0.910640 | 1.038298 | -0.038298 | +0.263428 | -0.007087 |
| 750 | 0.897599 | 0.993237 | +0.006763 | +0.233932 | -0.015818 |
| 1,000 | 0.852201 | 0.953251 | +0.046749 | +0.187365 | +0.000948 |

- Combined and H4 prediction scores, persistence gap, and history gap improved
  from update 750 to 1,000, so this was not an optimization crash. Cyclic
  action gap declined over that interval, and the exact run is nevertheless
  consumed and cannot be extended under its preregistration.
- Action ranking learned early and strongly for the trained cyclic mapping.
  History ranking did not show a comparable learning response.
- Mean training losses over all updates were `0.511175` local-innovation
  energy score, `0.005875` same-RGB teacher alignment, `0.028017` cyclic-action
  ranking, and `0.030582` history ranking. Their last-update values were
  `0.553186`, `0.004038`, `0.015277`, and `0.026591`, respectively.

## Scientific interpretation

- Relative to the predecessor, the combined mechanism produced sensitivity to
  the trained cyclic action corruption but much weaker cumulative future
  prediction. Because target and ranking changed together, this run cannot
  causally assign those outcomes to either change in isolation.
- The result does not show that the model learned generally controllable
  dynamics. Strong cyclic-action separation alongside nearly zero all-hold
  separation is consistent with learning intervention-specific degradation.
- The dense history path is present and trainable, but the result is consistent
  with the model relying on an easier current-frame shortcut; the receipts do
  not expose its internal cause. Direct reverse/reset ranking was not enough
  to make real ordered history predictive. This is now the dominant
  mechanism-level blocker; encoder collapse is not.
- More examples or more updates to this exact objective are not justified.
  A successor must change how counterfactual control/history information is
  learned or how temporal state is forced to carry predictive information. It
  should reserve its decisive controls for evaluation rather than train on the
  exact same corruptions.
- Any successor remains a bounded development experiment. It must use fresh
  initialization from the accepted N320 encoder only, keep one joint
  RGB/action JEPA backward, use no navigation labels, and leave held-out and
  sealed material unopened.
- This result itself grants no successor training, GPU, data, or execution
  authority. Any successor relies on a separate controlling authorization.

## Execution and custody audit

- Frozen source commit:
  `aa4d441c837258c9f24052949e06a36cf3325522`. Independent review and one-shot
  authority commit: `fdd1f837d112ce42f4ea272d963d46ad8e619c3c`.
- Train schedule SHA-256:
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`,
  16,000 rows / 1,000 scenes. Validation schedule SHA-256:
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`,
  2,048 rows / 150 scenes.
- RGB access was exactly 183,680 successful opens from 183,680 attempts:
  112,000 training views plus 71,680 validation views. Counterfactual reuse
  added no RGB access. The receipts recorded 16,000 wrong-action control
  sequences and 32,000 auxiliary history-control sequences.
- Test/held-out, sealed, label, arbitrary-checkpoint, retry/resume checkpoint,
  and retry/resume counts were all zero.
- The fixed teacher's initial and final state SHA-256 were identical:
  `dd3c8f053808848f1caa63b5870b0948382c9c875b7d6848ab8a1cf05a8f3e4b`.
  It recorded zero EMA updates. The accepted N320 initialization was opened
  exactly once; no predecessor predictor checkpoint was opened.
- Reservation and artifact receipts contain identical bindings for all eight
  reviewed runtime source files. A repository-side hash check also matched
  every bound source file.
- Terminal JSON receipt file bindings:
  - reservation: `fa720617dd28109109d7fa75e0e06dd869edc8363ff2665bec44321b8c264520`,
    5,093 bytes;
  - metrics: `21eeddd08d09aa5aa816c235be6d019d28e8677c8029495aaf12636e8ff36925`,
    53,505 bytes;
  - artifact: `f4162c47e5d6404cf09d6bc6982e1acfb7eb59d29e0c253e1ece632cd46a57a1`,
    5,270 bytes;
  - access: `c96198856a060a141fa005f947ec0242664379c88bf3b54701099af6f2b1291d`,
    1,284 bytes;
  - result: `4b6b1968cf6e7ffd5d87835da696683cda3e45d66088b2b509d9efa65607f107`,
    2,369 bytes;
  - completed: `cd40b375e382da82dfdfa1c5c291fd1860c402f95772849a03ac1725e36becd7`,
    1,856 bytes.
- All six canonical content hashes and canonical file encodings were
  independently recomputed and matched. Completion byte counts, file hashes,
  content hashes, and cross-bindings all matched. A second independent receipt
  audit found zero deviations.
- The reservation's inherited top-level `wrong_action_contrast` leaf says
  disabled/weight zero, while the nested additional-science contract,
  artifact, access, training-loss, and source receipts correctly record the
  local-innovation cyclic contrast. This is a schema ambiguity, not an
  execution-integrity failure.
- Four checkpoint filenames were inventoried from artifact JSON metadata only.
  No generated `.pt` file was opened, hashed, loaded, copied, statted, exposed
  through a checkpoint filesystem listing, or reused. They remain inaccessible
  under STOP.

STOP grants no checkpoint, navigation, held-out, promotion, or deployment
authority. The existing V4 sealed benchmark remains unopened; it is a legacy
development-only asset and permanently ineligible for final evaluation.

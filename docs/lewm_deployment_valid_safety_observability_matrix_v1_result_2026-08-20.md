# Deployment-valid safety observability matrix V1 result

Date: 20 August 2026  
Parent source commit: `4798995c20d2e8eada17ff0c1333d72364b24e8d`  
Policy: `EVALUATION_FIRST_SINGLE_SEED`, `DEVELOPMENT_MODE_END_TO_END_EXECUTION`  
Final classification: **`CURRENT_DEPLOYMENT_SENSOR_CONTRACT_SAFETY_NO_GO`**

## Result

No deployment-valid condition passed the frozen common gate. At its frozen
calibration threshold, the best operational result was RGB plus
proprioception: held-out unsafe recall was 1.0000 with zero unsafe selections,
but safe retention was only 0.1316, only 3/8 states retained a safe candidate,
and five states falsely abstained. It also missed the aggregate-AUC, stuck,
event-recall, detection-delay, safe-retention, state-coverage, and best-safe
top-3 gates.

This single-seed small-model result is not an information-theoretic proof that
the sensors contain no safety information. It does show that none of the four
frozen sensor/model contracts supplies the required safety-versus-useful-action
operating point on this panel. Another post-hoc head on the same inputs is not
recommended. The next experiment should add a reliable safety-relevant channel
(contact/body-contact sensing, motor current or torque, linear acceleration,
depth, or LiDAR), or explicitly narrow the safety claim to failure modes that
are observable under the present contract.

## Frozen panel and materialisation

- Panel: 48 states, 576 frozen candidate branches, four maze families, split
  32/8/8 states and 384/96/96 branches for fit/calibration/held-out.
- Dense horizon: 15 policy ticks through H3; registered H1/H2/H3 boundaries are
  ticks 5/10/15.
- Dense RGB lineage: 8,688 occurrences, 7,154 unique 224x224 frames. No valid
  RGB was rerendered.
- Frozen final ViT-L comparator index:
  `7c24306dd1082940f948e47584ac525e451717258255585c04f82956736571f0`.
- Dense proprioception index:
  `4ffbf87ea46aa70e134030daac07de5f19ecc27bc7708148f77ecdb1ae1e55ad`.
- All 576 registered candidates were deterministically replayed solely to
  record deployment-valid sensors. Post-slew actions, tick poses, contact and
  stuck traces had zero mismatches. No candidate identity or outcome changed.
- Thirty-six snapshot byte digests reproduced. Twelve did not, matching the
  known replay-lineage behavior; all twelve nevertheless reproduced every
  scientific tick check and were retained.
- Proprioception shards occupy 1,171,384 bytes. The reusable uint8 RGB mmap is
  `[7154,224,224,3]`, occupies 1,076,877,440 bytes, and has index digest
  `14cb69844ae7faf2fa41e245bc3c07248b6a7a3f370d5d19a9c173b0b7a911ce`.

The 42-dimensional proprioceptive vector contains projected gravity (3),
body-frame angular velocity (3), joint position relative to default (12),
joint velocity (12), and previous policy action (12). A separate four-channel
stream contains current post-slew and previous applied `vx/yaw`. Fit-only mean
and variance standardisation was frozen before fitting. No requested channel
was degenerate.

Excluded from every model were global position, absolute yaw, body linear
velocity, simulator graph/occupancy, privileged geometry, safety labels,
future pose, route intent, motor torque, IMU linear acceleration, and foot
contact. The last three sensor groups were absent or ineligible in the frozen
corpus.

## Label prevalence

| Split | Ticks | Contact active | Contact cumulative | Stuck active | Stuck cumulative | Aggregate cumulative |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 5,760 | 0.0720 | 0.3028 | 0.0990 | 0.3413 | 0.5040 |
| Calibration | 1,440 | 0.0708 | 0.3486 | 0.0986 | 0.3799 | 0.5771 |
| Held-out | 1,440 | 0.0326 | 0.1222 | 0.0688 | 0.2597 | 0.3896 |

Fall/unsafe termination remained degenerate at zero and was not trained.
Clearance was retained descriptively only; it was not a model output.

## Evaluator-first fixture

The common evaluator passed all pre-training fixtures: one-tick transient
contact, persistent contact, delayed stuck, a safe branch, all-unsafe and
one-safe candidate sets, no admission, an exact threshold tie, perfect and
reversed probability rankings, kinematic selection of the unique safe
candidate, and deterministic complete JSON reload.

## Model contracts and training

All conditions use a one-layer causal GRU with width 128, balanced BCE over the
five frozen per-tick targets, AdamW at `1e-3` with weight decay `1e-4`, 60
epochs, final epoch only, and one condition-keyed RNG derived from the frozen
seed family `2026082008`. No checkpoint or hyperparameter was selected from
calibration or held-out performance.

| Condition | Allowed evidence | Parameters | Keyed seed | Initial loss | Final loss | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---|
| ACTION_CONTROL_ONLY | candidate and previous `vx/yaw` | 69,237 | 3901640947 | 1.0198 | 0.9022 | `bc80ad410f83ab8503976a2cca850c833e05759af9e0cb85c46b406644eb8dcf` |
| RAW_RGB | current/future/difference RGB plus action/control | 289,173 | 1926419654 | 1.0175 | 0.3009 | `0d39132f7eb365d852af3610791f0f7a819799aee9e1597aea75de9818710109` |
| PROPRIOCEPTION | current/future/difference proprioception plus action/control | 127,605 | 3249073349 | 1.0208 | 0.2692 | `815e1d8aeccaa82b483e29c17a79813810342a886c12341f437502ab90215267` |
| RGB_PLUS_PROPRIOCEPTION | both evidence streams plus action/control | 347,541 | 530660527 | 1.0111 | 0.0104 | `7ab037b6da5f34e3d1515d0cba4a5da4696830f0008bb63e116cca3abcb22d28` |

The very low multimodal fit loss did not transfer: its held-out aggregate,
contact, and stuck AUC drops from fit were 0.2330, 0.2387, and 0.2992.

## Calibration

One scalar aggregate-logit temperature and one admission threshold were fit on
the same eight calibration states per condition. Each threshold maximised safe
retention subject to unsafe recall at least 0.95; ties were resolved toward the
more conservative threshold.

| Condition | Temperature | Threshold | Calibration recall | Calibration safe retention |
|---|---:|---:|---:|---:|
| ACTION_CONTROL_ONLY | 1.1876 | 0.1601 | 0.9861 | 0.0417 |
| RAW_RGB | 6.0648 | 0.0000 | 1.0000 | 0.0000 |
| PROPRIOCEPTION | 7.4252 | 0.2621 | 0.9583 | 0.1667 |
| RGB_PLUS_PROPRIOCEPTION | 8.8601 | 0.2686 | 0.9722 | 0.2083 |

## Held-out branch and event metrics

| Condition | Unsafe AUC | AP | Recall | FNR | Safe retention | ECE | Contact AUC | Contact tick recall | Stuck AUC | Stuck tick recall | Missed transient contact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ACTION_CONTROL_ONLY | 0.7793 | 0.8618 | 1.0000 | 0.0000 | 0.0000 | 0.1054 | 0.5747 | 0.6596 | 0.8654 | 0.7576 | 0.3902 |
| RAW_RGB | 0.7033 | 0.7670 | 1.0000 | 0.0000 | 0.0000 | 0.1347 | 0.6719 | 0.2979 | 0.8582 | 0.3939 | 0.7073 |
| PROPRIOCEPTION | 0.7679 | 0.8167 | 0.9828 | 0.0172 | 0.0526 | 0.0760 | 0.7671 | 0.2766 | 0.6541 | 0.3030 | 0.7561 |
| RGB_PLUS_PROPRIOCEPTION | 0.7670 | 0.7761 | 1.0000 | 0.0000 | 0.1316 | 0.0933 | 0.7613 | 0.2553 | 0.7008 | 0.1010 | 0.7561 |
| Frozen final-layer ViT-L | 0.7255 | — | — | — | 0.0000 | — | 0.6175 | — | 0.7745 | — | 0.9756 |

Median detected-event delay was 0, 0, 1, and 2 ticks for action/control, raw
RGB, proprioception, and fusion respectively. Missing events are separately
captured by event recall; they are not turned into finite delays.

The descriptive held-out operating curves reinforce the lack of a robust
operating point. Maximum safe retention at unsafe recall at least 0.95 was
0.2105/0.0789/0.1842/0.4474. Maximum unsafe recall at safe retention at least
0.40 was 0.8621/0.8103/0.9138/0.9655 in the same condition order. The latter
fusion point is a held-out curve diagnostic, not an authorised replacement for
the calibration-frozen threshold.

## Candidate filtering and mission progress

| Condition | Safe states retained | False abstentions | Unsafe selected | Mean realised distance progress (m) | Fraction of oracle-safety kinematic benchmark | Normalized regret | Best-safe top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACTION_CONTROL_ONLY | 0/8 | 8 | 0 | 0.0000 | 0.000 | NA | 0.000 |
| RAW_RGB | 0/8 | 8 | 0 | 0.0000 | 0.000 | NA | 0.000 |
| PROPRIOCEPTION | 2/8 | 6 | 0 | 0.3159 | 1.446 | 0.2970 | 0.000 |
| RGB_PLUS_PROPRIOCEPTION | 3/8 | 5 | 0 | 0.4024 | 1.842 | 0.0626 | 0.250 |
| Oracle safety + kinematics | 8/8 | 0 | 0 | 0.2184 | 1.000 | 0.0000 | 1.000 |

The progress fraction can exceed one because the oracle-safety comparator is
an oracle *filter* followed by the unchanged nominal kinematic ranking; it is
not an oracle ranking by realised progress. The fusion model happened to retain
a smaller subset with higher realised progress. That does not rescue it: five
states stopped despite available safe actions and best-safe top-3 was only
0.25. Reject-all behavior is therefore reported as mission non-performance,
not as safety success.

## Per-family filtering

Each cell is aggregate AUC / safe retention / states retaining safe candidates
out of two.

| Condition | Large enclosed | Medium enclosed | Small enclosed | Loop alias |
|---|---|---|---|---|
| ACTION_CONTROL_ONLY | 0.5000 / 0 / 0 | 0.9556 / 0 / 0 | 0.8857 / 0 / 0 | 0.7556 / 0 / 0 |
| RAW_RGB | 0.6875 / 0 / 0 | 0.7185 / 0 / 0 | 0.7143 / 0 / 0 | 0.9333 / 0 / 0 |
| PROPRIOCEPTION | 0.7250 / 0 / 0 | 0.6815 / 0.0667 / 1 | 0.8286 / 0.1000 / 1 | 0.6222 / 0 / 0 |
| RGB_PLUS_PROPRIOCEPTION | 0.6750 / 0 / 0 | 0.7926 / 0.2667 / 2 | 0.8464 / 0.1000 / 1 | 0.4852 / 0 / 0 |

Both learned embodied conditions collapsed operationally in the large-maze and
loop-alias families.

## Representation and modality diagnosis

- Raw RGB minus frozen final-layer ViT-L: aggregate AUC `-0.0222`, contact AUC
  `+0.0544`, safe retention `0.0000`. This does **not** meet the predeclared
  strong final-layer representation-gap rule.
- Proprioception minus action/control: aggregate AUC `-0.0113`, safe retention
  `+0.0526`, contact AUC `+0.1924`, but stuck AUC `-0.2113` and both per-tick
  recalls were worse. This is component-specific evidence, not a qualifying
  proprioceptive interface.
- Fusion minus the stronger unimodal result: aggregate AUC `-0.0009`, safe
  retention `+0.0789`, contact AUC `-0.0058`, stuck AUC `-0.1573`. This misses
  the frozen strong-multimodal-tendency rule.
- The privileged static-grid guard remains frozen at unsafe recall 0.6724,
  false-negative rate 0.3276, and safe retention 0.5789; it is not a qualified
  candidate safety filter.

The result therefore does not distinguish a single clean representation
bottleneck. The available modalities carry different component signals, but
the present sensor/model contracts do not jointly support transient contact,
persistent stuck, calibrated aggregate rejection, and retained useful action.

## Runtime, storage, and custody

- Proprioception replay: 1,506.1 aggregate compute-seconds, 567.8 seconds
  parallel wall span.
- Four fits: 310.0 seconds total; peak allocated VRAM 1,484,044,288 bytes.
- Bound replay plus training compute: 1,816.1 seconds.
- New external cache/checkpoint storage: 1,081,413,172 bytes.
- Final result SHA-256:
  `b2aefe064f025007cfdf7ddbd224ddd5bdc0e9b834800a45d9350d4191331a89`.

Exactly one keyed seed was used per condition. No JEPA predictor was opened or
trained. No new state or candidate identity was created. No memory, novelty,
beacon-discovery, routing, planning-model, or navigation layer was trained or
implemented. Nothing remains running.

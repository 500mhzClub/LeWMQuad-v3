# Dense temporal true-future safety observability V1

Date: 2026-08-20  
Source parent: `9ba5e1f0e6742f32c45e9c101a1b941e91444bea`  
Policy: `EVALUATION_FIRST_SINGLE_SEED`, `DEVELOPMENT_MODE_END_TO_END_EXECUTION`  
Classification: **`RGB_ONLY_DENSE_TEMPORAL_SAFETY_NO_GO`**

## Scope and preserved results

This experiment preserves `TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO` and
`STRUCTURED_SAFETY_LABEL_OR_ALIGNMENT_DEFECT`. It asks only whether safety is
decodable from dense *true-future* RGB trajectories through H3. No JEPA
predictor was opened or trained, and no memory, novelty, navigation, or
planning model was trained.

The frozen purpose-built panel remained unchanged: 48 states, 576 branches,
four maze families, and the original 32/8/8 fit/calibration/held-out split.

## Dense evidence

The branch ledger does not contain complete tick poses, so the authorised
deterministic replay fallback was used. Each frozen pre-action state was
redriven, each registered post-slew candidate was replayed, and only the
registered branch identity was retained. H1/H2/H3 are ticks 5/10/15; every
branch therefore has 15 future frames plus one state-shared pre-action frame.
All replay-recovered H1-H3 poses, actions, and V2 component labels matched
their frozen receipts.

The older path-level ledger and the later replay component receipt disagree
on 77 of 1,728 registered horizon rows. This known historical distinction is
not hidden or recast as a contact/stuck event. Dense tick components come from
the replay; the authoritative aggregate unsafe outcome is introduced only at
its first known right-censoring boundary (H1, H2, or H3). No component identity
or earlier event time is invented.

| Evidence item | Result |
|---|---:|
| States / branches | 48 / 576 |
| Ticks per branch | 15 |
| Frame occurrences | 8,688 |
| Unique RGB frames / tokens | 7,154 |
| Token shape and dtype | `[768,1024]`, FP16 |
| Token bytes | 11,252,269,056 |
| RGB bytes | 259,924,007 |
| Total high-capacity cache | 11,514,226,681 bytes |
| Replay aggregate CPU time | 1,306.81 s |
| Encoding time / throughput | 137.20 s / 52.14 unique frames/s |
| Peak encoding VRAM | 1,814,315,008 bytes |
| Token-index digest | `7c24306dd1082940f948e47584ac525e451717258255585c04f82956736571f0` |
| Evidence receipt SHA-256 | `a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1` |

## Temporal observability audit

Events are overwhelmingly micro-timescale. Median event-run duration was one
policy tick for contact, stuck, and aggregate unsafe in every split. Most runs
occurred entirely between H1/H2/H3 sample instants.

| Split | Component | Positive branches | Positive ticks | Event runs | Entirely between sparse endpoints |
|---|---|---:|---:|---:|---:|
| Fit | Contact | 189/384 | 415 | 351 | 247 (70.4%) |
| Fit | Stuck | 204/384 | 570 | 402 | 281 (69.9%) |
| Fit | Aggregate | 268/384 | 874 | 580 | 390 (67.2%) |
| Calibration | Contact | 50/96 | 102 | 92 | 75 (81.5%) |
| Calibration | Stuck | 53/96 | 142 | 94 | 68 (72.3%) |
| Calibration | Aggregate | 69/96 | 226 | 159 | 121 (76.1%) |
| Held out | Contact | 24/96 | 47 | 44 | 33 (75.0%) |
| Held out | Stuck | 44/96 | 99 | 81 | 59 (72.8%) |
| Held out | Aggregate | 52/96 | 142 | 110 | 77 (70.0%) |

Across the complete panel, 56.7% of contact-positive branches were not active
at any sparse endpoint; 48.8% of stuck-positive branches were active at at
least one endpoint. Among the 407 authoritative unsafe branches, 88 (21.6%)
were contact-only and 126 (31.0%) were stuck-only; 175 branches contained both
replay-recovered components.

## Evaluator and model

The evaluator fixture passed one-tick transient contact, persistent contact,
delayed stuck, safe, all-unsafe, one-safe, reject-all, threshold-tie, and
deterministic JSON cases before training.

`DENSE_TEMPORAL_SAFETY_HEAD_V1` implements the frozen architecture: shared
1024-to-32 token projection; two 48-channel spatial convolutions over the
24-by-32 grid; spatial mean/max pooling; 16-D action and 32-D control-history
features; a one-layer 128-D causal GRU; and five per-tick logits. It has
201,157 trainable parameters. The ViT-L encoder remained frozen.

One seed, `2026082007`, trained for exactly 60 epochs with AdamW, learning rate
`1e-3`, weight decay `1e-4`, and balanced BCE. Mean fit loss was 1.01074 at
epoch 1, 0.75555 at epoch 10, 0.52042 at epoch 20, 0.21297 at epoch 30,
0.02521 at epoch 40, 0.00480 at epoch 50, and 0.00243 at epoch 60. Training
took 360.91 s and peaked at 1,303,461,376 bytes of VRAM.

Final checkpoint SHA-256:
`599684f1c509fb15a368e3e151f3c9df172f04d268423821079a0510c86c5e37`.

## Calibration

The fitted scalar temperature was 7.33235. The frozen recall-first calibration
rule selected threshold 0.0: this obtains calibration recall 1.0 but safe
retention 0.0. The result is scientifically informative, not a usable filter.

## Held-out result

| Metric | Dense true future | Required | Pass |
|---|---:|---:|:---:|
| Aggregate unsafe AUC | 0.7255 | >= 0.80 | No |
| Contact/collision AUC | 0.6175 | >= 0.75 | No |
| Stuck AUC | 0.7745 | >= 0.85 | No |
| Aggregate recall / FNR | 1.000 / 0.000 | >= 0.95 / <= 0.05 | Yes |
| Safe retention | 0.000 | >= 0.40 | No |
| Aggregate ECE | 0.0897 | <= 0.10 | Yes |
| Contact tick recall | 0.0213 | >= 0.80 | No |
| Stuck tick recall | 0.0505 | >= 0.90 | No |
| Median first-event delay | 5 ticks | <= 1 tick | No |
| Missed transient-contact rate | 0.9756 (40/41) | <= 0.20 | No |

Component H3 recalls were 0.4167 for contact and 0.5227 for stuck. Aggregate
average precision was 0.8030 and Brier score 0.2099. Fifty-two component
events were never detected by their active-event heads.

### Family results

| Family | Aggregate AUC | Contact AUC | Stuck AUC | Aggregate safe retention |
|---|---:|---:|---:|---:|
| `large_enclosed_maze` | 0.6625 | 0.2656 | 0.8214 | 0.000 |
| `medium_enclosed_maze` | 0.8519 | 0.7125 | 0.8203 | 0.000 |
| `small_enclosed_maze` | 0.6357 | 0.7474 | 0.6084 | 0.000 |
| `loop_alias_stress` | 1.0000 | 0.8824 | 0.9185 | 0.000 |

The variation does not rescue the interface: the single frozen admission
threshold rejects every held-out candidate in every family.

## Candidate filtering and kinematic planning

All 96 held-out candidates were rejected. Consequently all eight states
abstained, including eight false abstentions; no state retained a safe
candidate. Selected unsafe rate was zero only because nothing was selected.
Selected route progress was 0, normalized regret was undefined, and best-safe
top-3 recovery was 0. The oracle-safety kinematic upper bound selected 0.2184 m
mean distance progress, had zero unsafe selection, zero abstention, and
best-safe top-3 of 1.0.

Frozen comparators remain:

| Comparator | Aggregate AUC | Contact AUC | Stuck AUC | Safe retention | Abstention |
|---|---:|---:|---:|---:|---:|
| Sparse endpoint head | 0.6565 | 0.4549 | 0.7863 | 0.000 | 1.000 |
| Joint safety-auxiliary actual future | 0.7459 | 0.6429 | 0.8680 | 0.000 | 1.000 |
| Action-only | 0.7471 | n/a | n/a | 0.0526 | 0.750 |
| Privileged static-grid guard | n/a | n/a | n/a | 0.5789 | n/a |

Dense temporal input improves on the sparse endpoint head's aggregate and
contact AUC, but it is worse than the joint auxiliary successor and does not
produce a usable safety-retention operating point. It therefore does not meet
the predeclared positive-tendency rule either.

## Decision

The final classification is **`RGB_ONLY_DENSE_TEMPORAL_SAFETY_NO_GO`**.

The dense visual trace reveals that most events are transient and missed by
sparse endpoints, but the registered RGB-only temporal head strongly fits the
small fit corpus and fails scene-disjoint held-out discrimination and event
detection. The next safety representation must include deployment-valid
proprioception or another safety-relevant modality; another RGB-only seed or
post-hoc architecture is not authorised by this result.

Result JSON SHA-256:
`96dc7e4d80c1a19726498cbe7bef5cd9b6825fb2c202601b6af1110087a69922`.

No JEPA predictor was opened or trained. No model other than the single
registered safety-head seed was trained. Deterministic physics replay was used
only to recover the missing tick poses and RGB evidence for existing branch
identities; no new state or candidate branch was created. Nothing remained
running at handoff.

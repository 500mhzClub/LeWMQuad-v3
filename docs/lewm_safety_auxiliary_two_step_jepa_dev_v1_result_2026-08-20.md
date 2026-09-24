# Safety-Auxiliary Two-Step JEPA Development V1

Date: 2026-08-20  
Source commit: `c26a89a7ea6a8aeec06db9397d97b6a67a1dbc6c`  
Final classification: `SAFETY_AUXILIARY_JEPA_DEVELOPMENT_NO_SIGNAL`

## Scope and bindings

This development experiment preserved `TRUE_FUTURE_SAFETY_HEAD_NO_GO` and
continued exactly one RGB two-step model from seed `2026080901`, using continued
training seed `2026082004`. The historical checkpoint was hash-verified as
`75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4` and was
not overwritten.

The purpose-built safety panel was reused unchanged: 48 states and 576 branches,
split into 384 fit, 96 calibration, and 96 held-out rows. The target-latent index
matched `df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874`.
The branch ledger SHA-256 was
`9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c`.

The original two-step training stream was retained through 3,922 canonical rows:

- manifest digest: `6ff053033475debd3d8bb415080efb15adfaefc31f01295b956bd85c12b6dac0`;
- cache-map digest: `a45bcc7d46da3c085f0603e79e568f1228b76c489868d6a96aed2b1485d85a7e`;
- normalisation SHA-256: `f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab`.

The historical checkpoint contains the predictor and optimizer, but no online
encoder or target-EMA state. Accordingly, the frozen cached ViT-L token contract
was retained; safety gradients reached the predictor through the predicted-future
path, but no historical online encoder could lawfully be reconstructed. The route
panel also lacks a separately retained exact pre-action ViT-L grid. The
candidate-invariant H1 hold target was therefore used as the current-context
proxy, matching the preceding post-hoc safety experiment and recorded as a
development limitation.

No simulation, rendering, target encoding, branch regeneration, or label change
occurred.

## Evaluator-first fixture

The deterministic fixture passed all cases: actual/predicted/current-context
conditions, oracle comparison, guarded kinematic selection, perfect and reversed
discrimination, one/all unsafe, one safe, false-negative admission,
false-positive rejection, no admitted candidate, calibrated/miscalibrated scores,
and byte-deterministic JSON.

Fixture SHA-256: `2cdfb782112d7aaf32bc1c6583b4b708634794fc047ec4a31bb2893b8368d145`.

## Safety-label prevalence

Counts are positive rows / rows, followed by prevalence. Fall and unsafe
termination are zero throughout and were not assigned separate trainable outputs.

| Split | Family | H | Contact | Clearance | Stuck | Aggregate unsafe |
|---|---|---:|---:|---:|---:|---:|
| fit | overall | 1 | 88/384 (.2292) | 12/384 (.0312) | 111/384 (.2891) | 181/384 (.4714) |
| fit | overall | 2 | 152/384 (.3958) | 14/384 (.0365) | 164/384 (.4271) | 241/384 (.6276) |
| fit | overall | 3 | 189/384 (.4922) | 14/384 (.0365) | 204/384 (.5312) | 277/384 (.7214) |
| fit | large_enclosed_maze | 1 | 22/96 (.2292) | 0/96 | 26/96 (.2708) | 37/96 (.3854) |
| fit | large_enclosed_maze | 2 | 39/96 (.4062) | 0/96 | 38/96 (.3958) | 57/96 (.5938) |
| fit | large_enclosed_maze | 3 | 45/96 (.4688) | 0/96 | 48/96 (.5000) | 64/96 (.6667) |
| fit | medium_enclosed_maze | 1 | 15/96 (.1562) | 0/96 | 22/96 (.2292) | 37/96 (.3854) |
| fit | medium_enclosed_maze | 2 | 29/96 (.3021) | 2/96 (.0208) | 38/96 (.3958) | 53/96 (.5521) |
| fit | medium_enclosed_maze | 3 | 37/96 (.3854) | 2/96 (.0208) | 48/96 (.5000) | 66/96 (.6875) |
| fit | small_enclosed_maze | 1 | 21/96 (.2188) | 0/96 | 29/96 (.3021) | 56/96 (.5833) |
| fit | small_enclosed_maze | 2 | 37/96 (.3854) | 0/96 | 37/96 (.3854) | 65/96 (.6771) |
| fit | small_enclosed_maze | 3 | 48/96 (.5000) | 0/96 | 47/96 (.4896) | 71/96 (.7396) |
| fit | loop_alias_stress | 1 | 30/96 (.3125) | 12/96 (.1250) | 34/96 (.3542) | 51/96 (.5312) |
| fit | loop_alias_stress | 2 | 47/96 (.4896) | 12/96 (.1250) | 51/96 (.5312) | 66/96 (.6875) |
| fit | loop_alias_stress | 3 | 59/96 (.6146) | 12/96 (.1250) | 61/96 (.6354) | 76/96 (.7917) |
| calibration | overall | 1 | 26/96 (.2708) | 12/96 (.1250) | 30/96 (.3125) | 54/96 (.5625) |
| calibration | overall | 2 | 40/96 (.4167) | 12/96 (.1250) | 46/96 (.4792) | 68/96 (.7083) |
| calibration | overall | 3 | 50/96 (.5208) | 12/96 (.1250) | 53/96 (.5521) | 72/96 (.7500) |
| calibration | large_enclosed_maze | 1 | 8/24 (.3333) | 12/24 (.5000) | 5/24 (.2083) | 16/24 (.6667) |
| calibration | large_enclosed_maze | 2 | 13/24 (.5417) | 12/24 (.5000) | 12/24 (.5000) | 19/24 (.7917) |
| calibration | large_enclosed_maze | 3 | 15/24 (.6250) | 12/24 (.5000) | 13/24 (.5417) | 19/24 (.7917) |
| calibration | medium_enclosed_maze | 1 | 12/24 (.5000) | 0/24 | 13/24 (.5417) | 17/24 (.7083) |
| calibration | medium_enclosed_maze | 2 | 14/24 (.5833) | 0/24 | 17/24 (.7083) | 19/24 (.7917) |
| calibration | medium_enclosed_maze | 3 | 17/24 (.7083) | 0/24 | 18/24 (.7500) | 21/24 (.8750) |
| calibration | small_enclosed_maze | 1 | 1/24 (.0417) | 0/24 | 4/24 (.1667) | 5/24 (.2083) |
| calibration | small_enclosed_maze | 2 | 3/24 (.1250) | 0/24 | 9/24 (.3750) | 11/24 (.4583) |
| calibration | small_enclosed_maze | 3 | 7/24 (.2917) | 0/24 | 11/24 (.4583) | 13/24 (.5417) |
| calibration | loop_alias_stress | 1 | 5/24 (.2083) | 0/24 | 8/24 (.3333) | 16/24 (.6667) |
| calibration | loop_alias_stress | 2 | 10/24 (.4167) | 0/24 | 8/24 (.3333) | 19/24 (.7917) |
| calibration | loop_alias_stress | 3 | 11/24 (.4583) | 0/24 | 11/24 (.4583) | 19/24 (.7917) |
| heldout | overall | 1 | 6/96 (.0625) | 0/96 | 16/96 (.1667) | 35/96 (.3646) |
| heldout | overall | 2 | 14/96 (.1458) | 0/96 | 33/96 (.3438) | 48/96 (.5000) |
| heldout | overall | 3 | 24/96 (.2500) | 0/96 | 44/96 (.4583) | 58/96 (.6042) |
| heldout | large_enclosed_maze | 1 | 5/24 (.2083) | 0/24 | 1/24 (.0417) | 16/24 (.6667) |
| heldout | large_enclosed_maze | 2 | 6/24 (.2500) | 0/24 | 8/24 (.3333) | 19/24 (.7917) |
| heldout | large_enclosed_maze | 3 | 8/24 (.3333) | 0/24 | 10/24 (.4167) | 20/24 (.8333) |
| heldout | medium_enclosed_maze | 1 | 0/24 | 0/24 | 6/24 (.2500) | 6/24 (.2500) |
| heldout | medium_enclosed_maze | 2 | 2/24 (.0833) | 0/24 | 6/24 (.2500) | 6/24 (.2500) |
| heldout | medium_enclosed_maze | 3 | 4/24 (.1667) | 0/24 | 8/24 (.3333) | 9/24 (.3750) |
| heldout | small_enclosed_maze | 1 | 0/24 | 0/24 | 2/24 (.0833) | 5/24 (.2083) |
| heldout | small_enclosed_maze | 2 | 1/24 (.0417) | 0/24 | 7/24 (.2917) | 9/24 (.3750) |
| heldout | small_enclosed_maze | 3 | 5/24 (.2083) | 0/24 | 11/24 (.4583) | 14/24 (.5833) |
| heldout | loop_alias_stress | 1 | 1/24 (.0417) | 0/24 | 7/24 (.2917) | 8/24 (.3333) |
| heldout | loop_alias_stress | 2 | 5/24 (.2083) | 0/24 | 12/24 (.5000) | 14/24 (.5833) |
| heldout | loop_alias_stress | 3 | 7/24 (.2917) | 0/24 | 15/24 (.6250) | 15/24 (.6250) |

## Model, gradient-scale smoke, and training

The continued predictor has 17,201,920 parameters. The shared convolutional
safety branch has 191,404 parameters, below the 250,000-parameter limit. It
produces 12 logits: three cumulative horizons by contact, clearance, stuck, and
aggregate unsafe.

The fit-only smoke passed token/label alignment, both safety paths, action/control
sensitivity, finite losses and gradients, predictor and output-gradient coverage,
checkpoint reload, deterministic inference, and exact historical behavior with
the safety branch disabled. The initial shared-gradient norms were 0.262092
(JEPA) and 0.009716 (unscaled predicted safety), yielding frozen
`lambda_safety = 6.7438008` and the required initial scaled ratio 0.25.

The monitored ratio subsequently drifted: the median pre-clip safety-to-JEPA
shared-gradient ratio was 24.1158 and the maximum was 602.0971. The existing
combined predictor-gradient clipping was applied at every update, but no
objective-specific post-clip ratio was retained. Thus the required sustained
0.10–0.50 ratio and post-clip ≤1.0 evidence were not met. This is reported, not
retuned after results.

| Epoch | Original JEPA loss | Route JEPA | Safety true | Safety predicted | Route total |
|---:|---:|---:|---:|---:|---:|
| 1 | .645576 | .460329 | 1.104198 | 1.063074 | 15.075977 |
| 2 | .638937 | .463736 | .878854 | .878393 | 12.314260 |
| 3 | .635853 | .463636 | .755487 | .710379 | 10.349140 |
| 4 | .633714 | .468312 | 1.278987 | 1.155394 | 16.885290 |
| 5 | .631761 | .466942 | .757398 | .684751 | 10.192510 |
| 6 | .630018 | .466733 | .593116 | .523219 | 7.995074 |
| 7 | .628568 | .466748 | .501468 | .436588 | 6.792814 |
| 8 | .627386 | .467080 | .529602 | .499345 | 7.406094 |
| 9 | .626143 | .467056 | .448375 | .416665 | 6.300716 |
| 10 | .625140 | .470037 | .417149 | .366896 | 5.757481 |
| 11 | .624076 | .469525 | .379190 | .320719 | 5.189573 |
| 12 | .623001 | .469918 | .357977 | .367071 | 5.359503 |

Every epoch consumed all 981 original JEPA batches and 96 safety batches. The
final-epoch-only checkpoint is
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/safety_auxiliary_two_step_jepa_dev_v1/safety_auxiliary_two_step_seed_2026082004_epoch12.pt`,
208,848,287 bytes, SHA-256
`b4945e49c6680ca8b89cafa67d6aa75689b70ce84ac0ec3764dcdeaecde33295`.

## Calibration

The one scalar H3 aggregate-unsafe temperature was `3.4965773`. The frozen
calibration rule selected threshold `0.0`, with calibration recall 1.0. Under the
registered admission convention (`unsafe_probability < threshold`), this rejects
every candidate. The criterion could achieve recall but not useful retention.

## Held-out actual-future safety result

| Condition | AUC | AP | Recall | FNR | Safe retention | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action-only baseline | .7471 | .8350 | 1.0000 | 0 | .0526 | .1165 | .2088 |
| Prior post-hoc current context | .6606 | .7590 | 1.0000 | 0 | 0 | .1377 | .2287 |
| Augmented current-context path | .7305 | .8319 | 1.0000 | 0 | 0 | .1234 | .2270 |
| Augmented actual-future path | .7459 | .8428 | 1.0000 | 0 | 0 | .1589 | .2122 |
| Privileged static-grid guard | NA | NA | .6724 | .3276 | .5789 | NA | NA |
| Oracle safety | 1.0000 | 1.0000 | 1.0000 | 0 | 1.0000 | 0 | 0 |

Actual-future component diagnostics:

| Component | Positive rows | AUC | Recall | FNR | Precision |
|---|---:|---:|---:|---:|---:|
| Collision/contact | 24 | .6429 | 1.0000 | 0 | .2500 |
| Clearance | 0 | NA | NA | NA | NA |
| Stuck | 44 | .8680 | 1.0000 | 0 | .4583 |
| Fall/unsafe termination | 0 | NA | NA | NA | NA |

Per-family actual-future aggregate safety:

| Family | AUC | Recall | FNR | Safe retention |
|---|---:|---:|---:|---:|
| large_enclosed_maze | .5875 | 1.0000 | 0 | 0 |
| medium_enclosed_maze | .7852 | 1.0000 | 0 | 0 |
| small_enclosed_maze | .8143 | 1.0000 | 0 | 0 |
| loop_alias_stress | .8444 | 1.0000 | 0 | 0 |

The actual-future gate passed recall, FNR, contact recall, stuck recall, and the
no-only-unsafe-admission checks. It failed aggregate AUC, safe retention, ECE,
and retention of a safe candidate in six states. There were no no-safe-candidate
held-out states, so that abstention condition was correctly treated as vacuously
satisfied.

## Guarded kinematic planning

The augmented actual-future condition admitted zero candidates in all eight
states. It therefore selected no unsafe branch, but produced eight false
abstentions, zero selected route progress, undefined normalized regret, and zero
best-safe top-3 recovery. Oracle safety retained a safe candidate in 8/8 states,
selected mean distance progress 0.2184 m, achieved zero normalized regret and
best-safe top-3 of 1.0. Action-only safety retained a safe candidate in 2/8
states and selected mean progress 0.4238 m only on that selected subset.

Because the actual-future gate failed, predicted-future safety scoring was not
opened. No matched one-step model or additional predictor seed was trained or
evaluated.

## Predictive non-regression

The mandatory frozen 240-branch counterfactual evaluation was run against the
historical two-step checkpoint. Values are augmented / historical.

| H | Changed-token cosine | Normalized error | Top-1 | MRR | Pairwise |
|---:|---:|---:|---:|---:|---:|
| 1 | .735437 / .735554 | .521061 / .520531 | .277778 / .234375 | .495828 / .458571 | .745107 / .720013 |
| 2 | .716681 / .711919 | .508103 / .516031 | .293403 / .251736 | .504251 / .464831 | .750158 / .722538 |
| 3 | .701771 / .697965 | .528174 / .534374 | .328125 / .272569 | .521996 / .467541 | .770676 / .734059 |
| 4 | .691494 / .686581 | .541182 / .549438 | .298611 / .234375 | .499490 / .441940 | .748895 / .711806 |

| H | Augmented occupied IoU | Historical occupied IoU |
|---:|---:|---:|
| 2 | .195434 | .184200 |
| 3 | .160739 | .150946 |
| 4 | .127786 | .129627 |

All five frozen H3 non-regression limits passed. H3 family diagnostics are:

| Family | Cosine aug/base | Error aug/base | Top-1 aug/base | Pairwise aug/base | Augmented occupancy IoU |
|---|---:|---:|---:|---:|---:|
| large_enclosed_maze | .6986/.6918 | .5339/.5469 | .3333/.3333 | .7803/.7652 | .1395 |
| local_composite_motifs | .6165/.6103 | .6432/.6536 | .2222/.1389 | .6212/.5783 | .0784 |
| loop_alias_stress | .7171/.7201 | .4931/.4874 | .3889/.2778 | .8838/.8535 | .1606 |
| medium_enclosed_maze | .7043/.7063 | .5248/.5194 | .3056/.2222 | .7929/.7702 | .1393 |
| open_obstacle_field | .7680/.7387 | .4173/.4680 | .3333/.1667 | .8068/.6553 | .1082 |
| rough_local_dynamics | .7345/.7390 | .5075/.4985 | .3750/.4167 | .7765/.8030 | 0 |
| small_enclosed_maze | .6807/.6829 | .5711/.5671 | .2500/.2917 | .6818/.6818 | .2303 |
| visual_sensor_stress | .6945/.6947 | .5345/.5340 | .4167/.3333 | .8220/.7652 | .4338 |

Thus no `PREDICTIVE_NON_REGRESSION_FAILURE` was added. The safety failure was not
hidden by a predictive-fidelity trade-off.

## Decision, runtime, storage, and custody

Final classification: `SAFETY_AUXILIARY_JEPA_DEVELOPMENT_NO_SIGNAL`.

The actual-future representation did not cross its primary AUC, retention, ECE,
or state-coverage gates; it also did not improve the safety-retention trade-off
over action-only. Predicted-future evaluation was therefore correctly skipped.
In addition, sustained gradient-scale control did not meet the frozen monitoring
requirements. The result does not authorise a matched one-step continuation.

The training runtime was not durably serialized because the first post-training
result-reduction attempt failed on ordinary JSON/report plumbing. The auditable
source-creation-to-final-checkpoint interval is at most 2,346.9 seconds; the final
checkpoint-only calibration, safety reduction, and non-regression run took 57.84
seconds. The checkpoint directory occupies 208,848,287 bytes; generated fixture
and machine-result files occupy 546,032 bytes.

| Artefact | SHA-256 |
|---|---|
| Training/evaluation source | `ec78b0a70406afbc2c08fa52f393fbbd46c6fc3021bea206a1812dd265025938` |
| Final checkpoint | `b4945e49c6680ca8b89cafa67d6aa75689b70ce84ac0ec3764dcdeaecde33295` |
| Evaluator fixture | `2cdfb782112d7aaf32bc1c6583b4b708634794fc047ec4a31bb2893b8368d145` |
| Machine result | `8c817488a424e1a7874c4eaf9dc8255b81a88d9a644f02fac3adc72c1969b184` |

Exactly one continued two-step model was trained, with seed `2026082004`.
No matched one-step model, additional training seed, simulator, renderer, encoder,
global memory, novelty layer, closed-loop navigation, or beacon-capture layer was
run. Nothing remained running at handoff.

# FACTORISED_MICRO_SAFETY_DATA_SCALING_V2 result

Date: 20 August 2026  
Source commit: `056cc7d4b18384be97d9352eacd3b3409f146df6`  
Final classification: `FACTORISED_MICRO_SAFETY_DATA_SCALING_NO_SIGNAL`

## Conclusion

Increasing the nested training inventory from 48 to 192 states improved fresh scene-disjoint contact AUC from 0.6997 to 0.8487 and stuck AUC from 0.7233 to 0.8328. It did not recover a useful safety–mobility operating point. FIT-192 retained only 14/102 safe branches (0.1373) and a safe candidate in 10/24 states; it falsely abstained in 11 states, selected four unsafe candidates, had normalized safe-progress regret 0.3378, and recovered the best safe candidate in its admitted top three in only 0.1250 of states.

The ordered primary curve was not broadly monotonic (5/8 quantities), safe retention increased by only 0.0686, and state retention increased by four rather than six. No post-hoc held-out threshold pair passed the complete prospective gate at any fit size. The unchanged factorised architecture is therefore closed under the current enhanced embodied sensor and candidate contract. The exact next decision is `CHANGED_GEOMETRY_SENSOR_CONTRACT_OR_NARROWER_SAFETY_CLAIM`.

This preserves `FACTORISED_MICRO_SAFETY_TRUE_FUTURE_NO_SIGNAL` and does not reinterpret either result as an information-theoretic sensor no-go.

## Bindings and inventories

- FIT-48: original 48 states / 576 branches; existing checkpoint `93f919238ff7b757b77f5281f45c59818c9f2b33fa5fbd96a2554b7aea14776e`; not retrained.
- FIT-96: FIT-48 plus the former 24-state calibration and 24-state held-out panels; 96 states / 1,152 branches.
- FIT-192: FIT-96 plus 96 prospectively frozen new fit states; 192 states / 2,304 branches.
- Fresh calibration: 24 states / 288 branches, six states per family.
- Fresh held-out: 24 states / 288 branches, six states per family.
- New panel manifest digest: `350e914c58757e0b32cc146f0d80d9dd877bd8307e179643bf1ddb4d5d329957`.
- New sensor-index digest: `a37e0c3aaffc3afeb71471c5a13bb8b16632003bdcf313099e8454cadd3fad69`.
- All 144 new state identities were frozen before candidate execution. They contain 144 distinct scenes and clusters, with zero scene overlap against the original FIT-48, the preceding fresh 48-state panel, and predictor training/selection scenes.
- Collection produced 1,728/1,728 finite branches with exact requested and post-slew action identity and complete 15-tick pose/safety traces.

### Panel prevalence

| Inventory | Safe | Unsafe | Contact + | Stuck + | Overlap | States with no safe candidate |
|---|---:|---:|---:|---:|---:|---:|
| FIT-96 | 374 | 778 | 453 | 600 | 293 | 13 |
| FIT-192 | 689 | 1,615 | 993 | 1,252 | 648 | 35 |
| Calibration | 88 | 200 | 136 | 157 | 93 | 4 |
| Held-out | 102 | 186 | 123 | 133 | 70 | 0 |

Both fresh splits contained safe and unsafe branches, at least 24 positives for each component, at least four safe-candidate states per family, and contact and stuck examples in every family. The frozen adequacy gate passed.

## Evaluator and unchanged architecture

The complete synthetic fixture passed transient contact, persistent contact, delayed stuck, safe branch, all-unsafe, one-safe, separate-component, deterministic OR, strict threshold tie, abstention, deterministic kinematic selection, and row-ledger serialization cases.

The architecture and objectives were unchanged:

- independent contact specialist: 97,346 parameters;
- independent stuck specialist: 107,906 parameters;
- total: 205,252 parameters;
- shared model parameters and normalization statistics: zero;
- balanced active-event BCE, cumulative BCE, and H3 within-state pairwise ranking with weight 0.25.

## Seeds, training, and checkpoints

One seed family, `2026082011`, was used. Condition names were included in the SHA-256 RNG derivation.

| Condition | Derived seed | Epoch-60 contact loss | Epoch-60 stuck loss | Runtime | Peak VRAM | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| FIT-48 | historical `2026082010` | preserved | preserved | not retrained | not retrained | `93f919238ff7b757b77f5281f45c59818c9f2b33fa5fbd96a2554b7aea14776e` |
| FIT-96 | 104285942 | 0.002991 | 0.054468 | 38.74 s | 74,531,328 B | `3e8b87e12b1fcee2db05c6aa5ea2c46557bc9abf80d0ad5f94f34af863eeede0` |
| FIT-192 | 1832144148 | 0.001861 | 0.010596 | 77.22 s | 75,354,624 B | `a778cc82293eb1e28be991f0c4fbc5d9d5cd87ba5c1fc2bf321028321f5806f4` |

Both new conditions passed their fit-only input, leakage, separation, gradient, temporal-order, action-sensitivity, save/reload, and deterministic-inference smoke tests before calibration or held-out access. Final epoch only was used.

## Common fresh calibration

| Condition | Contact temperature | Stuck temperature | Contact threshold | Stuck threshold | Eligible frontier pairs / all | Calibration recall (unsafe/contact/stuck) | Safe retained | Safe states | Selected unsafe |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| FIT-48 | 9.70239 | 5.37930 | 0.133353 | 0.258988 | 2,466 / 84,100 | 0.995 / 0.963 / 0.924 | 4/88 | 3/24 | 0 |
| FIT-96 | 5.44597 | 4.76812 | 0.084779 | 0.187267 | 3,998 / 81,200 | 0.995 / 0.993 / 0.904 | 16/88 | 8/24 | 0 |
| FIT-192 | 6.66747 | 6.80041 | 0.127570 | 0.167022 | 5,613 / 80,910 | 0.995 / 0.934 / 0.955 | 18/88 | 10/24 | 0 |

The joint rule and all thresholds were selected from calibration only and frozen before held-out scoring.

## Fresh held-out component metrics

### Contact specialist

| Condition | AUC | AP | Recall | FNR | ECE | Brier | Event-tick recall | Detection delay | Transient miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIT-48 | 0.6997 | 0.7070 | 0.9512 | 0.0488 | 0.1349 | 0.2278 | 0.8186 | 0 ticks | 0.2545 |
| FIT-96 | 0.7907 | 0.8117 | 0.8862 | 0.1138 | 0.0762 | 0.1705 | 0.8101 | 0 ticks | 0.2545 |
| FIT-192 | 0.8487 | 0.8705 | 0.9187 | 0.0813 | 0.0560 | 0.1367 | 0.8439 | 0 ticks | 0.2364 |

### Stuck specialist

| Condition | AUC | AP | Recall | FNR | ECE | Brier | Event-tick recall | Detection delay | Transient miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIT-48 | 0.7233 | 0.6581 | 0.9323 | 0.0677 | 0.0630 | 0.2099 | 0.7770 | 0 ticks | 0.2549 |
| FIT-96 | 0.7832 | 0.6771 | 0.9398 | 0.0602 | 0.0970 | 0.1944 | 0.6760 | 0 ticks | 0.3922 |
| FIT-192 | 0.8328 | 0.7476 | 0.9699 | 0.0301 | 0.0683 | 0.1687 | 0.7944 | 0 ticks | 0.2549 |

Training AUC was effectively 1.0 for both components at all fit sizes. Held-out-minus-fit AUC gaps narrowed from -0.3003/-0.2766 at FIT-48 to -0.1513/-0.1672 at FIT-192, but remained large.

## Combined filtering and kinematic planning

| Condition | Unsafe recall / FNR | Safe retained | Safe states | Only-unsafe admitted states | Selected unsafe | False abstentions | Selected progress | Oracle fraction | Normalized regret | Best-safe top-1 / top-3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FIT-48 | 0.9946 / 0.0054 | 7/102 (0.0686) | 6/24 | 0 | 1 | 18 | 0.1424 m | 0.7249 | 0.4084 | 0.0833 / 0.1250 |
| FIT-96 | 0.9677 / 0.0323 | 14/102 (0.1373) | 10/24 | 1 | 2 | 13 | 0.1273 m | 0.6481 | 0.3366 | 0.1667 / 0.1667 |
| FIT-192 | 0.9624 / 0.0376 | 14/102 (0.1373) | 10/24 | 3 | 4 | 11 | 0.1853 m | 0.9438 | 0.3378 | 0.0833 / 0.1250 |

FIT-192 passed unsafe recall/FNR, both component-recall gates, contact AUC, and oracle-progress fraction. It failed safe retention, stuck AUC, state retention, only-unsafe-state, zero-unsafe-selection, false-abstention, regret, best-safe-top-3, and family-collapse gates.

## Learning curve

| Quantity | FIT-48 | FIT-96 | FIT-192 | 48→96 | 96→192 | Monotonic in desired direction |
|---|---:|---:|---:|---:|---:|---|
| Contact AUC | 0.6997 | 0.7907 | 0.8487 | +0.0910 | +0.0580 | yes |
| Stuck AUC | 0.7233 | 0.7832 | 0.8328 | +0.0600 | +0.0496 | yes |
| Safe retention | 0.0686 | 0.1373 | 0.1373 | +0.0686 | +0.0000 | yes |
| States retaining safe | 6 | 10 | 10 | +4 | +0 | yes |
| False abstentions | 18 | 13 | 11 | -5 | -2 | yes |
| Selected progress | 0.1424 | 0.1273 | 0.1853 | -0.0151 | +0.0581 | no |
| Normalized regret | 0.4084 | 0.3366 | 0.3378 | -0.0719 | +0.0012 | no |
| Best-safe top-3 | 0.1250 | 0.1667 | 0.1250 | +0.0417 | -0.0417 | no |

Seven of eight primary quantities improved directionally from FIT-48 to FIT-192, but only five were monotonic. The prespecified positive-tendency requirements also failed because safe retention rose by 0.0686 rather than 0.20, only four additional states retained a safe candidate rather than six, and FIT-192 introduced additional unsafe selections.

## Per-family results

Columns are contact AUC, stuck AUC, aggregate unsafe recall, safe retention, safe states, selected unsafe, false abstentions, selected progress, oracle fraction, regret, and best-safe top-3.

| Fit | Family | C-AUC | S-AUC | Recall | Retention | Safe states | Unsafe selected | False abstain | Progress | Oracle frac. | Regret | Top-3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | large | 0.744 | 0.832 | 1.000 | 0.192 | 4 | 0 | 2 | 0.118 | 0.580 | 0.413 | 0.333 |
| 48 | medium | 0.718 | 0.698 | 0.978 | 0.037 | 1 | 1 | 5 | 0.231 | 0.931 | NA | 0.167 |
| 48 | small | 0.766 | 0.723 | 1.000 | 0.037 | 1 | 0 | 5 | 0.152 | 0.617 | 0.389 | 0.000 |
| 48 | loop | 0.673 | 0.725 | 1.000 | 0.000 | 0 | 0 | 6 | 0.000 | 0.000 | NA | 0.000 |
| 96 | large | 0.709 | 0.875 | 0.891 | 0.231 | 4 | 1 | 2 | 0.116 | 0.569 | 0.333 | 0.333 |
| 96 | medium | 0.858 | 0.768 | 1.000 | 0.185 | 3 | 0 | 3 | 0.247 | 0.998 | 0.104 | 0.333 |
| 96 | small | 0.780 | 0.725 | 0.978 | 0.037 | 1 | 1 | 4 | 0.289 | 1.174 | 0.389 | 0.000 |
| 96 | loop | 0.832 | 0.820 | 1.000 | 0.091 | 2 | 0 | 4 | -0.191 | -2.155 | 0.548 | 0.000 |
| 192 | large | 0.731 | 0.894 | 0.957 | 0.192 | 4 | 2 | 1 | 0.178 | 0.876 | 0.333 | 0.500 |
| 192 | medium | 0.914 | 0.807 | 0.978 | 0.185 | 3 | 1 | 2 | 0.178 | 0.721 | 0.444 | 0.000 |
| 192 | small | 0.875 | 0.802 | 0.956 | 0.074 | 2 | 1 | 3 | 0.285 | 1.157 | 0.254 | 0.000 |
| 192 | loop | 0.882 | 0.820 | 0.960 | 0.091 | 1 | 0 | 5 | -0.049 | -0.549 | 0.202 | 0.000 |

Every condition had at least one family with no retained safe candidate or unsafe selection, so the no-family-collapse gate failed.

## Per-state selection values

Cell format is `selected-candidate/admitted/admitted-safe/admitted-unsafe/selected-distance-progress-m`; `H` means hold and `!` marks an unsafe selected candidate.

| State | Family | FIT-48 | FIT-96 | FIT-192 |
|---|---|---|---|---|
| scale-held-0-00 | large | 11/1/1/0/0.133 | 2!/3/1/2/0.251 | 11/1/1/0/0.133 |
| scale-held-0-01 | large | H/0/0/0/NA | 11/5/2/3/-0.002 | 1!/1/0/1/-0.288 |
| scale-held-0-02 | large | H/0/0/0/NA | H/0/0/0/NA | 0!/2/1/1/0.434 |
| scale-held-0-03 | large | 1/1/1/0/-0.272 | H/0/0/0/NA | H/0/0/0/NA |
| scale-held-0-04 | large | 0/1/1/0/0.464 | 11/1/1/0/0.067 | 0/1/1/0/0.464 |
| scale-held-0-05 | large | 0/2/2/0/0.147 | 0/2/2/0/0.147 | 0/2/2/0/0.147 |
| scale-held-1-00 | medium | H/0/0/0/NA | H/0/0/0/NA | 11!/1/0/1/0.005 |
| scale-held-1-01 | medium | H/0/0/0/NA | 4/2/2/0/0.325 | H/0/0/0/NA |
| scale-held-1-02 | medium | H/0/0/0/NA | 1/2/2/0/0.306 | 2/1/1/0/0.266 |
| scale-held-1-03 | medium | 4!/2/1/1/0.231 | 11/1/1/0/0.111 | H/0/0/0/NA |
| scale-held-1-04 | medium | H/0/0/0/NA | H/0/0/0/NA | 11/1/1/0/0.138 |
| scale-held-1-05 | medium | H/0/0/0/NA | H/0/0/0/NA | 0/3/3/0/0.305 |
| scale-held-2-00 | small | H/0/0/0/NA | 1!/1/0/1/0.426 | 1!/1/0/1/0.426 |
| scale-held-2-01 | small | 11/1/1/0/0.152 | 11/1/1/0/0.152 | H/0/0/0/NA |
| scale-held-2-02 | small | H/0/0/0/NA | H/0/0/0/NA | 2/2/1/1/0.291 |
| scale-held-2-03 | small | H/0/0/0/NA | H/0/0/0/NA | 3/1/1/0/0.136 |
| scale-held-2-04 | small | H/0/0/0/NA | H/0/0/0/NA | H/0/0/0/NA |
| scale-held-2-05 | small | H/0/0/0/NA | H/0/0/0/NA | H/0/0/0/NA |
| scale-held-3-00 | loop | H/0/0/0/NA | 4/1/1/0/-0.038 | H/0/0/0/NA |
| scale-held-3-01 | loop | H/0/0/0/NA | 0/1/1/0/-0.344 | 11/4/2/2/-0.049 |
| scale-held-3-02 | loop | H/0/0/0/NA | H/0/0/0/NA | H/0/0/0/NA |
| scale-held-3-03 | loop | H/0/0/0/NA | H/0/0/0/NA | H/0/0/0/NA |
| scale-held-3-04 | loop | H/0/0/0/NA | H/0/0/0/NA | H/0/0/0/NA |
| scale-held-3-05 | loop | H/0/0/0/NA | H/0/0/0/NA | H/0/0/0/NA |

Complete unrounded per-state values, including selected heading improvement and oracle candidate progress, are preserved in `result.json` and in each condition's row-level ledger.

## Post-hoc held-out oracle frontiers

These are diagnostic only; no held-out threshold replaced the calibration-selected rule.

| Condition | Pairs | Max safe retention at recall ≥0.95 | Max safe states at recall ≥0.95 | Max progress with zero unsafe selections | Min regret at recall ≥0.95 | Any complete-gate pair |
|---|---:|---:|---:|---:|---:|---|
| FIT-48 | 84,100 | 0.2353 | 14 | 0.4637 m | 0.0000 | no |
| FIT-96 | 82,360 | 0.3725 | 18 | 0.4012 m | 0.0000 | no |
| FIT-192 | 82,940 | 0.3824 | 18 | 0.1380 m | 0.1009 | no |

Because no oracle-frontier threshold pair passed the complete gate, calibration is not the sole bottleneck at any scale.

## Row-level evidence

Each immutable ledger contains all 576 common fresh calibration and held-out rows with raw per-tick logits, calibrated probabilities, component and aggregate labels, strict threshold decisions, admitted sets, selected candidates, action/control inputs, and route metrics.

| Condition | Ledger content digest | Ledger SHA-256 |
|---|---|---|
| FIT-48 | `78ea074a53dc6e5a58bf9084b7688c6add1504987ae525cd0c442501d02214e2` | `547a5068a0cee440dd7f43e8a4aab562e139e80b816d33814361c0c1e3410f48` |
| FIT-96 | `b984bc609087f05080b868e27d560ce3a778b2c0efa3dd1b86166e6457a34b05` | `1a8f364acbc25dae478769479bb2114df92c300cd378c07915ee52a14b34a9fd` |
| FIT-192 | `f584597e88ecf1b3fbd7c93030c00c5184e2946735d7fd2f9eb5a096598c1f50` | `e79d7fce32f5170209ef961810c6cf216943d73e2442ce809b523abfe8a66e63` |

## Runtime, storage, and custody

- Eligibility: 256 pre-outcome receipts; 7,497.60 simulator-compute seconds.
- New branch generation: 1,242.09 wall seconds; 4,802.25 simulator-compute seconds.
- Training: 38.74 s FIT-96 and 77.22 s FIT-192.
- Training, calibration, three evaluations, ledgers, and oracle frontiers: 233.48 s total script runtime.
- New sensor shards: 6,917,795 bytes.
- New checkpoints: 1,670,022 bytes total.
- Row ledgers: 782,431 bytes total.
- V2 generated directory: 16,391,388 bytes; high-capacity cache directory: 21,716,503 bytes.

Exactly two new fit-size conditions were trained under one keyed seed family. FIT-48 was not retrained. No JEPA predictor, RGB model, depth/LiDAR model, memory, novelty, routing, or navigation system was opened or trained. No process remained running at commit.

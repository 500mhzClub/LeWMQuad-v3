# Go2 counterfactual predictor qualification v1.2

Date: 2026-08-11
Status: complete, development-only counterfactual assay

## Scope and interpretation

This assay asks whether the 32 frozen epoch-21 world models predict the *correct future for a proposed action*. It reuses the 20 successful oracle-v1.2 pilot states, executes all 12 frozen candidates at each state, and compares every predicted H=1-4 latent future directly with the realised target future. The eight training-seed quadruplets are the replication units. Equal-family estimates are primary; corpus-weighted estimates are separate. Horizons are not combined.

No utility/energy scorer was trained or invoked. The 120-state scorer-fit corpus and the 200-state final utility corpus were not generated. No candidate was ranked by oracle utility. Consequently, these results qualify direct counterfactual fidelity and action specificity; they do **not** establish a planning-utility improvement.

The exact machine-readable Stage B/C result is
[`result.json`](../.generated/go2_counterfactual_fidelity_v1_2/predictor_assay/result.json), with canonical report digest
`3b5c500b4b1326056ce18c6276d7842f4230faec36f8f29cc65945f54527bbcb` and prospective assay-spec digest
`a26fa0ec9ee9e0df3bbe71fff6d7594bb714227aaa66a66631836d94a676feab`.

## Frozen lineage and Stage A corpus

The source state manifest is the successful oracle-v1.2 pilot manifest
`5f380bf7f49ef10437c7d9644f04dbef065f0550dfd30d0ec36208cda25d08cf`.
The candidate bank (`85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9`), continuous-progress contract (`840328d918f446bad1a5855e72f13f8937fc9a42eafd87818bf8cd94305e2c3d`), graded-safety contract (`5cf4572be2490c1b6f748abc704fff3a3c15fb1ea8dc060e49314e2bbaf01e0f`) and complete oracle v1.2 (`3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4`) were unchanged.

| Artefact | Bound value |
|---|---|
| Stage-A assay specification | `39545af7599da2f2a1bf171c050489eea9f8637137bc1a9c0af3a193d1aaaf3a` |
| 240-branch identity manifest | `ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a` |
| Corpus content digest | `f84eb3271f1a3b7052bbf2e84240453e84772b0a530e60ec47f723a44e2e10e9` |
| Branch ledger SHA-256 | `2b71c488851c6d4b7e3a36a46637a4e5be4896ae48a84d1498c6e8a8d3d74c81` |
| Completion receipt | `b448775b8c62539e5b5f9b3c1f0d2d86da85f40311a38a4f6b2cef550cbb0c2f` |
| Latent index | `861285ec9c8fc6c92c6f3a31cade0f031172bf6818d76d1899634a60c7e5c291` |
| Verified latent shard set | `eeb381e28d851db60d4341654860b77e9a0aef0abae1c3e7673d75d82bc5916f` |

All 240/240 branches were valid and all 240 reproduced the preserved oracle outcome; there were no invalid branches. The corpus contains 20 states, 12 candidates per state, four future observations per branch, 20 context-latent shards and 240 horizon-latent shards. Family state counts were 3 each for `large_enclosed_maze`, `local_composite_motifs`, `loop_alias_stress`, and `medium_enclosed_maze`, and 2 each for `open_obstacle_field`, `rough_local_dynamics`, `small_enclosed_maze`, and `visual_sensor_stress`.

## Rendering, preprocessing and target encoding verification

The textured-v03 renderer contract is
`df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b`;
the render contract is
`2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17`;
the centre-crop/preprocessing contract and preprocessing digests are
`2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9` and
`8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5`.

The frozen V-JEPA 2.1 target encoder binding is
`15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5`;
its 5,151,198,524-byte checkpoint SHA-256 is
`7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`.
Stored targets are raw final-block tokens rounded to float16, shape
`[branch, 4, 768, 1024]`, with row-major 24x32 patch order from
`norms_block[-1]`; consumers reload float16 as float32 and apply layer
normalisation on the 1024-dimensional token axis.

The required six-branch smoke passed. Repeated H1 rendering produced identical
`[224,224,3] uint8` pixels with SHA-256
`5d443a450eb4724c5240f37c590881d4975b0c7d5f5b9b5caeefb44afcc7de2b`.
The smoke encoded context shape was `[1,3,768,1024]` and target shape was
`[6,4,768,1024]`; token order, atomic shard receipts and exact identity/branch
digests validated. A second invocation generated zero branch rows and zero
latent shards, establishing resume-only-missing behaviour without changing
the receipt, ledger or latent index.

## Stage B: direct correct-future fidelity

The changed-token mask and thresholds were frozen before target scoring. H1
uses the frozen step-1 threshold; H2-H4 reuse the frozen step-2 threshold.
Cosine is mean token cosine on the target-specific changed mask. Normalised
error is predictor MSE divided by persistence MSE on the same mask. In the
tables below, cell order is RGB one-step, RGB rollout, proprioceptive one-step,
proprioceptive rollout. `B_RGB` and `B_prop` are rollout benefits within their
respective input conditions; `M=(B_RGB+B_prop)/2`; `J=B_prop-B_RGB`. Positive
cosine `M` and positive normalised-error-reduction `M` favour rollout.

### Equal-family cell means

Each model cell is `correct cosine / advantage over persistence / normalised error`.

| Horizon | Persistence cosine | RGB one-step | RGB rollout | Prop one-step | Prop rollout |
|---|---:|---:|---:|---:|---:|
| H1 | 0.494265 | 0.735481 / 0.241216 / 0.520778 | 0.738227 / 0.243963 / 0.515413 | 0.736101 / 0.241837 / 0.519577 | 0.737921 / 0.243657 / 0.516018 |
| H2 | 0.443658 | 0.710005 / 0.266347 / 0.519603 | 0.715872 / 0.272213 / 0.509217 | 0.709732 / 0.266074 / 0.520194 | 0.715184 / 0.271526 / 0.510536 |
| H3 | 0.436567 | 0.688447 / 0.251879 / 0.551169 | 0.700001 / 0.263434 / 0.530915 | 0.688085 / 0.251518 / 0.551830 | 0.699841 / 0.263274 / 0.531226 |
| H4 | 0.431295 | 0.673646 / 0.242351 / 0.571136 | 0.687894 / 0.256599 / 0.546932 | 0.673637 / 0.242342 / 0.571232 | 0.687762 / 0.256467 / 0.547263 |

Every cell beat persistence at every horizon.

### Paired seed effects, equal-family primary

Values are means with two-sided 95% t-intervals over the eight training-seed
quadruplets.

| Metric | H | B_RGB | B_prop | M | J |
|---|---|---:|---:|---:|---:|
| Cosine | H1 | 0.002747 [0.001726, 0.003768] | 0.001820 [-0.000000, 0.003640] | 0.002283 [0.001091, 0.003476] | -0.000927 [-0.002665, 0.000811] |
| Cosine | H2 | 0.005866 [0.003214, 0.008519] | 0.005452 [0.002966, 0.007939] | 0.005659 [0.003354, 0.007964] | -0.000414 [-0.002691, 0.001864] |
| Cosine | H3 | 0.011555 [0.008595, 0.014515] | 0.011756 [0.008902, 0.014610] | 0.011655 [0.008977, 0.014334] | 0.000201 [-0.002062, 0.002465] |
| Cosine | H4 | 0.014248 [0.011557, 0.016939] | 0.014124 [0.011641, 0.016608] | 0.014186 [0.011859, 0.016513] | -0.000123 [-0.002395, 0.002148] |
| Normalised-error reduction | H1 | 0.005365 [0.003447, 0.007283] | 0.003559 [-0.000069, 0.007186] | 0.004462 [0.002161, 0.006763] | -0.001806 [-0.005343, 0.001730] |
| Normalised-error reduction | H2 | 0.010385 [0.005738, 0.015033] | 0.009658 [0.005212, 0.014104] | 0.010022 [0.005934, 0.014109] | -0.000727 [-0.004713, 0.003258] |
| Normalised-error reduction | H3 | 0.020254 [0.015003, 0.025505] | 0.020605 [0.015546, 0.025663] | 0.020429 [0.015654, 0.025205] | 0.000350 [-0.003532, 0.004233] |
| Normalised-error reduction | H4 | 0.024204 [0.019474, 0.028935] | 0.023969 [0.019682, 0.028256] | 0.024087 [0.020025, 0.028149] | -0.000235 [-0.004173, 0.003703] |

Rollout improved direct counterfactual fidelity at all four horizons. The
rollout benefit grew with horizon, while every cosine and normalised-error
proprioception-by-rollout interaction interval included zero.

## Stage C: action-specific branch retrieval

For each predicted candidate, the gallery contains all 12 realised candidate
targets from the same state. Similarity uses the complete aligned 768-token
grid; retrieval does not use the changed mask. Ties are broken by frozen
candidate index. Chance references are top-1 `1/12 = 0.083333`, top-3
`3/12 = 0.25`, MRR `0.258601`, mean rank `6.5`, and pairwise accuracy `0.5`.

### Equal-family cell means

Each cell is `top-1 / top-3 / MRR / mean rank / pairwise accuracy`.

| H | RGB one-step | RGB rollout | Prop one-step | Prop rollout |
|---|---:|---:|---:|---:|
| H1 | 0.2663 / 0.6094 / 0.4799 / 3.7014 / 0.7317 | 0.2661 / 0.6220 / 0.4836 / 3.6556 / 0.7359 | 0.2693 / 0.6152 / 0.4838 / 3.6753 / 0.7341 | 0.2739 / 0.6155 / 0.4869 / 3.6654 / 0.7350 |
| H2 | 0.2676 / 0.6194 / 0.4840 / 3.6682 / 0.7347 | 0.2826 / 0.6183 / 0.4935 / 3.6289 / 0.7383 | 0.2648 / 0.6263 / 0.4838 / 3.6656 / 0.7349 | 0.2791 / 0.6198 / 0.4916 / 3.6283 / 0.7383 |
| H3 | 0.2622 / 0.5933 / 0.4719 / 3.8633 / 0.7397 | 0.2882 / 0.6003 / 0.4869 / 3.7808 / 0.7472 | 0.2487 / 0.5820 / 0.4624 / 3.9195 / 0.7346 | 0.2845 / 0.6066 / 0.4861 / 3.7895 / 0.7464 |
| H4 | 0.2127 / 0.5278 / 0.4218 / 4.3635 / 0.6942 | 0.2268 / 0.5414 / 0.4366 / 4.2248 / 0.7068 | 0.2099 / 0.5271 / 0.4196 / 4.3809 / 0.6926 | 0.2376 / 0.5451 / 0.4426 / 4.2064 / 0.7085 |

All cell means outperformed their corresponding chance references (mean rank
was below chance; the other listed metrics were above chance). H3-H4 provide
the clearest paired rollout evidence: equal-family rollout `M` was positive
with an interval above zero for top-1, MRR, mean-rank reduction, and pairwise
accuracy at both horizons. H3 top-3 also had an interval above zero; the other
top-3 intervals included zero.

### Paired rollout effects, equal-family primary

For mean rank, positive `M` means a reduction in rank. The table gives `M`
and `J` with two-sided 95% t-intervals.

| Metric | H | M | J |
|---|---|---:|---:|
| Top-1 | H1 | 0.002170 [-0.010858, 0.015198] | 0.004774 [-0.012587, 0.022135] |
| Top-1 | H2 | 0.014648 [0.003290, 0.026007] | -0.000651 [-0.017541, 0.016239] |
| Top-1 | H3 | 0.030924 [0.014973, 0.046876] | 0.009766 [-0.031323, 0.050854] |
| Top-1 | H4 | 0.020942 [0.005971, 0.035913] | 0.013672 [-0.010879, 0.038223] |
| Top-3 | H1 | 0.006402 [-0.008536, 0.021340] | -0.012370 [-0.024121, -0.000618] |
| Top-3 | H2 | -0.003798 [-0.025270, 0.017674] | -0.005425 [-0.020093, 0.009242] |
| Top-3 | H3 | 0.015734 [0.002005, 0.029462] | 0.017578 [-0.008667, 0.043824] |
| Top-3 | H4 | 0.015842 [-0.005354, 0.037038] | 0.004340 [-0.028012, 0.036693] |
| MRR | H1 | 0.003440 [-0.006797, 0.013678] | -0.000500 [-0.012316, 0.011316] |
| MRR | H2 | 0.008662 [-0.000879, 0.018203] | -0.001661 [-0.014371, 0.011050] |
| MRR | H3 | 0.019361 [0.007964, 0.030758] | 0.008727 [-0.019584, 0.037037] |
| MRR | H4 | 0.018877 [0.007346, 0.030408] | 0.008258 [-0.012799, 0.029316] |
| Mean-rank reduction | H1 | 0.027886 [-0.055404, 0.111176] | -0.035807 [-0.113988, 0.042373] |
| Mean-rank reduction | H2 | 0.038303 [-0.045655, 0.122260] | -0.001953 [-0.083401, 0.079495] |
| Mean-rank reduction | H3 | 0.106228 [0.014905, 0.197552] | 0.047526 [-0.091616, 0.186668] |
| Mean-rank reduction | H4 | 0.156576 [0.053039, 0.260112] | 0.035807 [-0.130428, 0.202043] |
| Pairwise accuracy | H1 | 0.002535 [-0.005037, 0.010107] | -0.003255 [-0.010363, 0.003852] |
| Pairwise accuracy | H2 | 0.003482 [-0.004150, 0.011115] | -0.000178 [-0.007582, 0.007227] |
| Pairwise accuracy | H3 | 0.009657 [0.001355, 0.017959] | 0.004321 [-0.008329, 0.016970] |
| Pairwise accuracy | H4 | 0.014234 [0.004822, 0.023647] | 0.003255 [-0.011857, 0.018368] |

With the single H1 top-3 exception, action-specific interaction intervals
included zero. There is therefore no stable evidence that deployment-valid
proprioception changes the rollout benefit.

### Similarity margins, ties and confusion

Equal-family own-branch-minus-best-other margins remained negative, but
rollout made them less negative at every horizon. Own-branch-minus-mean-other
margins were positive in every cell.

| H | RGB one-step best / mean | RGB rollout best / mean | Prop one-step best / mean | Prop rollout best / mean |
|---|---:|---:|---:|---:|
| H1 | -0.03080 / 0.03797 | -0.02860 / 0.03673 | -0.03058 / 0.03838 | -0.02869 / 0.03692 |
| H2 | -0.03422 / 0.05574 | -0.03026 / 0.05286 | -0.03432 / 0.05517 | -0.03089 / 0.05285 |
| H3 | -0.04902 / 0.05079 | -0.04265 / 0.04878 | -0.04959 / 0.04989 | -0.04270 / 0.04890 |
| H4 | -0.06087 / 0.04437 | -0.04973 / 0.04404 | -0.06167 / 0.04386 | -0.04983 / 0.04391 |

The exact own-versus-wrong tie rate was `0.045455` for every H1/H2 cell and
zero for every H3/H4 cell. Each pooled confusion matrix contains 1,920
queries. Diagonal counts, in cell order RGB one-step / RGB rollout / prop
one-step / prop rollout, were H1 `509 / 507 / 513 / 525`, H2
`513 / 545 / 510 / 541`, H3 `497 / 550 / 474 / 544`, and H4
`410 / 439 / 404 / 461`. All sixteen full 12x12 cross-candidate confusion
matrices, plus their per-seed and per-family versions, are retained at
`paired_seed_analysis.retrieval_confusion_across_seeds` in the bound result.

## Weighting and heterogeneity

Corpus-weighted rollout `M` estimates are reported separately below; values
are means with two-sided 95% t-intervals.

| H | Cosine | Normalised-error reduction | Top-1 | Pairwise accuracy |
|---|---:|---:|---:|---:|
| H1 | 0.002809 [0.001148, 0.004470] | 0.005264 [0.002151, 0.008376] | 0.002604 [-0.010421, 0.015630] | 0.002652 [-0.004077, 0.009380] |
| H2 | 0.009655 [0.007284, 0.012025] | 0.016687 [0.012590, 0.020784] | 0.016406 [0.006147, 0.026666] | 0.003741 [-0.003454, 0.010935] |
| H3 | 0.017829 [0.015041, 0.020616] | 0.030560 [0.025782, 0.035338] | 0.032031 [0.018966, 0.045096] | 0.010582 [0.003989, 0.017175] |
| H4 | 0.022578 [0.019711, 0.025444] | 0.038502 [0.033613, 0.043390] | 0.022396 [0.007921, 0.036871] | 0.013613 [0.005173, 0.022052] |

Per-family rollout `M` values expose material heterogeneity. Each cell below
is `changed cosine / top-1 recovery`.

| Family | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| large_enclosed_maze | 0.00064 / -0.00174 | 0.00705 / 0.04340 | 0.01527 / 0.02951 | 0.01669 / -0.01910 |
| local_composite_motifs | 0.00531 / -0.00347 | 0.00309 / -0.01042 | 0.00911 / 0.02083 | 0.01134 / 0.02778 |
| loop_alias_stress | 0.00008 / 0.01042 | 0.00864 / 0.01563 | 0.01533 / 0.03819 | 0.01893 / 0.04688 |
| medium_enclosed_maze | 0.00073 / 0.01215 | 0.01015 / 0.04514 | 0.01318 / 0.05729 | 0.01560 / 0.05729 |
| open_obstacle_field | 0.00384 / -0.00781 | -0.00095 / -0.01823 | -0.00496 / -0.04948 | -0.00920 / -0.04427 |
| rough_local_dynamics | 0.00350 / -0.00521 | 0.00561 / 0.01042 | 0.00840 / 0.03125 | 0.00829 / 0.04948 |
| small_enclosed_maze | 0.00403 / 0.04427 | 0.01516 / 0.03125 | 0.02365 / 0.03125 | 0.02732 / 0.02344 |
| visual_sensor_stress | 0.00013 / -0.03125 | -0.00345 / 0.00000 | 0.01326 / 0.08854 | 0.02452 / 0.02604 |

`open_obstacle_field` is the principal negative late-horizon family and was
retained without exclusion. The exact per-family values for all metrics and
all four cells are retained under `paired_seed_analysis.per_family` in the
bound result; `local_composite_motifs` remains diagnostic-only in downstream
interpretation.

### Seed-level rollout effects

The table records equal-family `M` for changed cosine and top-1 as
`cosine / top-1` for each training seed.

| Seed | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| 2026080901 | 0.00309 / -0.00868 | 0.00586 / 0.00694 | 0.01300 / 0.05903 | 0.01534 / 0.05295 |
| 2026080902 | 0.00345 / -0.00694 | 0.00711 / 0.02344 | 0.01305 / 0.03212 | 0.01612 / 0.01128 |
| 2026080903 | 0.00333 / -0.00347 | 0.00892 / 0.04340 | 0.01434 / 0.03993 | 0.01561 / -0.00955 |
| 2026080904 | 0.00187 / 0.02865 | 0.00677 / 0.00781 | 0.01341 / 0.05122 | 0.01678 / 0.02344 |
| 2026080905 | 0.00140 / 0.00347 | 0.00045 / 0.01042 | 0.00573 / 0.01215 | 0.00986 / 0.01302 |
| 2026080906 | 0.00027 / 0.01563 | 0.00815 / 0.01823 | 0.01358 / 0.00174 | 0.01451 / 0.02691 |
| 2026080907 | 0.00068 / -0.02083 | 0.00447 / 0.00434 | 0.01271 / 0.02691 | 0.01554 / 0.01910 |
| 2026080908 | 0.00417 / 0.00955 | 0.00355 / 0.00260 | 0.00742 / 0.02431 | 0.00974 / 0.03038 |

The exact seed-level B_RGB, B_prop, M and J arrays for every reported metric
are retained in the bound result.

## Stage D: frozen occupancy-probe spatial retention

The independently frozen probe-package digest is
`b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686`;
the occupied-IoU qualification floor is `0.35`. The minimal labels are already
materialised: 20 states, 240 branches and 960 horizon labels, with label
contract `955359baf22407975b87e68923a5376050e3859956c5cda34dbe2abc841fc5bf`,
label-index digest
`a81f1c63f9fa181bfa728b1cb5da2ad4573f2aa80cb5801c9d54acab34d411e2`,
label-corpus digest
`a402ee134a0ec854b9936699e42e0a2c715ea70ac99a2c0393ee09ba6ac41a27`,
and completion receipt
`97d2ba6f65ca217c6aac038bdd3aeed627cfd76ed0a09c5cf95bf3e6939c8dc8`.

The Stage-D assay-spec digest is
`336c796d6256934492edf67650ddd0b71f3c661a5c9610b89ad8abff9c51fca1`;
the frozen true-target gate digest is
`4bf9a92144fa728d953c9dffebb235c9b476ded59d7462a107fe2e6ade0894e4`;
and the completed occupancy report digest is
`09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6`.
The exact result is
[`occupancy_results/result.json`](../.generated/go2_counterfactual_fidelity_v1_2/occupancy_results/result.json).

### True-target qualification gate

The gate uses whole-pilot pooled observable occupied IoU exactly as frozen.

| H | Whole-pilot pooled IoU | Equal-family IoU | Corpus-weighted IoU | Verdict |
|---|---:|---:|---:|---|
| H1 | 0.348499 | unavailable | 0.432096 | UNAVAILABLE: below 0.35 |
| H2 | 0.353808 | 0.387905 | 0.433730 | QUALIFIED |
| H3 | 0.365929 | 0.379702 | 0.425132 | QUALIFIED |
| H4 | 0.365202 | 0.372028 | 0.421689 | QUALIFIED |

At H1, `rough_local_dynamics` had no defined occupied-IoU row. The required
eight-family estimate is therefore unavailable; no seven-family average or
imputation was used. More importantly, the frozen whole-pilot gate itself was
`0.348499 < 0.35`. H1 predictor occupancy was consequently not evaluated and
cannot be interpreted as predictor degradation. H2-H4 passed and only those
horizons were opened to predicted-latent evaluation.

### Qualified-horizon predictor results

Each predicted cell below is `occupied IoU / (true target - predicted) gap`.
These are equal-family means over the eight seed quadruplets.

| H | True target | RGB one-step | RGB rollout | Prop one-step | Prop rollout |
|---|---:|---:|---:|---:|---:|
| H2 | 0.387905 | 0.223015 / 0.164890 | 0.229111 / 0.158794 | 0.224011 / 0.163894 | 0.229116 / 0.158788 |
| H3 | 0.379702 | 0.172874 / 0.206828 | 0.178735 / 0.200968 | 0.166055 / 0.213648 | 0.175791 / 0.203912 |
| H4 | 0.372028 | 0.161622 / 0.210406 | 0.167423 / 0.204605 | 0.158702 / 0.213326 | 0.164713 / 0.207314 |

Paired rollout effects are means with two-sided 95% t-intervals. Positive
values mean rollout improves occupied IoU.

| H | B_RGB | B_prop | M | J | Corpus-weighted M |
|---|---:|---:|---:|---:|---:|
| H2 | 0.006096 [-0.002378, 0.014570] | 0.005105 [-0.002972, 0.013183] | 0.005601 [-0.001782, 0.012983] | -0.000991 [-0.008479, 0.006498] | 0.005004 [-0.004707, 0.014715] |
| H3 | 0.005860 [-0.004445, 0.016166] | 0.009736 [-0.001031, 0.020503] | 0.007798 [-0.000962, 0.016558] | 0.003876 [-0.007841, 0.015592] | 0.011171 [0.001020, 0.021323] |
| H4 | 0.005801 [-0.004402, 0.016004] | 0.006011 [-0.001323, 0.013346] | 0.005906 [-0.002043, 0.013855] | 0.000210 [-0.007732, 0.008153] | 0.008077 [0.000288, 0.015865] |

The primary equal-family rollout intervals include zero at H2-H4. The
secondary corpus-weighted effect is above zero at H3-H4. All spatial
proprioception-by-rollout interaction intervals include zero. Thus the frozen
probe retains enough target-latent signal to evaluate H2-H4 and yields a
positive but primary-inconclusive rollout tendency; this remains a spatial
retention diagnostic, not a utility or planning result.

Per-family values are `true-target occupied IoU / rollout M`:

| Family | H2 | H3 | H4 |
|---|---:|---:|---:|
| large_enclosed_maze | 0.48306 / 0.00336 | 0.52378 / 0.01118 | 0.47860 / 0.00766 |
| local_composite_motifs | 0.28265 / 0.01226 | 0.32143 / 0.00605 | 0.22564 / -0.00609 |
| loop_alias_stress | 0.32563 / 0.01323 | 0.33796 / 0.01432 | 0.36077 / 0.00951 |
| medium_enclosed_maze | 0.40562 / -0.01538 | 0.41867 / 0.00808 | 0.43920 / 0.00408 |
| open_obstacle_field | 0.11701 / -0.01178 | 0.14847 / -0.00874 | 0.12552 / -0.00692 |
| rough_local_dynamics | 0.10000 / 0.00000 | 0.10417 / 0.00000 | 0.08172 / 0.00000 |
| small_enclosed_maze | 0.62583 / 0.04738 | 0.49487 / 0.02434 | 0.56132 / 0.03632 |
| visual_sensor_stress | 0.76343 / -0.00426 | 0.68828 / 0.00715 | 0.70345 / 0.00269 |

No probe weights were changed or refit. Stage D opened no predictor checkpoint;
it consumed the already bound Stage-B/C predicted-latent shards. Probe state
digest `588295858ab326f31084e542bd1d86c23b5d08defe41567533e3b12bd10c84ac`
matched the frozen 100,785,421-byte weights file with SHA-256
`95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322`.

## Runtime, storage and recovery provenance

| Stage | Runtime | Bound storage |
|---|---:|---:|
| Stage A branch completion invocation | 242.421 s (76.980 s in completed branch rows) | 18,548,622 rendered/branch bytes plus 8,731,056 bound JSON/ledger bytes |
| Stage A target encoding | 216.282 s initial encoding; 0.024 s final no-op resume verification | 1,604,321,280 latent bytes |
| Checkpoint lineage/hash verification | 6.533 s | 6,611,613,664 read-only checkpoint bytes |
| Stage B/C predictor scoring | 1,522.578 s | 48,356,388,586 prediction-ledger/shard/index/receipt bytes |
| Stage B/C analysis | 0.064 s | 15,199,537-byte result JSON |
| Stage B/C final invocation total | 1,535.050 s | target encoder checkpoint was 5,151,198,524 read-only bytes |
| Stage D final probe invocation | 24.584 s total (0.171 s true-target/gate; 0.139 s predicted-latent application) | 3,932,160 label-shard bytes; 48,318,382,080 read-only source prediction-shard bytes |

Recovery was idempotent and preserved invalid/interrupted work:

- The interrupted 45-state pre-outcome scorer-fit identity set was preserved as invalid and never reused. No 120-state scorer corpus or scorer training was started.
- Two pre-freeze Stage-A implementation attempts were preserved separately: one omitted the assay gate binding; the next narrowed the encoder smoke index. Neither was mixed into this corpus.
- Two validate-only Stage B/C defects (legacy-pilot canonical JSON encoding and a list/mapping digest check) produced no prediction result before correction.
- Predictor scoring was manually interrupted after 169 durable checkpoint-state units when a symlink-parent free-space check gave a false ENOSPC signal. No scientific artefact or contract changed. The identical registered command resumed from durable checkpoint-state receipts; the exact-resume unit was one checkpoint by one state, containing all 12 candidates.
- All 640 completed predictor state shards were independently rehashed; final validation loaded zero predictors and found no unavailable changed-mask rows at any horizon.
- The first Stage-D reporting pass correctly found that H1 `rough_local_dynamics` had no defined occupied-IoU rows, but its summariser incorrectly required every family to be numerically defined. It stopped before freezing the gate or opening prediction shards. Undefined-result propagation and reporting were corrected without changing label generation, true-target record generation or probe inference. The pre-fix source SHA-256 was `366c36f766bcf064cb7e68a46c7cb7922cf6e4138394b6cb98e1656668bd71c9`; the active source is `2b1fe088105054cf691e74960c21c30332d1e33ee71e42340c5ff8afb23d50ee`; source-equivalence receipt `b5e608d78fb0ab3b8afcdc3d04f58c666956cbdf6f6e923da2b0c03ee1a6d0dc` binds the unchanged protected scientific AST.
- A final reporting-only repair made `episode_clusters` and `clusters_per_family` count all registered clusters while retaining separate defined-IoU counts. It regenerated no label, true-target or predicted-state record, changed no scientific value or gate verdict, and preserved the superseded gate/result as invalid. Recovery digest: `287d1e4a226304242c279c548a542cfd8fca54c5fd4159109d3382a6f9591fda`.

## Verified scientific checkpoints

Before the first checkpoint load, all 32 epoch-21 files and the frozen run
package (`cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991`)
were verified. The confirmatory commit was
`443e5914694a533534486b629e95ec15f8df9b7a`, and the frozen confirmatory
report digest was
`60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161`.

| Seed | Cell | Checkpoint SHA-256 | Bytes |
|---:|---|---|---:|
| 2026080901 | rgb_one_step | `20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a` | 206,534,551 |
| 2026080901 | rgb_rollout | `75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4` | 206,534,551 |
| 2026080901 | proprio_one_step | `41d1c5a48d7adacf2e2b698318782de29c7b95342181bdf5fd5578d35346f1d1` | 206,691,255 |
| 2026080901 | proprio_rollout | `75ab2a5dd5c48ebb2f33935d962d957c4e62eab3427ce0ad8108d690a1df9218` | 206,691,255 |
| 2026080902 | rgb_one_step | `085702386da4b36bafe6ff432ca955a2b1a9a69de9a8023aa4fc3b099953f0ff` | 206,534,551 |
| 2026080902 | rgb_rollout | `90bbf9a8117dbf528d9693415becd5c9e9605ecad02520f3e00513dfee691530` | 206,534,551 |
| 2026080902 | proprio_one_step | `76c9c5328217aceee64e7d4a60524d8317b459c68ca4da05a24509dfd2c94dc9` | 206,691,255 |
| 2026080902 | proprio_rollout | `030a28078acbc495a3a79e0e513501586a8271938200f194a4632ed08b49fca8` | 206,691,255 |
| 2026080903 | rgb_one_step | `a7878c6159cceae8f69f84927bd1ee3a4c3d8dbf6d1e97003eb9ebdae1f91bc4` | 206,534,551 |
| 2026080903 | rgb_rollout | `b769ef91f1ef17377f7c7f184c85ea0a9859ead2b87aa8351a89b7a05192aad1` | 206,534,551 |
| 2026080903 | proprio_one_step | `ab440561a867e1961c156ed556271ece87a51b959c3ce4e4b527a1d9136c46d2` | 206,691,319 |
| 2026080903 | proprio_rollout | `591538880c4beaf982196f36364a18fb4167d80f6671912975d4ad454f731545` | 206,691,319 |
| 2026080904 | rgb_one_step | `1386b6303ac5b47fea7a67e831a375d164ba372ee4bf60fd87609ed35352d1ff` | 206,534,551 |
| 2026080904 | rgb_rollout | `aad6711b6d15e6664038ace1fe0f376516256062c2235334b74bfb68135e419a` | 206,534,551 |
| 2026080904 | proprio_one_step | `bdfed483f5a173eec40a8b9d6c586b478a2a4929f7291b13c087d98eb336eee0` | 206,691,255 |
| 2026080904 | proprio_rollout | `bdc2c7d5f09472e3fcf7813ff316c8f6bef021a0ee1f2900178bbb3b52b8e0e0` | 206,691,255 |
| 2026080905 | rgb_one_step | `5d78f18e0d0052479cb81a43acbaa953bebeb6fc13dac58c506211c46416a1e9` | 206,534,551 |
| 2026080905 | rgb_rollout | `c474a5b09c041aa263950b3b2b8bd2369d3644aec7019268610fea4b846b6386` | 206,534,551 |
| 2026080905 | proprio_one_step | `8f53f8991ffbf1994a9b6ff74087c8c23c908c8ad49bdd892ac2b65394501cb8` | 206,691,319 |
| 2026080905 | proprio_rollout | `3a615f5d00dc106d24c3719489ba04e52a8bb4f97e49a385f1d8f4908d24aad4` | 206,691,319 |
| 2026080906 | rgb_one_step | `846fbe05f78e9b513841cb08f71858e9fdb7dd4430181bca140d29f72574a200` | 206,534,551 |
| 2026080906 | rgb_rollout | `fc480799cc637f5c3d4bd582da233e38b76d422b48833075e018d49df517aa1a` | 206,534,551 |
| 2026080906 | proprio_one_step | `464fd320561b1e92770231e849df53ef3ee5ab08f85e54c40218830863beb309` | 206,691,255 |
| 2026080906 | proprio_rollout | `daa77c7ed9600fcd17b37fd1cd3a73c1ad0902f439e111502436ca247030f6cb` | 206,691,255 |
| 2026080907 | rgb_one_step | `86d9f6108f40b8d2cf49e5264fc998412493258258b86391067c71193066afbc` | 206,534,615 |
| 2026080907 | rgb_rollout | `4501841125eee43568e6031d4061d23b309c080f11b129538dadb6cfc8a05432` | 206,534,615 |
| 2026080907 | proprio_one_step | `673f4c4251706ef49f16fdb9d1e48e391cdf871f34965e779fda5609aa78aff8` | 206,691,319 |
| 2026080907 | proprio_rollout | `4c09a551ab89f55260cbbd24937ea81d5cb081bd319a8f61118e7d2ddd488f89` | 206,691,319 |
| 2026080908 | rgb_one_step | `025aff4d9bc7380b4a51e4ac08282bbeafb2be189bf27d31d48bcf247f2b02f2` | 206,534,615 |
| 2026080908 | rgb_rollout | `a39f5050c02ab7b002c6b1c76256dc2b5783046cf5b877cc6d5354880c45b89a` | 206,534,615 |
| 2026080908 | proprio_one_step | `aa4cef094a3d503ea1062a59a4d36b9f4b66eb03a6f958b32ef8e3d54f5ab94a` | 206,691,319 |
| 2026080908 | proprio_rollout | `6027d657ce81d8ae968354031e666c9a608d800e25663629944590f448488b4f` | 206,691,319 |

# Decision-ready conclusion

The frozen assay supports three narrow conclusions:

1. Rollout training improves direct counterfactual future fidelity at H=1–4 under the prespecified eight-seed, equal-family estimator. The main rollout intervals are strictly favorable at every horizon for both changed-token cosine and normalized-error reduction.

2. Rollout training improves some action-specific branch-retrieval metrics, most clearly at H=3–4 and for top-1 already at H=2. H=1 top-1 is null/imprecise. Retrieval evidence is metric-dependent and H=1–2 contain a material structural degeneracy: three exact duplicate target-branch pairs occur in every state.

3. The frozen occupancy probe qualifies on true targets only at H=2–4. Predicted latents retain partial occupancy information, but all primary equal-family rollout-effect intervals include zero. Thus improved occupancy retention is a positive tendency, not an established treatment effect.

The assay does not test planning utility, selected-candidate reward, oracle-utility ranking, or a supervised scorer. No utility scorer was trained or invoked.

All treatment intervals below use the eight paired training seeds, df=7. Seed order is always `2026080901 … 2026080908`.

Notation:

- `R1`: RGB one-step
- `RR`: RGB rollout
- `P1`: proprio one-step
- `PR`: proprio rollout
- `B_R = RR − R1`
- `B_P = PR − P1`
- `M = (B_R + B_P)/2`
- `J = B_P − B_R`

For lower-is-better metrics, the report sign-reverses the contrast so positive means rollout benefit.

## 1. Dataset and evaluation contract

### Corpus

| Quantity | Frozen value |
|---|---:|
| States | 20 |
| Unique scenes | 20 |
| Episode/state clusters | 20 |
| Attempted branches | 240 |
| Valid branches | 240 |
| Invalid branches | 0 |
| Oracle-equal replay outcomes | 240 |
| Candidates per state | 12 |
| States with all candidate indices 0–11 | 20/20 |
| Truncated branches | 0 |
| H1 valid targets | 240/240 |
| H2 valid targets | 240/240 |
| H3 valid targets | 240/240 |
| H4 valid targets | 240/240 |
| Predictor comparisons per horizon | 7,680/7,680 |
| Missing predictor/branch/horizon rows | 0 |

Family allocation:

| Family | States | Branches |
|---|---:|---:|
| large_enclosed_maze | 3 | 36 |
| local_composite_motifs | 3 | 36 |
| loop_alias_stress | 3 | 36 |
| medium_enclosed_maze | 3 | 36 |
| open_obstacle_field | 2 | 24 |
| rough_local_dynamics | 2 | 24 |
| small_enclosed_maze | 2 | 24 |
| visual_sensor_stress | 2 | 24 |

There is one state per scene and episode cluster.

### Aggregation and replication

The direct-fidelity equal-family estimator is:

1. Compute each candidate-row metric.
2. Average the 12 candidate rows within the state/episode cluster.
3. Average state clusters within each family.
4. Take the unweighted mean of the eight family means.

The direct-fidelity corpus-weighted estimator instead pools changed-token contributions across all 240 branches:

- cosine pools changed-token similarities;
- normalized error pools prediction SSE and persistence SSE before taking their ratio.

Consequently it weights families by realized branch/token mass, rather than equally.

The 20 state clusters are the environmental observations beneath each model result, but they are not the replication units for the reported uncertainty. No state-, scene-, or environment-resampling interval was registered. All treatment t-intervals hold the 20 states fixed and use the eight training-seed quadruplets as the independent model-treatment replications.

### Rendering and encoder binding

| Binding | Digest |
|---|---|
| textured-v03 renderer wrapper | `df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b` |
| rendering contract | `2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17` |
| center-crop/preprocessing contract | `2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9` |
| preprocessing identity | `8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5` |
| target-encoder binding | `15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5` |
| target-encoder checkpoint | `7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6` |
| predictor run package | `cf0456bef0cbe7cd8f2cd666b600f91ebf845f6156d180569edf36be53552991` |
| predictor normalization | `f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab` |
| B/C assay specification | `a26fa0ec9ee9e0df3bbe71fff6d7594bb714227aaa66a66631836d94a676feab` |

The visual path was the frozen v03 contract: 224×224 rendered frame, crop rows 28:196, resize to 512×384, ImageNet preprocessing, 24×32 token grid, 768 tokens of dimension 1024.

### Action and history reconstruction

At each frozen snapshot:

- The actual previous applied 3-D command was recovered from the redriven controller.
- The deterministic slew limiter was applied tick by tick, carrying that previous command through four candidate blocks.
- Each block comprised five 10 Hz ticks.
- Frozen per-tick limits were 0.25 for vx/vy and 0.35 for yaw rate.
- Requested and post-slew plans were stored as `[4,5,3]`.
- Physical execution was checked against the registered post-slew plan within `1e-6`.
- Predictor action input used the same post-slew plan, flattened block-major/tick-major over active `vx,yaw` channels to `[4,10]`.

Observed control history was 15 previous applied `vx,yaw` samples, arranged as `[3,5,2]`, and was identical across candidates and all four cells. During autoregression, the sliding control window received deterministic efference copy from the candidate plan—not measured future control.

Proprioceptive cells received only 15 observed 30-D samples, `[3,5,30]`. Future proprio slots used the frozen invalid mask and absence token.

The validated predictor-input allow-list excluded:

- future RGB;
- true target latents;
- oracle components or utility;
- branch outcomes;
- future proprioception;
- privileged future simulator state.

Target handles were accessed only after prediction for scoring.

## 2. Direct future fidelity

Changed cosine uses the frozen target-specific changed-token mask. Normalized error is predictor MSE divided by persistence MSE on that mask. Persistence is the final observed context latent.

For cosine, positive `B` is rollout-minus-one-step. For normalized error, the report uses `B = one-step error − rollout error`, so positive means improvement. The corresponding raw rollout-minus-one-step error interval is therefore below zero.

### Cell levels and persistence

| H | Persistence cosine | R1 cosine | RR cosine | P1 cosine | PR cosine | R1 advantage | RR advantage | P1 advantage | PR advantage | R1 error | RR error | P1 error | PR error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | .494265 | .735481 | .738227 | .736101 | .737921 | .241216 | .243963 | .241837 | .243657 | .520778 | .515413 | .519577 | .516018 |
| H2 | .443658 | .710005 | .715872 | .709732 | .715184 | .266347 | .272213 | .266074 | .271526 | .519603 | .509217 | .520194 | .510536 |
| H3 | .436567 | .688447 | .700001 | .688085 | .699841 | .251879 | .263434 | .251518 | .263274 | .551169 | .530915 | .551830 | .531226 |
| H4 | .431295 | .673646 | .687894 | .673637 | .687762 | .242351 | .256599 | .242342 | .256467 | .571136 | .546932 | .571232 | .547263 |

Normalized-error persistence is 1.0 by definition.

### Primary equal-family contrasts

Each entry is `mean; sample SD; [two-sided 95% t-interval]`.

| Metric | H | B_R | B_P | M | J |
|---|---|---|---|---|---|
| cosine | H1 | .002747; .001222; `[.001726,.003768]` | .001820; .002177; `[-.000000470,.003640]` | .002283; .001427; `[.001091,.003476]` | -.000927; .002079; `[-.002665,.000811]` |
| cosine | H2 | .005866; .003173; `[.003214,.008519]` | .005452; .002974; `[.002966,.007939]` | .005659; .002757; `[.003354,.007964]` | -.000414; .002724; `[-.002691,.001864]` |
| cosine | H3 | .011555; .003541; `[.008595,.014515]` | .011756; .003414; `[.008902,.014610]` | .011655; .003204; `[.008977,.014334]` | .000201; .002707; `[-.002062,.002465]` |
| cosine | H4 | .014248; .003219; `[.011557,.016939]` | .014124; .002971; `[.011641,.016608]` | .014186; .002784; `[.011859,.016513]` | -.000123; .002717; `[-.002395,.002148]` |
| error reduction | H1 | .005365; .002294; `[.003447,.007283]` | .003559; .004339; `[-.000068678,.007186]` | .004462; .002752; `[.002161,.006763]` | -.001806; .004230; `[-.005343,.001730]` |
| error reduction | H2 | .010385; .005559; `[.005738,.015033]` | .009658; .005318; `[.005212,.014104]` | .010022; .004890; `[.005934,.014109]` | -.000727; .004767; `[-.004713,.003258]` |
| error reduction | H3 | .020254; .006280; `[.015003,.025505]` | .020605; .006050; `[.015546,.025663]` | .020429; .005712; `[.015654,.025205]` | .000350; .004644; `[-.003532,.004233]` |
| error reduction | H4 | .024204; .005658; `[.019474,.028935]` | .023969; .005128; `[.019682,.028256]` | .024087; .004859; `[.020025,.028149]` | -.000235; .004710; `[-.004173,.003703]` |

The main `M` interval is strictly favorable at all four horizons for both metrics. In raw rollout-minus-one-step normalized-error units, the exact main intervals are:

- H1 `[-.006762623,-.002161400]`
- H2 `[-.014109498,-.005933591]`
- H3 `[-.025205023,-.015653543]`
- H4 `[-.028148909,-.020024933]`

The H1 proprio-only intervals are not strictly favorable: the exact cosine lower bound is `−4.6986317426417346e−7`, and the error-reduction lower bound is `−0.00006867848035240496`.

### Primary seed vectors

Cosine:

```text
H1 BR=[.001952,.005037,.003567,.002472,.002304,.000951,.002410,.003282]
   BP=[.004226,.001867,.003092,.001274,.000503,-.000414,-.001050,.005063]
   M =[.003089,.003452,.003329,.001873,.001404,.000268,.000680,.004172]
   J =[.002274,-.003171,-.000475,-.001198,-.001802,-.001366,-.003460,.001781]

H2 BR=[.004050,.009024,.008761,.007915,.001565,.008826,.004996,.001793]
   BP=[.007666,.005191,.009079,.005623,-.000664,.007475,.003947,.005301]
   M =[.005858,.007108,.008920,.006769,.000451,.008151,.004471,.003547]
   J =[.003616,-.003833,.000317,-.002292,-.002229,-.001351,-.001048,.003508]

H3 BR=[.010906,.015239,.013666,.013414,.006432,.014393,.012388,.006000]
   BP=[.015086,.010852,.015004,.013414,.005029,.012774,.013042,.008849]
   M =[.012996,.013045,.014335,.013414,.005731,.013583,.012715,.007425]
   J =[.004180,-.004387,.001338,.000000073,-.001403,-.001619,.000654,.002849]

H4 BR=[.013230,.018665,.016304,.016161,.010519,.014664,.015522,.008920]
   BP=[.017452,.013568,.014908,.017391,.009195,.014361,.015561,.010559]
   M =[.015341,.016116,.015606,.016776,.009857,.014513,.015541,.009740]
   J =[.004223,-.005096,-.001396,.001230,-.001323,-.000303,.000038,.001639]
```

Normalized-error reduction:

```text
H1 BR=[.003617,.009578,.006810,.004975,.004442,.001955,.005071,.006474]
   BP=[.008505,.003680,.005820,.002815,.000975,-.000873,-.002359,.009907]
   M =[.006061,.006629,.006315,.003895,.002709,.000541,.001356,.008191]
   J =[.004888,-.005898,-.000990,-.002159,-.003468,-.002827,-.007430,.003432]

H2 BR=[.007207,.015844,.015487,.013816,.002744,.015686,.008999,.003299]
   BP=[.013596,.009129,.016225,.010062,-.001113,.013354,.006639,.009371]
   M =[.010402,.012486,.015856,.011939,.000816,.014520,.007819,.006335]
   J =[.006389,-.006715,.000737,-.003754,-.003857,-.002332,-.002360,.006072]

H3 BR=[.018940,.026622,.024052,.023204,.011073,.025622,.021990,.010529]
   BP=[.026214,.019163,.026494,.023703,.008720,.022623,.022678,.015241]
   M =[.022577,.022892,.025273,.023454,.009896,.024123,.022334,.012885]
   J =[.007275,-.007459,.002442,.000499,-.002353,-.002999,.000688,.004712]

H4 BR=[.022110,.032016,.028045,.027496,.017782,.024914,.026375,.014897]
   BP=[.029466,.023558,.025079,.029922,.015309,.024359,.026176,.017887]
   M =[.025788,.027787,.026562,.028709,.016545,.024636,.026276,.016392]
   J =[.007356,-.008458,-.002966,.002426,-.002474,-.000555,-.000199,.002990]
```

### Corpus-weighted secondary estimates

| Metric | H | Cells R1/RR/P1/PR | B_R `[CI]` | B_P `[CI]` | M; SD `[CI]` | J `[CI]` |
|---|---|---|---|---|---|---|
| cosine | H1 | .699236/.702019/.699488/.702322 | .002784 `[.000972,.004595]` | .002834 `[.000314,.005354]` | .002809; .001987 `[.001148,.004470]` | .000050 `[-.002818,.002918]` |
| cosine | H2 | .682066/.691247/.680829/.690957 | .009181 `[.005964,.012398]` | .010128 `[.007249,.013006]` | .009655; .002835 `[.007284,.012025]` | .000946 `[-.002901,.004793]` |
| cosine | H3 | .652943/.670015/.652325/.670910 | .017071 `[.013477,.020666]` | .018586 `[.015184,.021988]` | .017829; .003334 `[.015041,.020616]` | .001515 `[-.002717,.005746]` |
| cosine | H4 | .634361/.656693/.634658/.657481 | .022332 `[.018688,.025976]` | .022823 `[.019446,.026200]` | .022578; .003429 `[.019711,.025444]` | .000491 `[-.003572,.004553]` |
| error reduction | H1 | .563624/.558407/.563151/.557840 | .005217 `[.001822,.008611]` | .005311 `[.000589,.010033]` | .005264; .003723 `[.002151,.008376]` | .000094 `[-.005280,.005468]` |
| error reduction | H2 | .549517/.533648/.551655/.534150 | .015869 `[.010309,.021429]` | .017505 `[.012529,.022480]` | .016687; .004900 `[.012590,.020784]` | .001635 `[-.005013,.008284]` |
| error reduction | H3 | .594890/.565628/.595951/.564093 | .029262 `[.023101,.035423]` | .031858 `[.026026,.037690]` | .030560; .005715 `[.025782,.035338]` | .002596 `[-.004657,.009849]` |
| error reduction | H4 | .623526/.585443/.623019/.584099 | .038083 `[.031869,.044298]` | .038920 `[.033161,.044680]` | .038502; .005847 `[.033613,.043390]` | .000837 `[-.006091,.007764]` |

### Per-family main effects

| Family | Cosine M H1/H2/H3/H4 | Error-reduction M H1/H2/H3/H4 |
|---|---|---|
| large_enclosed_maze | .000639/.007045/.015266/.016691 | .001331/.012911/.026863/.028785 |
| local_composite_motifs | .005308/.003089/.009108/.011342 | .008448/.004202/.015220/.019374 |
| loop_alias_stress | .000083/.008638/.015328/.018929 | .000399/.014878/.026202/.032092 |
| medium_enclosed_maze | .000729/.010148/.013180/.015595 | .001268/.017628/.022595/.026198 |
| open_obstacle_field | .003843/−.000955/−.004957/−.009198 | .007919/−.001131/−.008216/−.015455 |
| rough_local_dynamics | .003501/.005609/.008405/.008287 | .007639/.010643/.015953/.015695 |
| small_enclosed_maze | .004033/.015155/.023651/.027322 | .008328/.026777/.041874/.047400 |
| visual_sensor_stress | .000130/−.003454/.013262/.024521 | .000363/−.005737/.022943/.038605 |

The aggregate effect is heterogeneous. `open_obstacle_field` is negative at H2–4 under both direct metrics.

### Effect magnitude against explicit references

| H | Cosine M | % of one-step cosine | % of persistence gap | M/seed SD | Error-reduction M | % of one-step error | % of persistence gap | M/seed SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | .002283 | 0.310% | 0.945% | 1.60 | .004462 | 0.858% | 0.930% | 1.62 |
| H2 | .005659 | 0.797% | 2.126% | 2.05 | .010022 | 1.928% | 2.087% | 2.05 |
| H3 | .011655 | 1.693% | 4.631% | 3.64 | .020429 | 3.704% | 4.555% | 3.58 |
| H4 | .014186 | 2.106% | 5.854% | 5.10 | .024087 | 4.217% | 5.617% | 4.96 |

These comparisons establish statistical stability under the seed estimator, but the absolute gains are small at H1–2.

## 3. Action-specificity construction

For each state, predictor checkpoint, candidate `i`, and horizon `H`:

- Query: that candidate’s predicted latent at the single horizon, shape `[768,1024]`.
- Gallery: the 12 realized target latents at that same horizon, exclusively from the same starting state.
- Target normalization: raw f16 was reloaded as float32, layer-normalized across the 1024-D feature axis, then each token was L2-normalized.
- Prediction normalization: frozen predictor normalization was applied at every autoregressive step, followed by token L2 normalization for retrieval.
- Similarity: aligned-token cosine, averaged across all 768 tokens.
- Retrieval did not use the changed-token mask.
- Candidates were ranked separately at H1, H2, H3, and H4. It was not cumulative H1…H trajectory retrieval.
- The matching branch identity was present exactly once.
- Distinct branch identities with identical target contents remained distinct gallery items.
- Near-identical outcomes were not merged or specially treated.
- Ranking used descending raw cosine, then lower frozen target index.
- The reported “tie” flag used `abs(own − wrong) ≤ 1e−12`; ranking itself used the raw numeric order.
- Pairwise accuracy used strict `own > wrong`, so a tolerance tie counted as failure.

Thus the task is a within-state, twelve-realized-branch, horizon-specific cosine-retrieval assay.

## 4. Complete action-specificity results

For mean and median rank, the contrast is reported as rank reduction, so positive favors rollout.

Each contrast below is `mean; SD; [95% CI]`.

### Primary equal-family results

| H | Metric | Cells R1/RR/P1/PR | B_R | B_P | M | J |
|---|---|---|---|---|---|---|
| H1 | top-1 | .2663/.2661/.2693/.2739 | −.0002;.0231`[-.0196,.0191]` | .0046;.0129`[-.0062,.0153]` | .0022;.0156`[-.0109,.0152]` | .0048;.0208`[-.0126,.0221]` |
| H1 | top-3 | .6094/.6220/.6152/.6155 | .0126;.0173`[-.0019,.0271]` | .0002;.0209`[-.0173,.0177]` | .0064;.0179`[-.0085,.0213]` | −.0124;.0141`[-.0241,−.0006]` |
| H1 | MRR | .4799/.4836/.4838/.4869 | .0037;.0169`[-.0104,.0178]` | .0032;.0107`[-.0057,.0121]` | .0034;.0122`[-.0068,.0137]` | −.0005;.0141`[-.0123,.0113]` |
| H1 | mean-rank reduction | 3.701/3.656/3.675/3.665 | .0458;.1270`[-.0604,.1520]` | .0100;.0899`[-.0652,.0852]` | .0279;.0996`[-.0554,.1112]` | −.0358;.0935`[-.1140,.0424]` |
| H1 | median-rank reduction | 2.938/2.844/2.922/2.867 | .0938;.1417`[-.0247,.2122]` | .0547;.1782`[-.0943,.2036]` | .0742;.1180`[-.0245,.1729]` | −.0391;.2189`[-.2221,.1440]` |
| H1 | own−best-other | −.0308/−.0286/−.0306/−.0287 | .0022;.0007`[.0016,.0028]` | .0019;.0015`[.0007,.0031]` | .0020;.0006`[.0015,.0026]` | −.0003;.0019`[-.0019,.0013]` |
| H1 | own−mean-other | .0380/.0367/.0384/.0369 | −.0012;.0014`[-.0024,−.0001]` | −.0015;.0007`[-.0020,−.0009]` | −.0013;.0010`[-.0022,−.0005]` | −.0002;.0009`[-.0009,.0005]` |
| H1 | pairwise | .7317/.7359/.7341/.7350 | .0042;.0115`[-.0055,.0138]` | .0009;.0082`[-.0059,.0077]` | .0025;.0091`[-.0050,.0101]` | −.0033;.0085`[-.0104,.0039]` |
| H2 | top-1 | .2676/.2826/.2648/.2791 | .0150;.0199`[-.0016,.0316]` | .0143;.0134`[.0031,.0255]` | .0146;.0136`[.0033,.0260]` | −.0007;.0202`[-.0175,.0162]` |
| H2 | top-3 | .6194/.6183/.6263/.6198 | −.0011;.0315`[-.0275,.0253]` | −.0065;.0219`[-.0248,.0118]` | −.0038;.0257`[-.0253,.0177]` | −.0054;.0175`[-.0201,.0092]` |
| H2 | MRR | .4840/.4935/.4838/.4916 | .0095;.0154`[-.0034,.0224]` | .0078;.0118`[-.0020,.0177]` | .0087;.0114`[-.0009,.0182]` | −.0017;.0152`[-.0144,.0111]` |
| H2 | mean-rank reduction | 3.668/3.629/3.666/3.628 | .0393;.1251`[-.0653,.1439]` | .0373;.0962`[-.0431,.1178]` | .0383;.1004`[-.0457,.1223]` | −.0020;.0974`[-.0834,.0795]` |
| H2 | median-rank reduction | 2.805/2.820/2.805/2.828 | −.0156;.1485`[-.1397,.1085]` | −.0234;.2003`[-.1909,.1440]` | −.0195;.1484`[-.1436,.1045]` | −.0078;.1903`[-.1669,.1513]` |
| H2 | own−best-other | −.0342/−.0303/−.0343/−.0309 | .0040;.0021`[.0022,.0058]` | .0034;.0025`[.0014,.0055]` | .0037;.0020`[.0020,.0054]` | −.0005;.0023`[-.0024,.0014]` |
| H2 | own−mean-other | .0557/.0529/.0552/.0529 | −.0029;.0028`[-.0052,−.0005]` | −.0023;.0015`[-.0036,−.0011]` | −.0026;.0020`[-.0043,−.0009]` | .0006;.0020`[-.0011,.0022]` |
| H2 | pairwise | .7347/.7383/.7349/.7383 | .0036;.0114`[-.0059,.0131]` | .0034;.0087`[-.0039,.0107]` | .0035;.0091`[-.0042,.0111]` | −.0002;.0089`[-.0076,.0072]` |
| H3 | top-1 | .2622/.2882/.2487/.2845 | .0260;.0300`[.0009,.0512]` | .0358;.0321`[.0089,.0627]` | .0309;.0191`[.0150,.0469]` | .0098;.0491`[-.0313,.0509]` |
| H3 | top-3 | .5933/.6003/.5820/.6066 | .0069;.0255`[-.0144,.0282]` | .0245;.0196`[.0082,.0409]` | .0157;.0164`[.0020,.0295]` | .0176;.0314`[-.0087,.0438]` |
| H3 | MRR | .4719/.4869/.4624/.4861 | .0150;.0240`[-.0051,.0351]` | .0237;.0192`[.0077,.0398]` | .0194;.0136`[.0080,.0308]` | .0087;.0339`[-.0196,.0370]` |
| H3 | mean-rank reduction | 3.863/3.781/3.919/3.789 | .0825;.1589`[-.0504,.2153]` | .1300;.1116`[.0367,.2233]` | .1062;.1092`[.0149,.1976]` | .0475;.1664`[-.0916,.1867]` |
| H3 | median-rank reduction | 2.898/2.891/2.984/2.898 | .0078;.3070`[-.2488,.2645]` | .0859;.1701`[-.0563,.2282]` | .0469;.2080`[-.1270,.2207]` | .0781;.2709`[-.1483,.3046]` |
| H3 | own−best-other | −.0490/−.0426/−.0496/−.0427 | .0064;.0024`[.0044,.0084]` | .0069;.0031`[.0043,.0095]` | .0066;.0022`[.0048,.0085]` | .0005;.0033`[-.0023,.0033]` |
| H3 | own−mean-other | .0508/.0488/.0499/.0489 | −.0020;.0028`[-.0044,.0004]` | −.0010;.0020`[-.0027,.0007]` | −.0015;.0020`[-.0032,.0002]` | .0010;.0029`[-.0014,.0034]` |
| H3 | pairwise | .7397/.7472/.7346/.7464 | .0075;.0144`[-.0046,.0196]` | .0118;.0101`[.0033,.0203]` | .0097;.0099`[.0014,.0180]` | .0043;.0151`[-.0083,.0170]` |
| H4 | top-1 | .2127/.2268/.2099/.2376 | .0141;.0171`[-.0002,.0284]` | .0278;.0279`[.0045,.0511]` | .0209;.0179`[.0060,.0359]` | .0137;.0294`[-.0109,.0382]` |
| H4 | top-3 | .5278/.5415/.5271/.5451 | .0137;.0308`[-.0121,.0394]` | .0180;.0330`[-.0096,.0456]` | .0158;.0254`[-.0054,.0370]` | .0043;.0387`[-.0280,.0367]` |
| H4 | MRR | .4218/.4366/.4196/.4426 | .0147;.0153`[.0019,.0276]` | .0230;.0215`[.0050,.0410]` | .0189;.0138`[.0073,.0304]` | .0083;.0252`[-.0128,.0293]` |
| H4 | mean-rank reduction | 4.363/4.225/4.381/4.206 | .1387;.1508`[.0126,.2648]` | .1745;.1664`[.0354,.3136]` | .1566;.1238`[.0530,.2601]` | .0358;.1988`[-.1304,.2020]` |
| H4 | median-rank reduction | 3.352/3.281/3.406/3.297 | .0703;.2784`[-.1624,.3030]` | .1094;.1941`[-.0529,.2716]` | .0898;.1795`[-.0602,.2399]` | .0391;.3186`[-.2273,.3054]` |
| H4 | own−best-other | −.0609/−.0497/−.0617/−.0498 | .0111;.0027`[.0089,.0134]` | .0118;.0040`[.0085,.0151]` | .0115;.0031`[.0089,.0141]` | .0007;.0026`[-.0015,.0029]` |
| H4 | own−mean-other | .0444/.0440/.0439/.0439 | −.0003;.0028`[-.0026,.0020]` | .0000;.0020`[-.0017,.0018]` | −.0001;.0020`[-.0018,.0015]` | .0004;.0027`[-.0019,.0026]` |
| H4 | pairwise | .6942/.7068/.6926/.7085 | .0126;.0137`[.0011,.0241]` | .0159;.0151`[.0032,.0285]` | .0142;.0113`[.0048,.0236]` | .0033;.0181`[-.0119,.0184]` |

Tolerance tie rates were exactly `.0454545` in all four cells at H1 and H2 and zero at H3 and H4; all treatment contrasts were zero.

### Seed vectors for the primary main effect M

The complete `B_R`, `B_P`, `M`, and `J` vectors are frozen in [result.json](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/predictor_assay/result.json>) under `paired_seed_analysis.equal_family`. The main `M` vectors are:

| Metric | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| top-1 | `[-.008681,-.006944,-.003472,.028646,.003472,.015625,-.020833,.009549]` | `[.006944,.023438,.043403,.007813,.010417,.018229,.004340,.002604]` | `[.059028,.032118,.039931,.051215,.012153,.001736,.026910,.024306]` | `[.052951,.011285,-.009549,.023437,.013021,.026910,.019097,.030382]` |
| top-3 | `[.004340,.011285,.013021,.004340,-.005208,.010417,-.025174,.038194]` | `[-.020833,-.015625,.026910,.006076,-.047743,.013889,-.016493,.023438]` | `[.040799,.015625,.033854,.015625,-.013021,.013021,.013889,.006076]` | `[.066840,.006944,.039063,.015625,-.000868,.003472,.007812,-.012153]` |
| MRR | `[-.003648,-.000292,.003819,.018241,.001168,.012044,-.019652,.015843]` | `[-.000246,.010678,.032762,.005713,-.002178,.014759,-.000501,.008309]` | `[.033826,.021558,.031528,.033452,.001909,-.000289,.020490,.012411]` | `[.051761,.014669,.007593,.018015,.010649,.013077,.017549,.017703]` |
| mean-rank reduction | `[-.029514,.057292,.083333,.048611,.023438,.067708,-.182292,.154514]` | `[.003472,.027778,.205729,.064236,-.103299,.096354,-.080729,.092882]` | `[.199653,.109375,.276042,.162326,-.030382,-.020833,.132812,.020833]` | `[.417535,.163194,.236111,.056424,.104167,.071181,.160590,.043403]` |
| median-rank reduction | `[-.031250,0,.093750,.156250,0,.156250,-.062500,.281250]` | `[0,-.156250,.156250,-.031250,-.250000,.031250,-.093750,.187500]` | `[.187500,-.093750,.343750,.187500,-.187500,-.156250,.218750,-.125000]` | `[.406250,.062500,.281250,-.156250,-.031250,.031250,.125000,0]` |
| own−best-other | `[.001983,.001554,.002179,.001857,.001342,.002103,.001927,.003426]` | `[.003079,.003196,.005467,.004790,.000444,.006903,.002042,.003673]` | `[.007071,.004476,.007732,.009266,.005150,.009373,.006894,.003077]` | `[.013527,.007755,.010831,.014587,.007363,.015075,.013621,.009136]` |
| own−mean-other | `[-.002833,.000047,-.000918,-.001420,-.000351,-.001199,-.002636,-.001476]` | `[-.004166,-.001513,.000518,-.002239,-.003169,-.000489,-.004974,-.004738]` | `[-.000451,-.000524,.000567,-.001946,-.001922,.000448,-.002729,-.005455]` | `[.001495,.001498,.000292,-.001903,-.000410,.002495,-.000983,-.003584]` |
| pairwise | `[-.002683,.005208,.007576,.004419,.002131,.006155,-.016572,.014047]` | `[.000316,.002525,.018703,.005840,-.009391,.008759,-.007339,.008444]` | `[.018150,.009943,.025095,.014757,-.002762,-.001894,.012074,.001894]` | `[.037958,.014836,.021465,.005129,.009470,.006471,.014599,.003946]` |

The H1 top-1 result is therefore not a positive finding: `M=.002170`, SD `.015584`, interval `[-.010858,.015198]`. RGB was slightly negative and proprio slightly positive, both imprecise. H2 top-1 is the first strictly positive main interval: `[.003290,.026007]`.

### Corpus-weighted secondary main effects

| Metric | H1 M; SD `[CI]` | H2 | H3 | H4 |
|---|---|---|---|---|
| top-1 | .002604;.015580 `[-.010421,.015630]` | .016406;.012272 `[.006147,.026666]` | .032031;.015627 `[.018966,.045096]` | .022396;.017314 `[.007921,.036871]` |
| top-3 | .006771;.014763 `[-.005571,.019113]` | −.003125;.025198 `[-.024191,.017941]` | .018750;.013502 `[.007462,.030038]` | .017969;.024037 `[-.002127,.038065]` |
| MRR | .003927;.011594 `[-.005765,.013620]` | .010045;.010338 `[.001402,.018688]` | .020850;.010381 `[.012172,.029529]` | .019850;.011932 `[.009875,.029825]` |
| mean-rank reduction | .029167;.088529 `[-.044845,.103178]` | .041146;.094666 `[-.037997,.120289]` | .116406;.086749 `[.043882,.188930]` | .149740;.111043 `[.056905,.242574]` |
| median-rank reduction | .125000;.231455 `[-.068501,.318501]` | .093750;.461703 `[-.292243,.479743]` | .031250;.088388 `[-.042645,.105145]` | .156250;.296934 `[-.091993,.404493]` |
| own−best-other | .002161;.000708 `[.001569,.002754]` | .003534;.002141 `[.001743,.005324]` | .006122;.002572 `[.003972,.008272]` | .010594;.003196 `[.007922,.013265]` |
| own−mean-other | −.001508;.001155 `[-.002473,−.000543]` | −.002696;.002272 `[-.004595,−.000797]` | −.001665;.002271 `[-.003564,.000233]` | −.000343;.002158 `[-.002147,.001462]` |
| pairwise | .002652;.008048 `[-.004077,.009380]` | .003741;.008606 `[-.003454,.010935]` | .010582;.007886 `[.003989,.017175]` | .013613;.010095 `[.005173,.022052]` |

### Per-family main effects

Family order is `[large, local, loop, medium, open, rough, small, visual]`.

| Metric | H1 per-family M | H2 | H3 | H4 |
|---|---|---|---|---|
| top-1 | `[-.0017,-.0035,.0104,.0122,-.0078,-.0052,.0443,-.0313]` | `[.0434,-.0104,.0156,.0451,-.0182,.0104,.0313,0]` | `[.0295,.0208,.0382,.0573,-.0495,.0313,.0313,.0885]` | `[-.0191,.0278,.0469,.0573,-.0443,.0495,.0234,.0260]` |
| top-3 | `[.0260,-.0087,.0035,.0122,-.0156,-.0026,.0599,-.0234]` | `[.0295,-.0295,.0122,-.0139,-.0156,.0052,.0391,-.0573]` | `[.0260,0,.0382,.0590,-.0495,.0208,.0156,.0156]` | `[-.0104,-.0069,.0920,.0313,-.0651,-.0104,.0365,.0599]` |
| MRR | `[.0086,-.0040,.0076,.0112,-.0112,-.0030,.0445,-.0263]` | `[.0352,-.0124,.0125,.0269,-.0213,.0078,.0299,-.0095]` | `[.0216,.0112,.0267,.0478,-.0480,.0230,.0204,.0522]` | `[-.0164,.0143,.0501,.0470,-.0436,.0325,.0198,.0474]` |
| mean-rank reduction | `[.1337,-.0590,.0191,.0434,-.0938,-.0130,.3151,-.1224]` | `[.2274,-.1458,.0573,.0712,-.1016,.1354,.1797,-.1172]` | `[.1389,-.1007,.2691,.3212,-.3750,.1797,.0495,.3672]` | `[-.1528,-.1198,.5087,.2535,-.4375,.2656,.0052,.9297]` |
| median-rank reduction | `[.1563,-.0938,0,.0313,-.1250,0,.7813,-.1563]` | `[.2500,-.1250,0,.0938,-.4375,-.0313,.4375,-.3438]` | `[.0938,-.3125,.0625,.2188,-.2500,.1563,.2188,.1875]` | `[-.1563,-.1250,.5313,.2188,-.4688,-.0313,.3438,.4063]` |
| own−best-other | `[.0027,.0052,.0005,.0021,.0007,.0001,.0040,.0010]` | `[.0073,-.0023,.0018,.0046,.0008,.0004,.0160,.0009]` | `[.0093,-.0076,.0072,.0074,-.0042,.0004,.0277,.0128]` | `[.0104,-.0085,.0102,.0159,-.0048,.0011,.0331,.0344]` |
| own−mean-other | `[.0001,-.0039,-.0031,-.0016,-.0005,0,.0007,-.0023]` | `[-.0009,-.0058,-.0028,-.0028,-.0031,.0004,.0012,-.0069]` | `[-.0015,-.0052,-.0008,-.0018,-.0044,.0009,-.0023,.0030]` | `[-.0039,-.0054,.0020,.0027,-.0045,.0013,-.0018,.0086]` |
| pairwise | `[.0122,-.0054,.0017,.0039,-.0085,-.0012,.0286,-.0111]` | `[.0207,-.0133,.0052,.0065,-.0092,.0123,.0163,-.0107]` | `[.0126,-.0092,.0245,.0292,-.0341,.0163,.0045,.0334]` | `[-.0139,-.0109,.0462,.0230,-.0398,.0241,.0005,.0845]` |

## 5. Confusion and degeneracy

### Exact H1/H2 target degeneracy

Every one of the 20 states contains these exact realized-frame duplicate pairs at H1 and H2:

- `straight_medium = go_then_turn_left`
- `turn_left_sustained = turn_left_then_go`
- `turn_right_sustained = turn_right_then_go`

There are therefore three duplicate pairs per state and no exact duplicates at H3 or H4.

Consequences:

- Candidates 7, 8, and 9 have zero H1/H2 top-1 accuracy because deterministic lower-index tie-breaking always favors their paired earlier candidate.
- The maximum possible strict branch-identity top-1 accuracy at H1/H2 is 9/12 = 0.75.
- The observed own/wrong tolerance-tie rate is exactly 1/22 = 0.0454545.
- Strict pairwise scoring treats each duplicate-pair tie as a failure.
- The registered generic pairwise chance reference remains 0.5. Accounting arithmetically for forced duplicate-pair losses gives 63/132 = 0.477273 for a target-independent random scorer and a perfect-matching ceiling of 126/132 = 0.954545. Those are explanatory consequences, not new registered endpoints.

This degeneracy applies identically across all four cells, so paired rollout comparisons remain comparable, but the absolute H1/H2 top-1 task is not a clean twelve-identifiable-target problem.

### Chance references

| Metric | Registered random reference |
|---|---:|
| top-1 | 1/12 = .083333 |
| top-3 | 3/12 = .250000 |
| expected MRR | .258601 |
| expected mean rank | 6.5 |
| expected median rank | 6.5 |
| pairwise ordering | .500000 |

All four cell point estimates beat those generic references. Their seed-level cell intervals are also favorable to the reference values, but there was no separately registered treatment-vs-chance test and no state/environment-level uncertainty interval. “All cells outperformed chance” is therefore a descriptive comparison supported by model-seed intervals conditional on these fixed states—not an environment-generalization claim.

### Pooled per-candidate behavior

Order: `straight_fast, straight_medium, straight_slow, arc_left, arc_right, turn_left, turn_right, turn_left_then_go, turn_right_then_go, go_then_turn_left, reverse_then_turn, hold`.

| H | Per-candidate top-1 accuracy | Winner frequency |
|---|---|---|
| H1 | `.255,.211,.248,.586,.333,.488,.580,0,0,0,.369,.141` | `.102,.133,.103,.154,.075,.096,.149,0,0,0,.107,.080` |
| H2 | `.209,.320,.180,.566,.439,.438,.547,0,0,0,.373,.223` | `.100,.182,.091,.133,.102,.087,.130,0,0,0,.107,.068` |
| H3 | `.180,.220,.144,.378,.319,.184,.383,.172,.406,.261,.330,.250` | `.077,.075,.058,.120,.072,.048,.087,.072,.128,.109,.104,.049` |
| H4 | `.120,.077,.225,.275,.205,.256,.392,.119,.275,.205,.186,.344` | `.046,.063,.079,.146,.047,.059,.117,.098,.097,.096,.046,.106` |

### Largest off-diagonal confusions

Counts are pooled across four cells and eight seeds, with 640 queries per source candidate:

- H1: `turn_right_then_go→turn_right` 371; `turn_left_then_go→turn_left` 312; `arc_right→turn_right` 211; `hold→reverse` 182.
- H2: `turn_right_then_go→turn_right` 350; `turn_left_then_go→turn_left` 280; `straight_slow→straight_medium` 238; `go_then_turn_left→straight_medium` 205.
- H3: `arc_right→turn_right_then_go` 198; `turn_right→arc_right` 186; `turn_left→turn_left_then_go` 175; `turn_left_then_go→arc_left` 171.
- H4: `turn_left_then_go→arc_left` 184; `arc_right→turn_right` 175; `straight_medium→straight_slow` 165; `go_then_turn_left→arc_left` 157.

The strongest H1/H2 confusions are forced by identical action prefixes and exact target duplication. Later confusions descriptively cluster among related straight-speed, left-turn/arc, and right-turn/arc actions. No frozen action-distance endpoint exists, so non-exact errors cannot be causally partitioned into physical similarity versus poor prediction.

### Similarity spread

Each entry is the range across seeds of `mean / within-checkpoint SD`, followed by the global min–max, for all predicted-query/gallery-target cosine scores.

| H | R1 | RR | P1 | PR |
|---|---|---|---|---|
| H1 | mean .782–.785; SD .106–.109; .408–.948 | .784–.787; .103–.107; .417–.949 | .782–.785; .107–.110; .407–.949 | .783–.787; .104–.107; .415–.948 |
| H2 | .744–.749; .121–.127; .371–.953 | .750–.755; .118–.122; .383–.954 | .745–.750; .122–.126; .372–.953 | .750–.756; .119–.121; .379–.953 |
| H3 | .720–.727; .128–.134; .366–.948 | .731–.738; .125–.129; .375–.949 | .721–.728; .129–.133; .368–.950 | .733–.737; .126–.128; .377–.948 |
| H4 | .703–.712; .132–.138; .381–.948 | .716–.726; .126–.131; .385–.949 | .705–.712; .133–.137; .381–.948 | .720–.724; .127–.129; .388–.948 |

There is no frozen statistic for “fraction of states where the predicted correct-vs-best-other margin is below a tolerance.” Computing one now would create a new endpoint. What is frozen is:

- tolerance-tie rate 4.54545% at H1/H2 and zero at H3/H4;
- 100% of states have the three exact target duplicate pairs at H1/H2;
- no exact target-frame duplication at H3/H4.

No family-specific “effective degeneracy” tolerance was registered. All families share the H1/H2 exact-prefix problem; none is exactly degenerate at H3/H4.

## 6. Occupancy gate and results

The frozen row observable was occupied IoU. If occupied union was zero, the value was NA.

Aggregation:

1. Mean defined branch rows within state/episode cluster.
2. Mean defined state clusters within family.
3. Primary equal-family estimate only if all eight families have at least one defined cluster.
4. Secondary corpus-weighted estimate is the mean of all defined branch rows.
5. Qualification gate is separately the pooled occupied intersection divided by pooled occupied union.

No NA was replaced with 0 or 1. No seven-family imputation was allowed.

### True-target gate

| H | Pooled true-target IoU | Floor | Verdict | Defined rows | Undefined rows | Defined states | Defined families |
|---|---:|---:|---|---:|---:|---:|---:|
| H1 | .348499173 | .35 | failed; predictor occupancy unavailable | 205 | 35 | 18/20 | 7/8 |
| H2 | .353808050 | .35 | qualified | 196 | 44 | 19/20 | 8/8 |
| H3 | .365928581 | .35 | qualified | 190 | 50 | 20/20 | 8/8 |
| H4 | .365201568 | .35 | qualified | 185 | 55 | 20/20 | 8/8 |

Undefined-row families:

- H1: local 2, open 9, rough 24. `rough_local_dynamics` was completely undefined, 0/2 states.
- H2: large 1, local 5, open 11, rough 22, small 5.
- H3: large 5, local 8, open 12, rough 20, small 5.
- H4: large 5, local 9, loop 1, open 13, rough 18, small 6, visual 3.

H1 predictor occupancy was not scored and is not an evidential predictor result.

### Qualified-horizon primary results

Cell gaps are true-target equal-family IoU minus predicted-latent IoU.

| H | True target | R1/RR/P1/PR predicted | R1/RR/P1/PR gap |
|---|---:|---|---|
| H2 | .387905 | .223015/.229111/.224011/.229116 | .164890/.158794/.163894/.158788 |
| H3 | .379702 | .172874/.178735/.166055/.175791 | .206828/.200968/.213648/.203912 |
| H4 | .372028 | .161622/.167423/.158702/.164713 | .210406/.204605/.213326/.207314 |

Primary contrasts, `mean; SD; [95% CI]`:

| H | B_R | B_P | M | J |
|---|---|---|---|---|
| H2 | .006096;.010136 `[-.002378,.014570]` | .005105;.009661 `[-.002972,.013183]` | .005601;.008831 `[-.001782,.012983]` | −.000991;.008957 `[-.008479,.006498]` |
| H3 | .005860;.012326 `[-.004445,.016166]` | .009736;.012879 `[-.001031,.020503]` | .007798;.010478 `[-.000962,.016558]` | .003876;.014015 `[-.007841,.015592]` |
| H4 | .005801;.012204 `[-.004402,.016004]` | .006011;.008773 `[-.001323,.013346]` | .005906;.009508 `[-.002043,.013855]` | .000210;.009500 `[-.007732,.008153]` |

Seed vectors:

```text
H2 BR=[.011060,.004141,.013767,.015684,-.012627,.016662,-.001440,.001522]
   BP=[.003622,-.003618,.019025,.007883,-.011708,.004953,.006096,.014588]
   M =[.007341,.000262,.016396,.011784,-.012168,.010808,.002328,.008055]
   J =[-.007437,-.007758,.005258,-.007801,.000919,-.011709,.007536,.013066]

H3 BR=[.007448,.013704,.016823,.012116,-.022854,.007272,.008791,.003584]
   BP=[.009383,-.001988,.012901,.009164,-.006605,.000138,.031094,.023801]
   M =[.008415,.005858,.014862,.010640,-.014729,.003705,.019942,.013692]
   J =[.001935,-.015691,-.003922,-.002952,.016249,-.007134,.022303,.020217]

H4 BR=[-.001575,.016344,.000639,.008019,-.017076,.018206,.003319,.018532]
   BP=[-.005022,.013409,.009547,-.000205,-.002874,.020763,.008935,.003538]
   M =[-.003299,.014876,.005093,.003907,-.009975,.019484,.006127,.011035]
   J =[-.003446,-.002936,.008908,-.008224,.014202,.002557,.005616,-.014994]
```

The main effect is positive for 7/8 seeds at H2, 7/8 at H3, and 6/8 at H4. It is not uniformly positive. Seed `2026080905` is negative at all qualified horizons, and seed `2026080901` is additionally negative at H4.

### Corpus-weighted secondary occupancy

| H | Cells R1/RR/P1/PR | M; SD `[CI]` | J `[CI]` |
|---|---|---|---|
| H2 | .248975/.255143/.249417/.253257 | .005004;.011616 `[-.004707,.014715]` | −.002328 `[-.013194,.008538]` |
| H3 | .197901/.206319/.189495/.203420 | .011171;.012143 `[.001020,.021323]` | .005508 `[-.009067,.020082]` |
| H4 | .185867/.194133/.181277/.189164 | .008077;.009316 `[.000288,.015865]` | −.000378 `[-.011830,.011073]` |

Only the secondary H3 and H4 main intervals exclude zero. The primary equal-family intervals do not.

### Per-family occupancy

Each row is `true target; R1/RR/P1/PR; main M`.

| H | Family | Result |
|---|---|---|
| H2 | large | .4831; .2576/.2589/.2570/.2625; M=.0034 |
| H2 | local | .2827; .1091/.1314/.1092/.1114; M=.0123 |
| H2 | loop | .3256; .1770/.1961/.1789/.1863; M=.0132 |
| H2 | medium | .4056; .1826/.1595/.1741/.1664; M=−.0154 |
| H2 | open | .1170; .0209/.0065/.0157/.0066; M=−.0118 |
| H2 | rough | .1000; 0/0/0/0; M=0 |
| H2 | small | .6258; .3368/.3735/.3369/.3950; M=.0474 |
| H2 | visual | .7634; .7001/.7071/.7203/.7047; M=−.0043 |
| H3 | large | .5238; .2193/.2262/.2150/.2304; M=.0112 |
| H3 | local | .3214; .0784/.0864/.0738/.0778; M=.0061 |
| H3 | loop | .3380; .1656/.1778/.1553/.1717; M=.0143 |
| H3 | medium | .4187; .1370/.1392/.1308/.1446; M=.0081 |
| H3 | open | .1485; .0168/.0028/.0080/.0045; M=−.0087 |
| H3 | rough | .1042; 0/0/0/0; M=0 |
| H3 | small | .4949; .2182/.2426/.2037/.2281; M=.0243 |
| H3 | visual | .6883; .5478/.5549/.5418/.5491; M=.0071 |
| H4 | large | .4786; .1667/.1697/.1576/.1698; M=.0077 |
| H4 | local | .2256; .0754/.0697/.0710/.0646; M=−.0061 |
| H4 | loop | .3608; .1834/.1918/.1732/.1837; M=.0095 |
| H4 | medium | .4392; .1400/.1415/.1335/.1401; M=.0041 |
| H4 | open | .1255; .0134/.0050/.0075/.0020; M=−.0069 |
| H4 | rough | .0817; 0/0/0/0; M=0 |
| H4 | small | .5613; .2084/.2437/.2136/.2509; M=.0363 |
| H4 | visual | .7034; .5058/.5180/.5133/.5064; M=.0027 |

Family signs are heterogeneous:

- H2: four positive, three negative, rough zero.
- H3: six positive, one negative, rough zero.
- H4: five positive, two negative, rough zero.

`small_enclosed_maze` contributes the largest positive effects. `open_obstacle_field` is negative throughout.

Relative to the mean one-step predicted IoU, primary M is approximately 2.51% at H2, 4.60% at H3, and 3.69% at H4. It closes only about 3.41%, 3.71%, and 2.79% of the true-target gap, respectively. M/seed-SD is .63, .74, and .62. These are positive tendencies, not established primary effects.

## 7. Claims supported by the completed assay

1. **Rollout improves direct counterfactual future fidelity:** established under the prespecified eight-seed equal-family estimator at H1–4. All main cosine and error-reduction intervals are favorable. The absolute improvements are modest, especially H1–2.

2. **Rollout improves action-specific future discrimination:** partially established. Main top-1 is positive at H2–4; H3 additionally supports top-3, MRR, mean-rank reduction, and pairwise accuracy; H4 supports MRR, mean-rank reduction, and pairwise accuracy. H1 top-1 is null. Results depend on the retrieval metric.

3. **The effect extends beyond the directly supervised H=2 horizon:** supported for direct fidelity at H3–4 and for several retrieval metrics at H3–4, conditional on these frozen states and model seeds.

4. **Predicted latents retain some occupancy geometry:** descriptively supported at H2–4 because the frozen probe qualifies on true targets and predicted IoUs are nonzero in most families. Predicted IoUs remain well below true-target IoUs.

5. **No material proprioception amplification:** supported as a null result under this assay. All direct-fidelity `J` intervals include zero; retrieval interactions are generally null aside from an isolated negative H1 top-3 interaction; all occupancy `J` intervals include zero.

## 8. Claims not supported

1. **Rollout improves occupancy retention as a treatment effect:** not established by the primary estimator. All H2–4 primary intervals include zero. H1 is unavailable because the true-target probe failed.

2. **Rollout improves planning or candidate utility:** not tested. No utility score, oracle-utility ranking, selected-candidate utility, or closed-loop planning endpoint was evaluated.

3. **The scorer would work:** not tested. No utility scorer was trained or invoked.

4. **Generalization beyond these 20 development states:** not supported. The t-intervals quantify model-training-seed variation conditional on the same successful-pilot states.

5. **Generalization to new families or environments:** not supported. The eight families are fixed, family counts are small and unequal, and late-horizon family effects are heterogeneous.

6. **Twelve fully distinguishable actions at H1/H2:** false. Three exact target pairs occur in every state.

7. **Every retrieval metric improves:** false. H2 top-3 is slightly negative, median-rank effects are imprecise throughout, and own-minus-mean-other margin worsens significantly at H1–2 even while own-minus-best-other margin improves.

## 9. Audit and provenance

### Primary machine-readable files

- [Scientific report](</home/andrewknowles/Workspace/LeWMQuad-v3/docs/lewm_go2_counterfactual_predictor_qualification_v1_2_result_2026-08-11.md>)
- [Stage-A identity manifest](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/stage_a_identity_manifest.json>)
- [Corpus receipt](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/corpus_receipt.json>)
- [Target-latent index](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/latents_index.json>)
- [B/C assay specification](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/predictor_assay/assay_spec.json>)
- [B/C result](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/predictor_assay/result.json>)
- [Occupancy label index](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/occupancy_labels/labels_index.json>)
- [Occupancy true-target gate](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/occupancy_results/true_target_gate.json>)
- [Occupancy result](</home/andrewknowles/Workspace/LeWMQuad-v3/.generated/go2_counterfactual_fidelity_v1_2/occupancy_results/result.json>)

### Complete artifact digests

“Canonical” refers to the registered payload/self-digest; “raw SHA” refers to the JSON or Markdown file bytes.

| Artifact | Canonical digest | Raw file SHA-256 |
|---|---|---|
| frozen commit | `ee47b47e7964c16360f265c4cfbe7f8181d16402` | — |
| Markdown report | — | `14a4276b1caee817a7097eb78f187c9e38b6d4c7eb70a16f88ec32ccd223894b` |
| Stage-A identity | `ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a` | `1734ba63949227155d9423ffd017c862f81b1b111711fbd23ccfe6a349831dc4` |
| branch corpus | `f84eb3271f1a3b7052bbf2e84240453e84772b0a530e60ec47f723a44e2e10e9` | receipt `0a3561fc75edda21201ec1ec4491c95971bd6ce0f7c6155c61c4001acbdbdd4b` |
| branch ledger | — | `2b71c488851c6d4b7e3a36a46637a4e5be4896ae48a84d1498c6e8a8d3d74c81` |
| target-latent index | `861285ec9c8fc6c92c6f3a31cade0f031172bf6818d76d1899634a60c7e5c291` | `70a9f9b405fa0cb4ad1960cd76841ba12f1580e45b0e56fb21ced4f9369a9ab2` |
| B/C assay spec | `a26fa0ec9ee9e0df3bbe71fff6d7594bb714227aaa66a66631836d94a676feab` | `af8ca3327b2a174f680a0f4fab814176e743900abdaf945bf68ae3fe15ae7a15` |
| B/C result | `3b5c500b4b1326056ce18c6276d7842f4230faec36f8f29cc65945f54527bbcb` | `d3f5ade362a2df4546d3c6cfe7d5f3fc1d3ee0216fa13eef4cba0e2a48f028be` |
| occupancy label index | `a81f1c63f9fa181bfa728b1cb5da2ad4573f2aa80cb5801c9d54acab34d411e2` | `0d37afe87ffda7289a8dd4be860a39516bf2d9b477fae974de3f74fcc6129b1e` |
| occupancy label corpus | `a402ee134a0ec854b9936699e42e0a2c715ea70ac99a2c0393ee09ba6ac41a27` | — |
| occupancy gate | `4bf9a92144fa728d953c9dffebb235c9b476ded59d7462a107fe2e6ade0894e4` | `ba83c87b0de17b173c638b759623a50388c03a823775dc7400ed764905eca921` |
| occupancy result | `09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6` | `f9e6e47f8b8208e00b31836b5347424c368b7a5dcf96d9037cf9925e04d1a0af` |
| occupancy NA-equivalence receipt | `b5e608d78fb0ab3b8afcdc3d04f58c666956cbdf6f6e923da2b0c03ee1a6d0dc` | `85f825ff334860cde6e7e2bdd0b419e3df7ce3c124e8868b265ae91d04a5c39c` |
| occupancy reporting repair | `287d1e4a226304242c279c548a542cfd8fca54c5fd4159109d3382a6f9591fda` | `780aac504b2e11c550406e279a42fef4d77232f1eaa20279cebc6cce60f5c08d` |
| frozen occupancy probe package | `b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686` | weights `95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322` |

### Predictor checkpoints

All 32 epoch-21 checkpoints were hash-verified against the run package before first load. All 32 were opened for B/C; Stage D opened none and consumed the frozen prediction shards.

| Seed | R1 | RR | P1 | PR |
|---|---|---|---|---|
| 2026080901 | `20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a` | `75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4` | `41d1c5a48d7adacf2e2b698318782de29c7b95342181bdf5fd5578d35346f1d1` | `75ab2a5dd5c48ebb2f33935d962d957c4e62eab3427ce0ad8108d690a1df9218` |
| 2026080902 | `085702386da4b36bafe6ff432ca955a2b1a9a69de9a8023aa4fc3b099953f0ff` | `90bbf9a8117dbf528d9693415becd5c9e9605ecad02520f3e00513dfee691530` | `76c9c5328217aceee64e7d4a60524d8317b459c68ca4da05a24509dfd2c94dc9` | `030a28078acbc495a3a79e0e513501586a8271938200f194a4632ed08b49fca8` |
| 2026080903 | `a7878c6159cceae8f69f84927bd1ee3a4c3d8dbf6d1e97003eb9ebdae1f91bc4` | `b769ef91f1ef17377f7c7f184c85ea0a9859ead2b87aa8351a89b7a05192aad1` | `ab440561a867e1961c156ed556271ece87a51b959c3ce4e4b527a1d9136c46d2` | `591538880c4beaf982196f36364a18fb4167d80f6671912975d4ad454f731545` |
| 2026080904 | `1386b6303ac5b47fea7a67e831a375d164ba372ee4bf60fd87609ed35352d1ff` | `aad6711b6d15e6664038ace1fe0f376516256062c2235334b74bfb68135e419a` | `bdfed483f5a173eec40a8b9d6c586b478a2a4929f7291b13c087d98eb336eee0` | `bdc2c7d5f09472e3fcf7813ff316c8f6bef021a0ee1f2900178bbb3b52b8e0e0` |
| 2026080905 | `5d78f18e0d0052479cb81a43acbaa953bebeb6fc13dac58c506211c46416a1e9` | `c474a5b09c041aa263950b3b2b8bd2369d3644aec7019268610fea4b846b6386` | `8f53f8991ffbf1994a9b6ff74087c8c23c908c8ad49bdd892ac2b65394501cb8` | `3a615f5d00dc106d24c3719489ba04e52a8bb4f97e49a385f1d8f4908d24aad4` |
| 2026080906 | `846fbe05f78e9b513841cb08f71858e9fdb7dd4430181bca140d29f72574a200` | `fc480799cc637f5c3d4bd582da233e38b76d422b48833075e018d49df517aa1a` | `464fd320561b1e92770231e849df53ef3ee5ab08f85e54c40218830863beb309` | `daa77c7ed9600fcd17b37fd1cd3a73c1ad0902f439e111502436ca247030f6cb` |
| 2026080907 | `86d9f6108f40b8d2cf49e5264fc998412493258258b86391067c71193066afbc` | `4501841125eee43568e6031d4061d23b309c080f11b129538dadb6cfc8a05432` | `673f4c4251706ef49f16fdb9d1e48e391cdf871f34965e779fda5609aa78aff8` | `4c09a551ab89f55260cbbd24937ea81d5cb081bd319a8f61118e7d2ddd488f89` |
| 2026080908 | `025aff4d9bc7380b4a51e4ac08282bbeafb2be189bf27d31d48bcf247f2b02f2` | `a39f5050c02ab7b002c6b1c76256dc2b5783046cf5b877cc6d5354880c45b89a` | `aa4cef094a3d503ea1062a59a4d36b9f4b66eb03a6f958b32ef8e3d54f5ab94a` | `6027d657ce81d8ae968354031e666c9a608d800e25663629944590f448488b4f` |

Verification verdict: 32/32 checkpoint bytes, hashes, run records, final-analysis receipts, and run-package bindings agreed.

### Interruptions and invalid attempts

- The invalid interrupted 45-state scorer-fit identity set was preserved and never entered this assay.
- Two pre-corpus Stage-A attempts were preserved under:
  - `.../invalid/go2_counterfactual_fidelity_v1_2_33ae1534_gate_assay_omission`
  - `.../invalid/go2_counterfactual_fidelity_v1_2_e9d31ab9_encoder_smoke_index_narrowing`
- Neither was mixed into the final corpus.
- Two B/C validate-only implementation errors occurred before loading any predictor or writing a prediction result.
- Predictor scoring was manually interrupted after 169 durable checkpoint×state units, before metric analysis. The identical operation resumed only missing units. Seed `2026080903/R1` records `resumed_completed_states=9`; no completed state shard was rerun.
- Stage D initially stopped on undefined H1 family aggregation before freezing the gate or reading prediction shards. The reporting-only correction retained all true records.
- A later episode-count reporting correction retained all 240 labels, 240 true records, and 640 predicted-state records byte-for-byte. Superseded reports were preserved.
- No branch, state, checkpoint, or model run was repeated because of an observed metric.
- No valid completed branch was regenerated.
- No utility or energy scorer was trained or invoked.

Measured runtime was approximately 242.4 s for Stage-A branching, 216.3 s for target encoding, 1,535.1 s for final B/C execution, and 24.6 s for the final occupancy no-op validation/report pass. The B/C prediction package occupies approximately 48.36 GB; target latents approximately 1.60 GB.

## 10. Material discrepancies and limitations

1. **Retrieval is horizon-specific, not cumulative trajectory retrieval.** The prose sometimes says “predicted trajectory,” but each rank uses only one horizon’s 768-token latent.

2. **The prose omits universal H1/H2 exact duplicates.** This creates deterministic candidate-index bias and a 0.75 top-1 ceiling. It materially weakens absolute H1/H2 branch-identification claims, though paired cell contrasts remain comparable.

3. **“Exact tie rate” is slightly imprecise wording.** The implementation reports tolerance ties at `1e−12`; ranking uses raw values.

4. **The B/C machine-readable recovery block is incomplete.** It says `events=[]` and `invalid_or_interrupted_attempts_preserved=false`, whereas the report and checkpoint receipt document the SIGINT/resume and preserved invalid attempts. This is a provenance-reporting defect, not a metric mismatch.

5. **H1 proprio-only direct intervals are rounded misleadingly in prose.** Their exact lower bounds are very slightly negative. The prespecified averaged main effect remains strictly positive.

6. **Occupancy qualification is marginal and NA-heavy.** H2 passes by only `.003808`; H1 misses by `.001501`. Undefined rows range from 18% to 23%, and the secondary estimator conditions on nonzero occupied union.

7. **Occupancy treatment results are estimator-sensitive.** Secondary H3/H4 intervals are positive, while all primary equal-family intervals include zero.

8. **Family heterogeneity is substantial.** `open_obstacle_field` is negative on late direct fidelity and most retrieval/occupancy summaries; `small_enclosed_maze` contributes some of the largest positive effects.

9. **Own-best and own-mean margins tell different stories.** Own-best margins remain negative in every cell and horizon, although rollout makes them less negative. Own-mean margins are positive, but rollout significantly reduces them at H1/H2.

10. **Environment generalization is unquantified.** These are the 20 successful oracle-pilot development states, not a new random environment sample.

The Markdown prose otherwise agrees numerically with the machine-readable results. No scientific computation was rerun for this evidence extract, no files were changed, and no assay or scorer process remains running.

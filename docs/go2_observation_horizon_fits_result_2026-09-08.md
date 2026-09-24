# Observation-horizon eighteen-fit result

All eighteen fresh short-horizon models completed their prescribed 1,200
updates: 21,600 updates total. Full admission reconstructed every training
ledger, both-role raw scores and all restricted final snapshots. There was no
checkpoint selection, seed selection, resume or benchmark-weight reuse.

The models predict eight actual 100-ms steps through 800 ms, with one known
command per transition token. They use the same three context schedules and
seeds, full/no-RGB treatments, and direct/supervised-rollout/JEPA conditions as
the predecessor. Target cadence and temporal architecture changed; whole-model
initialization identity across the two studies is not claimed. Within each new
seed, all six treatments had exactly the same initialization. Training retains
the declared loss coefficients and 0.06-m position scale.

The fixed native assignments remain seed 2026091001 full-JEPA and full-direct,
as declared before results. Neither the metrics below nor the separate mapping
optimization prototype changes that assignment or the prepared controller.

| Treatment | Family transfer 100-ms XY error, mm | Switch transfer 100-ms XY error, mm | Switch transfer 500-ms XY error, mm |
| --- | ---: | ---: | ---: |
| Full direct | 21.405 ± 3.135 | 20.055 ± 1.880 | 23.601 ± 3.357 |
| Full supervised rollout | 18.169 ± 2.734 | 17.738 ± 2.145 | 14.286 ± 1.816 |
| Full JEPA | 31.013 ± 7.596 | 32.834 ± 9.390 | 36.020 ± 12.706 |
| No-RGB direct | 19.016 ± 12.069 | 17.929 ± 11.929 | 17.350 ± 9.849 |
| No-RGB supervised rollout | 19.818 ± 2.884 | 20.213 ± 2.571 | 14.909 ± 3.143 |
| No-RGB JEPA | 26.379 ± 5.117 | 24.705 ± 1.763 | 23.867 ± 1.880 |

Values are means and sample standard deviations across the three optimization
seeds, not confidence intervals across independent mazes. At 100 ms, full
direct's switch-transfer yaw error is 0.016093 ± 0.005254 rad; full JEPA's is
0.041908 ± 0.003596 rad. Full supervised rollout has lower mean XY error but
variable yaw error, 0.054052 ± 0.062182 rad. There is no demonstrated JEPA or
general RGB advantage in this readout. No-RGB removes predictor RGB, not the
separate RGB-D observer or geometry processing in navigation.

For the fixed first seed, full direct's 100-ms XY/yaw errors are
17.798 mm/0.014110 rad on family transfer and 17.891 mm/0.013370 rad on switch
transfer. Full JEPA's are 29.849 mm/0.048195 rad and 29.838 mm/0.044143 rad.
These are prediction errors, not clearance bounds or arrival evidence.

The all-model readout compares the shared 500-ms target with exact matching
motion/contact denominators and unchanged context schedules. For the fixed
first seed:

| Treatment and source | Predecessor XY error, mm | Short-horizon XY error, mm | Predecessor yaw error, rad | Short-horizon yaw error, rad |
| --- | ---: | ---: | ---: | ---: |
| Full direct, family | 19.872 | 20.917 | 0.014313 | 0.017127 |
| Full direct, switch | 21.346 | 21.126 | 0.025617 | 0.024177 |
| Full JEPA, family | 41.386 | 33.657 | 0.064890 | 0.044907 |
| Full JEPA, switch | 38.314 | 32.020 | 0.093118 | 0.036888 |

JEPA improves against its own predecessor at the shared horizon but still
trails direct prediction. Direct results are mixed across the two sources.
The new 100-ms predictions cannot be called better than nonexistent predecessor
100-ms outputs. The native probe must test whether the correctly timed first
forecast helps actual closed-loop action selection.

At 100 ms, family transfer includes 344 motion targets, 348 contact labels and
four contact positives; switch transfer has 72 motion/contact targets and zero
positives. At 500 ms those counts are 338/348/10 and 72/72/0. All contact-censored
motion, unavailable contexts and undefined-yaw accounting remain explicit.
Full per-model, source/stratum/horizon, train/transfer and contact-Brier results
are retained in `metrics.json`; the tables do not redefine the population.

The fit phase used four fresh CPU workers, each with one numerical thread,
selected by the separate exact serial/parallel benchmark. It took
1,658.839750 seconds (27.65 minutes), with maximum worker RSS 2,718,117,888 bytes.
The launch measured 82,305,646,592 bytes available RAM and 69,858,574,336 bytes
free artifact storage. Its 1,125 bound sources and 202 artifacts (157,939,305
bytes) passed final verification. The all-model readout has 1,128 bound sources;
its post-admission phase took 10.862481 seconds. The earlier profiler ran
concurrently for a bounded part of fitting, so fit wall time is descriptive.

Artifact roots:
`go2_observation_horizon_fits_v1_attempt_001` and
`go2_observation_horizon_fit_readout_v1_attempt_001`.

| Artifact | SHA-256 |
| --- | --- |
| Target derivation result | `fe3ab252e6da0ebadba13927c0dad7410d2084145c3b50439f6848ea5b65a775` |
| Input-check result | `73e11e168f933633dbbd3b82668a5b021c076e18bbfc7704987008b4d290c1b5` |
| Separate fit benchmark result | `67a35ae4b94ceaa09627e806f9b1f894acb20aff0b1fb664bae1eda5db6909e6` |
| Fit launch | `58ef5b311211edc3a0bead90d0399f941553eb3d348386b44103a7c2a512f710` |
| Fit result | `45b4680b85c87bd69dcaed6a0058f091105632909661dba319d6c05f5b533418` |
| Readout launch | `61c9e06d3815d200a0c703f068562ce6ab30b5ea1d41456f2d59650f2f16ed30` |
| Readout metrics | `b3fe0f35fd6d9bbb6e559c3f917bc48a22e21788cb3e1d61c182b02c33f276cf` |
| Readout result | `b7b9aac1b8c44ee924e8d940e782f016a0693be5149c91bcb5f6ed676ad30a25` |

There is no new native execution in this result and still no verified arrival.
The data are development geometry families, not independent novel maze missions.
Matched reactive/nonpredictive, planning and memory baselines, physical
backtracking, independent layouts, realistic continuous timing and bounded
hardware evidence remain outstanding.

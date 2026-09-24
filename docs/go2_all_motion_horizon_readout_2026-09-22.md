# Matched horizon coverage across translation and heading training data

Status: both 440-update CPU fits completed successfully, with owner exit zero.
Encoding took 9149.3 seconds and the full fitting workflow 9181.13 seconds.
The prepared transfer evaluation completed on CPU cores 4--7, session 45020,
with exit zero in 550.88 seconds. Broader horizon coverage modestly improves
700-ms action translation XY error against the matched fixed-horizon fit, but
does not improve translation yaw or approach command-history accuracy. Actual
future images still give large motion-readout errors. Neither head is promoted;
the prospective navigation cohort remains unchanged.

Final checkpoint SHA-256 identities:
`fixed_500ms`: `c98b0412391e701768eb64a5562eb3f817a45f210da9e5ae2bc1835053c12246`;
`multi_100_800ms`: `6a264a6239702881bad90e3329c23801c53ecb5d726d45a5baf6d870f01ccf53`.

The preceding heading-only horizon extension modestly improved the 700-ms
readout but did not approach command-history accuracy. Its new horizon targets
contained translations below 28 mm, whereas 52/116 translation windows in the
completed maze-00 action run exceeded the original 500-ms training maximum of
105.21 mm. Existing original training recordings retain longer-horizon targets,
reaching 144.78 mm at 700 ms and 163.95 mm at 800 ms. This experiment uses that
available coverage without collecting new simulations or fitting on maze data.

| Setting | Fixed-horizon control | Multi-horizon treatment |
|---|---|---|
| Starting checkpoint | Original mixed-data readout | Identical |
| Original departure contexts | 3158 valid at every horizon | Identical |
| Full-heading departure contexts | 2808 valid at every horizon | Identical |
| Batch composition | 32 original plus 32 heading | Identical departures/order |
| Future interval | 500 ms in both halves | 100--800 ms in both halves |
| Horizon presentations per half | All 500 ms | 1760 per horizon |
| Optimization | 440 updates, AdamW 0.001, weight decay 0.0001, clip 1 | Identical |
| Architecture / target normalization | Unchanged | Identical |
| Checkpoint selection | Fixed final update | Identical |

Seed: 2026092202. The frozen encoder and action-conditioned predictor do not
change. Both heads start from SHA-256
`bbbb05fd2e2984ac4d818abc986ec0401e24e9d91f53ee4e5b49443b832bdf85`.
The common eight-horizon original population excludes 360 late recording
contexts; this exclusion applies to both arms. Original 500-ms targets reproduce
the earlier training targets within 1e-7. RGB paths and training-role labels were
checked, and any depth-retention markers were consulted before image loading.
Only existing RGB is used. Pooled features stay in RAM; no depth or dense tensor
cache is written. Preparation found 7,086 image paths; live encoding deduplicated
these to 6,526 unique byte contents and completed successfully. Encoding/fitting
used CPU cores 4--7, separately from the ongoing GPU navigation process.

## Completed translation-error diagnosis during fitting

A post-hoc decomposition of the 116 existing maze-00 translation windows
projects 700-ms prediction-minus-truth onto the actual displacement direction
and its perpendicular. This uses saved predictions and evaluator endpoints;
no sensor replay, fitting, calibration or active-experiment change occurred.
Recomputed endpoint errors reproduce the saved physical reader within 1e-10 m.

| Population | Windows | Learned XY RMSE | Learned parallel bias | Learned transverse RMSE | Command-history XY RMSE |
|---|---:|---:|---:|---:|---:|
| All translations | 116 | 68.48 mm | -61.39 mm | 3.60 mm | 10.55 mm |
| Within original 500-ms training displacement maximum | 64 | 47.19 mm | -41.26 mm | 3.92 mm | 12.09 mm |
| Above original maximum | 52 | 87.87 mm | -86.16 mm | 3.18 mm | 8.27 mm |

Negative parallel bias means underestimated progress. Learned forecasts
underestimate progress in 113/116 windows (61/64 within the original range and
52/52 above it). Parallel error accounts for 99.72% of total squared XY error;
the median predicted progress is 27.67% of actual displacement, versus 101.92%
for command history. Eleven learned predictions point backwards along the
actual displacement direction; all eleven are in the within-range group.

This supports systematic displacement underprediction on this trajectory, not
merely large perpendicular error or an error confined to motions above the
old training maximum. It does not establish a causal training-coverage effect
or justify a multiplicative correction. The current experiment's unchanged
actual-future and predicted-feature comparisons remain necessary to distinguish
where the error arises. Windows overlap; this is one exposed trajectory, not
independent trials or a counterfactual navigation result.

Reproducer: `scripts/read_go2_maze_translation_error_components_development.py`.
All per-window components and input identities are retained in
`translation_error_components_v1.json` under the completed maze-00 action root.

## Completed starting-head actual-future translation diagnostic

The unchanged starting mixed-data head was evaluated on actual current/future
images for the already fixed 32 translation windows, at 500 and 700 ms. The
CPU-only job on cores 0--3 completed in 141.23 seconds, exit zero, encoding 96
images into pooled features in RAM. It used no predictor inference, GPU,
training, new simulations or tensor cache. Saved online action forecasts and
command-history forecasts supply the same-window references. Both original
jobs continued on their existing CPU/GPU allocations.

| Starting head / reference | 500-ms XY / yaw RMSE | 700-ms XY / yaw RMSE |
|---|---:|---:|
| Actual future images | 53.20 mm / 5.44 degrees | 78.28 mm / 7.47 degrees |
| Saved action-conditioned predicted features | 42.49 mm / 4.91 degrees | 67.17 mm / 7.02 degrees |
| Command history | 6.13 mm / 0.99 degrees | 8.65 mm / 1.06 degrees |
| Zero motion | 63.89 mm / 8.22 degrees | 92.66 mm / 11.26 degrees |

At 700 ms, the actual-future head underestimates progress in all 32 windows,
with -68.74 mm mean parallel bias, median progress fraction 16.63%, and 99.82%
of squared XY error parallel to actual displacement. The saved action forecasts
underestimate 31/32, with median progress fraction 28.93%; command history's
median is 102.04%. At the trained 500-ms horizon, actual-future estimates also
underpredict in 28/32 windows, with 22.10% median progress fraction.

The actual-future input removes learned-predictor error but does not repair
translation estimation. This implicates the representation/readout combination
as an additional limitation, including at its original training horizon; it
does not isolate the encoder from the head, show that geometry is absent, or
establish that horizon extension will resolve the failure. Actual-future error
being larger than predicted-feature error is not an additive decomposition of
causal contributions. The frozen comparison remains unchanged; its later
full evaluation will recompute this baseline alongside both new heads.

Reproducer: `scripts/evaluate_go2_starting_translation_oracle_development.py`.
`starting_translation_oracle_v1/plan.json` and `result.json` under this training
root retain identities, all 64 rows, predictions, errors and parallel/transverse
components. Future images and native targets are offline evaluator-only inputs.
This remains one exposed trajectory with overlapping windows, not independent
navigation or counterfactual action evidence.

## Transfer population fixed before fitting

The completed maze-00 action trajectory supplies 32 equally spaced chronological
translation windows from all 116 eligible translations, plus the previously
selected 32 turn diagnostic windows. Selection does not use forecast error.
Requested and applied command tapes match all 20-ms steps through 700 ms.
The fixed population has 128 window/horizon rows and 317 unique image-frame
indices. Primary endpoints are XY/yaw RMSE at 700 ms on translations; 500-ms
translation and both turn horizons are secondary. All remain exposed development
data. None enters fitting.

Each head is evaluated on actual future features, action-conditioned predicted
features and action-blind predicted features, alongside the starting mixed head,
command history and zero motion. Actual future images and native poses are
offline evaluator inputs only. The starting head must reproduce saved online
forecasts within 2e-5. No head is automatically promoted and no new navigation
claim follows from these readouts.

This one-seed experiment changes temporal interval and motion magnitude together;
it cannot separate their effects. It is a readout intervention, not a JEPA
objective experiment. The selected windows overlap and do not establish
counterfactual action outcomes, cross-family transfer or independent navigation.

Training runner: `scripts/train_go2_all_motion_horizon_readout_development.py`.
Evaluator: `scripts/evaluate_go2_all_motion_horizon_readout_development.py`.
Both `--prepare` stages completed successfully. The default training command
completed with exit zero; the evaluator was then launched without `--prepare`
on the same CPU group. Do not repeat either preparation command or overwrite an
existing attempt. The evaluator completed successfully in session 45020.

Artifact root:
`.generated/navigation_development_artifacts_v1/go2_all_motion_horizon_readout_v1_attempt_001/`.
It retains `plan.json`, `samples.json`, `frame_paths.json`, `schedule.json`,
`transfer_plan.json`, `transfer_targets.json` and the historical `process.json`.
The training owner was PID 122325, tool session 64660, now exited successfully.
The separate navigation coordinator remains session 64661. Do not restart a
live job merely because an observation wait expires.
Completed fits retain two final checkpoints and `result.json`; completed transfer
evaluation is retained in `maze00_evaluation/`. The preceding completed comparison
is recorded in `go2_multihorizon_motion_readout_2026-09-22.md`.

## Completed fixed transfer comparison

All 128 preselected window/horizon rows completed: 32 translation and 32 turn
windows, each at 500 and 700 ms. Starting-head action predictions reproduced
saved online outputs within 2e-5. The comparison uses the fixed final checkpoints;
no outcome-dependent checkpoint or window selection occurred. Each cell below
is XY RMSE in mm / yaw RMSE in degrees.

| Head / future input or reference | Translation 500 ms | Translation 700 ms | Turn 500 ms | Turn 700 ms |
|---|---:|---:|---:|---:|
| Starting / actual future | 53.20 / 5.44 | 78.28 / 7.47 | 17.29 / 5.92 | 19.96 / 10.35 |
| Starting / action prediction | 42.49 / 4.91 | 67.17 / 7.02 | 16.82 / 6.34 | 18.36 / 10.24 |
| Starting / action-blind prediction | 47.64 / 7.55 | 73.17 / 10.55 | 15.02 / 8.08 | 16.37 / 12.48 |
| Fixed 500 ms / actual future | 53.90 / 6.01 | 79.52 / 8.01 | 19.14 / 5.79 | 20.41 / 9.69 |
| Fixed 500 ms / action prediction | 44.52 / 5.60 | 70.33 / 7.61 | 16.84 / 6.26 | 18.31 / 9.74 |
| Fixed 500 ms / action-blind prediction | 47.89 / 7.48 | 73.79 / 10.58 | 14.88 / 8.18 | 16.12 / 12.47 |
| Multi-horizon / actual future | 54.30 / 5.86 | 77.38 / 8.16 | 17.66 / 6.00 | 20.83 / 10.13 |
| Multi-horizon / action prediction | 42.55 / 5.25 | 66.28 / 7.62 | 16.75 / 6.61 | 17.55 / 9.55 |
| Multi-horizon / action-blind prediction | 48.89 / 7.18 | 72.65 / 10.35 | 14.75 / 8.67 | 16.56 / 12.82 |
| Command history | 6.13 / 0.99 | 8.65 / 1.06 | 6.00 / 0.59 | 6.72 / 0.77 |
| Zero motion | 63.89 / 8.22 | 92.66 / 11.26 | 16.40 / 9.81 | 17.76 / 14.32 |

On the primary 700-ms translation endpoint, multi-horizon action XY error is
5.8% lower than matched fixed-horizon fitting (70.33 to 66.28 mm), but only
1.3% lower than the starting head (67.17 mm). Yaw is essentially unchanged
versus the matched fit and worse than the starting head (7.62 versus 7.02
degrees). Actual-future translation XY improves slightly to 77.38 mm, while
actual-future yaw worsens. At 500 ms, actual-future XY also worsens. The
700-ms turn action errors improve modestly against fixed-horizon fitting,
while 500-ms turn yaw regresses. There is no uniform transfer improvement.

The broader coverage intervention did not repair the motion interface. Its
700-ms translation action error remains about 7.7 times command history's.
Actual future images remove learned-predictor error yet retain large translation
errors, so attributing this failure solely to prediction or horizon mismatch is
unsupported. This does not identify encoder versus readout causality or prove
that physical information is absent from dense features. Action predictions
still improve translation errors over the corresponding blind forecasts;
that relative benefit is insufficient to establish a practical navigation gain.

Neither checkpoint is promoted into navigation. Both final checkpoints, the
negative outcomes and all per-window predictions remain retained. Further work
should address transfer of the representation/readout interface rather than
assuming another horizon-only fit will solve it. The current prospective maze
comparison continues unchanged. This is one exposed trajectory with overlapping
windows, not a JEPA-objective comparison or independent navigation evidence.

Raw result: `maze00_evaluation/result.json` under the artifact root above;
fixed transfer-plan SHA-256:
`b19067be5562b9a82cc04310962a7737b0846add6c1b7a29913ad7cfcf928d60`.

## Completed training-example translation diagnostic

To distinguish gross fitting failure from transfer limitations, a separate
post-hoc actual-future evaluation selected original training departures with
700-ms displacement at least 50 mm and at least one recorded presentation of
that exact 700-ms pair to the multi-horizon head. There were 447 eligible
contexts in 64 recordings. Selection took 32 evenly spaced sorted recordings
and the median eligible frame in each, without using prediction errors. These
are ten family and twenty-two switch recordings. Every selected 500-ms pair
was presented four or five times to the fixed head; every selected 700-ms pair
was presented once or twice to the multi-horizon head. The latter saw only
11/32 exact selected 500-ms pairs during this continuation, which matters when
interpreting its 500-ms regression.

The 96-image CPU evaluation completed in 135.45 seconds, exit zero, with no
training, predictor inference, simulation or tensor cache. It uses actual
current/future RGB features and the same three frozen heads. Each cell is
XY RMSE in mm / yaw RMSE in degrees.

| Head/reference | Training examples 500 ms | Training examples 700 ms | Maze translations 500 ms | Maze translations 700 ms |
|---|---:|---:|---:|---:|
| Starting mixed | 7.53 / 0.94 | 28.20 / 3.44 | 53.20 / 5.44 | 78.28 / 7.47 |
| Fixed 500 ms | 4.74 / 0.68 | 28.31 / 3.43 | 53.90 / 6.01 | 79.52 / 8.01 |
| Multi-horizon | 14.80 / 1.49 | 11.19 / 1.77 | 54.30 / 5.86 | 77.38 / 8.16 |
| Zero motion | 79.63 / 10.06 | 108.49 / 13.00 | 63.89 / 8.22 | 92.66 / 11.26 |

Selected training 700-ms displacements range from 50.81 to 138.59 mm, with
median 111.11 mm. Thus the small training-example error is not explained by
testing only near-stationary examples. At 700 ms, broader coverage improves
actual-future training-example XY error by about 60% against the fixed head,
while its maze error remains 77.38 mm. At 500 ms, the fixed head estimates
these familiar larger motions accurately but transfers poorly to the maze.

This supports a substantial transfer limitation of the representation/readout
combination, rather than an inability to fit all larger translations. It does
not isolate appearance, heading, history, motion distribution, encoder or head
as the cause: the two populations differ and the training examples were
explicitly exposed. The diagnostic is not held-out generalization evidence,
does not establish the absence of geometry, and does not justify promoting a
head or simply extending training. Both inference populations use full-float
pooled features; training used an FP16 RAM feature cache.

Reproducer: `scripts/evaluate_go2_training_translation_readout_development.py`.
Selection, presentation counts, retained-depth markers, checkpoint identities
and all 64 evaluated rows are in `training_translation_fit_diagnostic_v1/`
under this experiment's artifact root. The maze table above reuses the completed
fixed evaluation; it was not rerun or used for fitting.

## Completed removal of future-image change

A subsequent frozen-head diagnostic replaced each future image's features
with the current image's features, retaining the same current input. This
sets the head's feature-difference input exactly to zero. The 32 training
translation and 64 maze windows are unchanged, at both 500 and 700 ms. Only
96 current images were encoded on CPU; actual-future predictions are reused
from the completed evaluations. All 192 rows completed with exit zero.
No fitting, predictor inference, calibration or navigation changes occurred.

Each cell is XY RMSE in mm / yaw RMSE in degrees. "Current only" means this
information-removal input, not a separately fitted baseline.

| Population / horizon | Starting current only | Starting actual future | Fixed current only | Fixed actual future | Multi current only | Multi actual future |
|---|---:|---:|---:|---:|---:|---:|
| Training translations, 500 ms | 61.94 / 8.52 | 7.53 / 0.94 | 61.22 / 8.16 | 4.74 / 0.68 | 76.31 / 9.70 | 14.80 / 1.49 |
| Training translations, 700 ms | 90.95 / 11.61 | 28.20 / 3.44 | 90.14 / 11.28 | 28.31 / 3.43 | 105.34 / 12.69 | 11.19 / 1.77 |
| Maze translations, 500 ms | 60.08 / 7.81 | 53.20 / 5.44 | 58.81 / 7.87 | 53.90 / 6.01 | 61.45 / 8.17 | 54.30 / 5.86 |
| Maze translations, 700 ms | 88.17 / 11.02 | 78.28 / 7.47 | 86.71 / 11.09 | 79.52 / 8.01 | 89.79 / 11.25 | 77.38 / 8.16 |
| Maze turns, 500 ms | 16.48 / 9.01 | 17.29 / 5.92 | 16.26 / 9.08 | 19.14 / 5.79 | 16.68 / 9.72 | 17.66 / 6.00 |
| Maze turns, 700 ms | 17.77 / 13.64 | 19.96 / 10.35 | 17.69 / 13.71 | 20.41 / 9.69 | 17.94 / 14.25 | 20.83 / 10.13 |

Actual visual change substantially improves training-example motion decoding
for every head and horizon. This argues against a simple explanation in which
the successful training predictions ignore the future image and use only
current appearance. On maze translations, actual change still improves both
XY and yaw, but by much less. On turns it improves yaw while worsening the
small XY component. No uniform visual-change benefit or absence of spurious
translation follows.

Identical-image pairs differ from moving training pairs, and this is not a
separately optimized current-image baseline or proof excluding every shortcut.
The observation narrows the diagnosis: useful training-domain change decoding
transfers poorly, without establishing whether encoder, decoder, appearance,
heading, or other training coverage is responsible. Do not treat this as a
reason to remove current-image features, subtract identity outputs online, or
claim that the model ignores its future input. Such changes were not tested.

Reproducer: `scripts/evaluate_go2_motion_readout_future_removal_development.py`.
`future_image_removal_v1/plan.json` and `result.json` retain source/checkpoint
identities, retention receipts and every prediction under this artifact root.

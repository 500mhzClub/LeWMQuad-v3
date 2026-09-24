# Controlled motion-readout horizon training

Status: both fixed 440-update fits completed on CPU on September 22, with owner
exit code zero. Feature extraction took 9271.9 seconds; total recorded fitting
workflow time was approximately 9305.8 seconds. The fixed transfer evaluation
also completed successfully (287.8 seconds, owner exit zero). Multi-horizon
training modestly improved the primary 700-ms scores versus matched fixed-horizon
training, but remains substantially worse than command history. No new navigation
success or model promotion is claimed.

Final checkpoint SHA-256 identities:
`fixed_500ms`: `fec532504a7c82196a43faee1c5fd6c8b176b5df10174f312848a584a0036d09`;
`multi_100_800ms`: `c3dfeb98c065c3deef88458b83f2193cfecaa0de98f0a12775a90735485b117e`.

The deployed mixed-data head was trained on 500-ms image pairs but is used at
100--800 ms. Recorded 700-ms navigation failures persist when the unchanged
head receives actual future images. Identity-offset subtraction and temporal
antisymmetrization did not provide a general repair. This experiment tests
training-horizon coverage while holding architecture and optimization fixed.

| Setting | Fixed-horizon control | Multi-horizon treatment |
|---|---|---|
| Initial state | Existing mixed-data final head | Identical |
| Architecture and target normalization | Unchanged | Identical |
| Updates / batch | 440 / 64 | Identical |
| First half of each batch | Same 32 original 500-ms training examples | Identical |
| Second half | Same 32 departure contexts, future at 500 ms | Same departures, futures distributed over 100--800 ms |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001, gradient clip 1 | Identical |
| Checkpoint | Fixed final step | Fixed final step |

Inputs are the 3,518 original training examples and eight previously completed
training-role full-heading recordings. The latter supply 2,808 departure
contexts with all eight future horizons available. Each generated 500-ms target
was checked against the earlier saved target within 1e-7; timestamps and absence
of contacts were checked from the existing recordings. Each arm sees identical
old examples and new departure contexts in identical order. The multi-horizon
half-batches have exactly 1,760 presentations at each horizon. No prospective
maze image, trajectory or label is used for fitting.

The encoder and predictor remain frozen. CPU cores 4--7 perform feature
extraction and head training, with four Torch threads and one OpenBLAS thread;
the existing navigation owner uses its separate CPU/GPU allocation. There are
7,068 image paths and a 2,779,250,688-byte pooled FP16 RAM feature cache. Encoding
deduplicates byte-identical images. No feature tensors or depth arrays are saved.
Only small metadata, loss histories and two final checkpoints are retained.

This is a readout-training intervention, not a JEPA-objective comparison. It
changes the future-image interval and corresponding motion magnitudes together;
it does not isolate which of those explains any effect. One seed is used.
Training loss is not transfer evidence. After fitting, evaluate both fixed final
checkpoints on retained exposed development diagnostics, with the starting head
and command-history control, before considering any separate navigation study.
There is no automatic promotion into the running prospective cohort, whose
sources and model identities remain fixed.

Runner: `scripts/train_go2_multihorizon_motion_readout_development.py`.
Artifact root:
`.generated/navigation_development_artifacts_v1/go2_multihorizon_motion_readout_v1_attempt_001/`.
`plan.json` retains the initial head, source and training-input identities;
`samples.json`, `frame_paths.json` and `schedule.json` retain the exact fitting
population and presentations. `process.json` identifies the live owner.
At launch the tool session is 2604. The separate navigation coordinator remains
session 64661; this new fitting job does not resume, restart or modify it.

## Transfer population fixed before fitting results

`scripts/evaluate_go2_multihorizon_motion_readout_development.py --prepare`
completed successfully. `transfer_plan.json` fixes the same 35 common executed
windows from the earlier exposed pilot, with all 300/500/700-ms observations.
It compares original, starting mixed-data, fixed-500-ms and multi-horizon heads
on actual-future, action-predicted and action-blind features. The primary
endpoint is 700 ms; 500 ms, 300 ms and the 300--700-ms interval remain explicit
secondary readouts. No checkpoints are chosen from these scores. The evaluator
reuses the earlier CPU comparison and checks reproduction of both original
online outputs and the saved starting mixed-data predictions.

The preparation also completed the command-history reference on exactly those
105 saved rows, using the selected candidate's recorded forecast and the same
physical targets. This required no new inference or simulation. Together with
the previously completed starting-head scores, the baseline is:

| Horizon / interval | Starting head, action future: XY / yaw RMSE | Starting head, actual future: XY / yaw RMSE | Command history: XY / yaw RMSE | Zero motion: XY / yaw RMSE |
|---|---:|---:|---:|---:|
| 300 ms | 16.02 mm / 2.59 deg | 16.19 mm / 2.40 deg | 3.21 mm / 0.29 deg | 4.58 mm / 7.06 deg |
| 500 ms | 18.83 mm / 5.56 deg | 21.15 mm / 5.45 deg | 4.21 mm / 0.57 deg | 6.76 mm / 11.64 deg |
| 700 ms | 19.85 mm / 8.52 deg | 21.02 mm / 9.37 deg | 5.07 mm / 0.69 deg | 9.03 mm / 16.26 deg |
| 300--700 ms | 10.07 mm / 6.67 deg | 11.01 mm / 8.00 deg | 3.34 mm / 0.60 deg | 5.46 mm / 9.64 deg |

These predominantly turning/holding windows are overlapping samples from one
exposed trajectory. Command history is the stronger baseline here, while zero
motion's low XY error reflects little translation. Improving on the starting
head alone will therefore not establish adequate motion forecasting or a JEPA
advantage. `command_history_reference.json` retains the complete new baseline
rows, and the transfer plan binds the earlier result and evaluator identities.
The evaluator subsequently ran without `--prepare` on CPU group 4--7 and
completed. It did not use the navigation GPU.

## Completed fixed transfer comparison

Both fixed final heads were evaluated on all 35 preselected common windows.
The original and starting-head outputs reproduced within the existing tolerance;
`pilot_evaluation/comparison.json` records completion and all control scores.
Values below are XY RMSE in millimetres / yaw RMSE in degrees. The 700-ms
endpoint was primary; other columns are the fixed secondary endpoints.

| Readout / future input | 300 ms | 500 ms | 700 ms | 300--700 ms interval |
|---|---:|---:|---:|---:|
| zero_motion | 4.58 / 7.06 | 6.76 / 11.64 | 9.03 / 16.26 | 5.46 / 9.64 |
| original_observed_future | 35.24 / 4.74 | 44.30 / 8.32 | 42.54 / 13.14 | 15.80 / 9.20 |
| original_anchored_observed_future | 28.26 / 6.09 | 37.21 / 9.92 | 36.26 / 14.81 | 15.80 / 9.20 |
| original_action | 37.01 / 4.34 | 37.16 / 7.23 | 37.64 / 10.00 | 14.62 / 6.46 |
| original_anchored_action | 29.97 / 6.04 | 30.75 / 8.96 | 31.65 / 11.76 | 14.62 / 6.46 |
| original_no_future_action | 30.29 / 4.51 | 29.98 / 7.38 | 29.02 / 10.26 | 13.66 / 7.10 |
| original_anchored_no_future_action | 23.32 / 5.89 | 23.99 / 8.83 | 24.63 / 11.67 | 13.66 / 7.10 |
| starting_mixed_observed_future | 16.19 / 2.40 | 21.15 / 5.45 | 21.02 / 9.37 | 11.01 / 8.00 |
| starting_mixed_action | 16.02 / 2.59 | 18.83 / 5.56 | 19.85 / 8.52 | 10.07 / 6.67 |
| starting_mixed_no_future_action | 16.64 / 4.06 | 17.86 / 6.95 | 18.94 / 9.93 | 10.47 / 7.16 |
| fixed_500ms_observed_future | 16.93 / 2.88 | 22.01 / 5.94 | 22.28 / 9.93 | 10.76 / 8.04 |
| fixed_500ms_action | 15.57 / 2.49 | 18.25 / 5.62 | 19.20 / 8.49 | 9.79 / 6.66 |
| fixed_500ms_no_future_action | 16.11 / 3.71 | 17.64 / 6.74 | 18.83 / 9.68 | 10.27 / 7.19 |
| multi_100_800ms_observed_future | 14.74 / 3.10 | 19.57 / 5.58 | 20.73 / 8.53 | 11.24 / 6.28 |
| multi_100_800ms_action | 14.68 / 3.27 | 17.39 / 5.71 | 18.90 / 7.83 | 9.57 / 5.44 |
| multi_100_800ms_no_future_action | 14.14 / 4.75 | 15.95 / 7.60 | 17.02 / 10.34 | 8.58 / 6.92 |
| command_history | 3.21 / 0.29 | 4.21 / 0.57 | 5.07 / 0.69 | 3.34 / 0.60 |

Compared with the matched fixed-500-ms control, the multi-horizon action head
improves primary XY RMSE from 19.20 to 18.90 mm and yaw from 8.49 to 7.83
degrees. Interval yaw improves from 6.66 to 5.44 degrees, but 300-ms yaw
worsens from 2.49 to 3.27 degrees. Actual-future 700-ms yaw also improves
(9.93 to 8.53 degrees), while retaining substantial error. Command history
remains much stronger at 700 ms (5.07 mm / 0.69 degrees). Action-blind XY
error is lower than action-conditioned XY error for the multi-horizon head,
while its yaw error is higher. There is no uniform action-conditioned advantage.

This result supports a limited effect of training horizon on the motion head,
not a sufficient motion-readout repair, a JEPA-objective advantage, or a reason
to promote this head into the fixed prospective cohort. The separate magnitude
diagnosis below identifies a remaining translation-coverage gap. A subsequent
matched readout experiment can extend horizons on the existing original
training trajectories, whose longer translations are available, while preserving
these completed results and keeping prospective maze data out of fitting.

That follow-up has now been prepared and launched with matched fixed-500-ms
and all-motion multi-horizon arms. Its fixed training and translation/turn
evaluation populations are recorded in
`go2_all_motion_horizon_readout_2026-09-22.md`. No new outcome is claimed yet.

## Motion-magnitude coverage diagnosis performed while fitting was active

A descriptive check of the saved training targets and completed layout-00
action trajectory identifies a limitation of this fixed experiment. No source,
schedule, checkpoint, transfer population or live navigation setting changed.

| Population | Samples | Median XY displacement | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Original training, 500 ms | 3518 | 8.07 mm | 92.37 mm | 105.21 mm |
| New full-heading training, 500 ms | 2808 | 6.51 mm | 17.56 mm | 25.85 mm |
| New full-heading training, 700 ms | 2808 | 8.97 mm | 20.61 mm | 25.76 mm |
| New full-heading training, 800 ms | 2808 | 10.00 mm | 21.51 mm | 27.95 mm |
| Maze-00 action arm, executed 700-ms translation windows | 116 | 103.43 mm | 133.74 mm | 144.27 mm |

Of the 116 translation windows, 52 exceed the original training maximum and
114 exceed the new 800-ms training maximum. The new data extends absolute yaw
targets from a 500-ms maximum of 13.23 degrees to 21.15 degrees at 800 ms, but
does not supply comparable long translations. Thus this continuation can test
broader temporal/rotation coverage; even a positive result on the predominantly
turning pilot would leave translation coverage unresolved. Conversely, failure
would not rule out a readout trained on suitable translation sequences.

These are empirical marginal displacement ranges, not a proof of feature,
scene or joint-distribution coverage, nor a causal attribution of navigation
failure. Windows overlap. The saved `motion_magnitude_coverage_diagnostic.json`
in the experiment root records the input paths, quantile method, all summaries
and counts. A later translation intervention should first use suitable existing
training-role trajectories if available; prospective maze recordings remain
evaluation-only. Finish the current matched experiment before choosing that
intervention.

The existing original training recordings already offer a possible remedy,
without collecting new simulations. A metadata/retained-path check restricted
to the original 3,518 training contexts found:

| Horizon | Valid zero-contact targets with both RGB paths | XY displacement p95 / maximum | Targets exceeding original 500-ms maximum |
|---|---:|---:|---:|
| 500 ms | 3518 | 92.37 / 105.21 mm | 0 |
| 600 ms | 3398 | 108.78 / 125.29 mm | 180 |
| 700 ms | 3278 | 121.95 / 144.78 mm | 650 |
| 800 ms | 3158 | 136.48 / 163.95 mm | 822 |

There are 3,158 departure contexts with valid targets across all eight horizons.
Invalid padded end-of-recording targets account for the exclusions; no valid
pair was missing its RGB paths and no contact target was admitted. At 700 ms,
the available maximum exceeds the maze-00 translation maximum of 144.27 mm.
This establishes available target range, not feature coverage or transfer
performance. No images were decoded, no fit was launched, and the current
study remains unchanged. `original_population_multihorizon_availability.json`
retains the exact training-source paths, counts and method. If current results
justify a follow-up, extending horizons on these original recordings can test
the translation gap while retaining a matched 500-ms control.

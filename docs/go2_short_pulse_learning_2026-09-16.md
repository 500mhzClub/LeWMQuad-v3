# Short-pulse training coverage

The frozen command-history predictor was evaluated on all 15,801 matched
700-ms execution windows from the completed 36-run development pilot. XY RMSE
was 7.897 mm versus nominal integration 11.625 mm, corrected neural prediction
7.183 mm, and the pose-based fitted control 6.500 mm. Yaw RMSE was 0.645 degrees
versus nominal 1.620 degrees. However, on 346 terminal translation pulses its
XY RMSE was 20.961 mm, worse than nominal 15.608 mm and pose-based 9.663 mm.
This does not support substituting it into navigation as a complete predictor.
The readout took 13.38 seconds and made no model changes or new simulations.
Artifact: `go2_command_history_executed_windows_v1_attempt_001`.

The next experiment collects the missing 100-ms translation excitation. Keep
the existing four geometry clusters and role split: 00/01 train, 02/03 exposed
development transfer. Each cluster has nine fresh episodes: three preceding
actions (hold, left turn, right turn) crossed with three pulse actions (forward,
left arc, right arc). Each episode has three quiet ticks, ten preceding-action
ticks, three zero ticks, one pulse tick, and eight zero ticks. There are 25
command ticks, 26 camera frames and 2,000 physical samples including settling.
All 36 assignments are fixed before collection. Preserve stops and failures.
These are action-coverage experiments, not new independent maze evaluations.

Reuse the existing geometry-family native session, frozen gait, public camera,
body and command recording, and native contact/speed stops. Physics pauses
during acquisition as in the original training collection; this is not a
real-time navigation experiment. Native poses remain target-only. Derive all
frames 8..17, including incomplete or censored windows explicitly, using the
existing observation-horizon target function. The central departure at frame
13 has exactly the planner's three committed zero ticks followed by one
translation tick and four zeros at its eight prediction horizons.

Four native collection workers reuse the previously measured compatible
collection mode (3.46x speedup, exact serial/parallel physics and pixels). This
does not justify parallel paced navigation simulations. Estimated collection
size is about 1.7 GB from the original per-frame storage; start with at least
3 GiB free and retain a 1 GiB reserve. These are development working-space
limits, not the obsolete collector's 40 GiB reserve. Fresh worker processes
own separate episode paths. Retain RGB/body and target records for matching
JEPA, direct, supervised-rollout and command-history fits. Do not use the 36
navigation trajectories as training data. No navigation model is promoted
by collection or offline prediction scores.

Collection completed: all 36 episodes, 936 camera frames, 72,000 physical
samples and 360 available contexts, with no physical or acquisition stops.
Wall time was 168.28 seconds. All 90 matched preceding-action/pulse/frame
groups have exactly identical native motion targets across their four
geometries. These are visual geometry variations of nine motion sequences,
not independent dynamical trials or new physical transfer evidence.

The fixed successor schedule has 4,694 contexts and the original 7,200 draws:
900 repeated original-context draws are replaced by five draws for each of
180 training pulse contexts. Every original context remains represented.
The first proposed schedule restricted replacement to repeated switch draws;
it rejected that restriction before writing output because too few repeats
were available. The final schedule samples replacement positions across both
original sources. Exactly 175 of 600 family batches remain unchanged; original
per-trial weights change. Data coverage and allocation therefore change
together. All models share this final schedule and the same 1,200-update budget.

The command-history ridge successor is complete. On the original transfer
population, 700-ms XY RMSE regresses from 5.890 to 6.144 mm and yaw from 0.838
to 0.931 degrees. On the 180 new pulse-transfer contexts, XY improves from
8.595 to 6.552 mm and yaw from 0.998 to 0.785 degrees. On the 18 frame-13
planner departures, XY improves from 9.354 to 7.269 mm. Native motion targets
are identical across geometry repetitions as noted above.

More relevantly, on all 15,801 saved navigation execution windows, new XY
RMSE is 7.708 mm versus the predecessor's 7.897 mm; it improves in 26/36 runs.
Yaw regresses from 0.645 to 0.684 degrees (improves in only 3/36 runs).
For all 346 actual translation pulses, XY improves from 20.961 to 18.371 mm,
still worse than nominal integration 15.608 mm and the pose-based fit 9.663 mm.
This is an exposed-trajectory prediction result, not alternative navigation.
The new fit is not promoted. Artifacts:
`go2_short_pulse_command_control_v1_attempt_001` and
`go2_short_pulse_command_executed_windows_v1_attempt_001`.

The command-history replay feature encoding matched the original fit predictor
on twelve full-tape training contexts to within 5.6e-17; changing the eighth
command did not change its 700-ms forecast. An initial check mistakenly included
a context with only five future commands and rejected it; all recorded pilot
plans used here have eight commands. No readout outcomes changed.

Three matched neural residual fits (JEPA, direct, supervised rollout) completed
in 302.16 seconds including 68.01 seconds loading. Peak RSS was 10.08 GB.
All three retain the predecessor's initial weight hash and 1,200-update budget.
Transfer and both executed readouts are complete. Evaluations cover the original
924 transfer contexts plus the new 180 pulse contexts, reported separately.
Executed readouts compare the previously fixed first pair (775 windows) and all
346 translation-pulse windows across the full 36-run pilot. These remain
overlapping windows from two exposed mazes; the tracking failure is retained.

700-ms XY RMSE, predecessor to pulse-augmented, in millimetres:

| Method | Original transfer (870 valid) | New pulse transfer (180) | First maze pair (775) | All navigation pulses (346) |
| --- | ---: | ---: | ---: | ---: |
| JEPA | 21.713 to 12.846 | 15.414 to 10.773 | 22.728 to 10.157 | 12.968 to 18.378 |
| Direct | 15.280 to 13.352 | 17.886 to 6.873 | 13.460 to 10.147 | 24.463 to 13.981 |
| Supervised rollout | 15.872 to 13.369 | 13.004 to 8.276 | 11.175 to 8.831 | 16.415 to 12.898 |

The saved pose-based comparator is 6.564 mm on the first pair and 9.663 mm
on all navigation pulses. Nominal command integration is 11.759 and 15.608 mm
respectively. Thus the direct and supervised successors beat nominal integration
on the pulse population but do not beat the pose-based comparator. JEPA's
first-pair XY gain does not carry over to the broader pulse population.

Yaw also prevents interpreting the JEPA XY gain as uniform improvement. Its
first-pair yaw RMSE worsens from 1.337 to 2.036 degrees and all-pulse yaw from
0.754 to 3.040 degrees. Direct all-pulse yaw improves from 0.785 to 0.668
degrees; supervised all-pulse yaw worsens from 0.668 to 1.150 degrees.

Neural artifacts:

- `go2_short_pulse_residual_matched_fits_v1_attempt_001`
- `go2_short_pulse_residual_transfer_comparison_v1_attempt_001`
- `go2_short_pulse_neural_executed_first_pair_v1_attempt_001`
- `go2_short_pulse_neural_executed_all_pulses_v1_attempt_001`

No fit is promoted and no new online navigation outcome was collected. This
turn made scientific progress: a frozen simpler predictor was tested on actual
execution, the missing excitation was collected, matched models were fitted,
and both improvements and regressions were measured. All jobs are terminal.
The next useful step is a prospective matched navigation comparison using the
fixed predictors, with the stronger simple comparator retained. Avoid another
unmotivated architecture/seed search or replacing measured outcomes with offline
scores. New layout units, realistic sensing/timing and hardware evidence remain
requirements of the active goal. Free storage after this collection is about
3.5 GiB: sufficient for these completed fits/readouts, but below the existing
4-GiB native-navigation launch reserve. Retire eligible completed depth before
the next recording; do not delete active inputs or failure evidence.

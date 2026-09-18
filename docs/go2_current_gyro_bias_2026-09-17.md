# Current tracker sensitivity to persistent gyro bias

**Complete: all six replays accepted every recorded frame. Both signed biases
produced substantial undetected raw-pose drift. No new navigation was run.**

| Journey | Yaw-rate bias (rad/s) | Accepted frames | Maximum XY error (mm) | Maximum orientation error (degrees) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 2243/2243 | 6.75 | 0.060 |
| 1 | +0.001 | 2243/2243 | 413.60 | 12.833 |
| 1 | -0.001 | 2243/2243 | 388.83 | 12.823 |
| 2 | 0 | 1543/1543 | 8.14 | 0.061 |
| 2 | +0.001 | 1543/1543 | 279.32 | 8.823 |
| 2 | -0.001 | 1543/1543 | 290.91 | 8.819 |

Journey durations are 224.2/154.2 sensor seconds. Since every case completed,
the common-prefix accuracy comparisons cover each complete recorded journey.
Nominal controls reproduce all 3,786 raw positions and rotations exactly.
The signed perturbations therefore isolate a current-tracker sensitivity on
these fixed inputs. No live noisy-gyro navigation outcome follows: downstream
floor registration, mapping and command selection were not reexecuted.

This strengthens the older shared-gyro-dependence finding with the current
tracker and full recent journeys. Gyro-conditioned fitting admits locally
consistent motion while the integrated global heading drifts. Passing the
existing raw-pose checks does not establish bounded global pose error. The
next sensing work should evaluate independent visual rotation information or
gyro-bias estimation, retaining nominal accuracy and failure outcomes, before
closed-loop testing with perturbed gyro. Do not tune admission thresholds to
accept more of these already fully accepted trajectories. No estimator or
world-model change has been made by this sensitivity study.

Complete result: `docs/go2_current_gyro_bias_result_2026-09-17.json`.
Figure: `docs/go2_current_gyro_bias_2026-09-17.png` (also SVG).
The generated four-panel figure was visually inspected: both bias signs show
nearly linear orientation drift and substantial path-dependent position error;
the nominal traces remain small. No rejection markers appear because all six
cases accepted their complete recordings.
All six owners and evaluations exited zero. Historical progress notes below
do not indicate remaining active jobs.

Six fixed full-journey raw-tracker replays: nominal and yaw-rate biases of
+0.001 and -0.001 rad/s on each of the two retained persistent-return successes
from the completed return-routing-memory study. These are the same signed
bias sizes as the older September 6 geometric-estimator study, approximately
0.0573 degrees/second. They are synthetic sensitivity conditions, not measured
Go2 IMU specifications or a calibrated sensor distribution.

The current sparse-corner tracker and its admission rules remain unchanged.
The existing `ErrorMember` perturbation changes both slow and fast gyro
histories consistently, preserving repeated samples, timestamps and all other
channels. Original 2-mm depth noise is reconstructed with recorded digest
checks. No images or depth archives are duplicated. Nominal controls must
match every recorded raw position and rotation exactly.

All predictions are saved before physical truth is read. Report full-sequence
survival, rejection reason, accepted-prefix position/orientation errors and
errors over the common prefix. Raw tracking survival does not establish floor
registration, mapping, planning, navigation success or hardware readiness.
There is no new command execution and no model or estimator fitting.

Order is nominal maze 1/2, positive bias maze 1/2, negative bias maze 1/2.
Each pair runs independently on CPU groups 0–7/16–23 and 8–15/24–31, one
numerical-library thread per process. There is no native simulation running.
Before launch the CPU was 96% idle, 72 GiB RAM available, both GPUs idle,
and the artifact volume had 4.5 GiB free. Resident swap was 3 GiB with no
observed active swapping. Existing full recordings remain pinned for replay.

Plan: `docs/go2_current_gyro_bias_plan_2026-09-17.json`.
Runner: `scripts/replay_go2_current_gyro_bias_development.py`.
Each source journey receives three exclusive `current_gyro_bias_*_v1`
subdirectories. All six results, including failures, enter the final readout.

## Nominal controls complete

Assignments 1/2 exited zero. All 2,243 and 1,543 poses respectively matched
the original raw positions and rotations exactly. Maximum raw XY errors were
6.75/8.14 mm; maximum orientation errors were 0.0600/0.0611 degrees. Replay
wall times were 242.99/165.47 seconds. Observed worker RSS was approximately
1.4 GiB each, with one CPU core occupied per worker. The biased cases follow
the frozen assignment order; nominal accuracy does not establish their outcome.

## Positive-bias cases complete

Assignments 3/4 accepted every frame, yet maximum XY error reached
413.60/279.32 mm and maximum orientation error reached 12.83/8.82 degrees.
These are complete recorded-trajectory results, not just failing prefixes.
The gyro-conditioned raw tracker did not reject the accumulated heading drift.
Negative-bias cases are running unchanged; all assigned outcomes remain required.

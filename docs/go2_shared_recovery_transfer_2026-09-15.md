# Shared recovery on two fresh development mazes

The signed reactive recovery completed a goal-and-return mission on the
exposed development maze. Test transfer without selecting new layouts or
training seeds from navigation outcomes. Two new layouts are constructed
with seed 2026091555 and excluded against the explicit 78-layout development
registry. The fixed inventory is
`docs/go2_shared_recovery_transfer_layout_inventory_2026-09-15.json`, SHA-256
`ff645dbfc2af51d7c0a8b805c6c6fa107a81ccafd11dfaf132ee9006c6d40ff8`.
These are new development mazes in the existing family, not a sealed test.

Use the existing first training seed 2026091001 for JEPA, direct prediction
and supervised rollout, with their already frozen model/correction bindings.
The fitted-motion and reactive controls retain their existing definitions.
All five controllers share the combined floor-statistics/deferred-copy tracker,
transported-normal floor registration, gyro-conditioned auxiliary obstacles,
signed actual-translation-veto view recovery, depth noise and original command
guards. Do not train or tune any model for these layouts.

Run the ten assignments sequentially. Layout 0 order: reactive, JEPA, direct,
supervised, fitted motion. Layout 1 uses the reverse order. All native jobs
use the established CPU group for their layout; no estimator replay or second
native job runs concurrently. Earlier concurrent tracking failures and lighter
workload successes make serial execution the less confounded comparison.
Typical successful runs take roughly four wall-clock minutes; budget-exhausting
runs have taken about twelve. Evaluate each completed assignment before the
next, preserving every failure.

Keep the 4,800-tick budget, 2-mm synthetic depth noise, ideal gyro, existing
camera mounts, 300-ms planning delay, 400-ms normal command window and current
near-arrival translation pulses. Confirm physical goal and return arrivals,
one-second quiet dwells, contacts, pose errors, selected-controller treatment,
timing failures and actual recovery exposure. Aggregate every fixed outcome.

This compares complete controllers with shared local perception/recovery.
Reactive selection still differs in predictive clearance and stopping
projection, so it does not isolate future-outcome ranking alone. It can test
whether the learned predictors outperform the stronger reactive and fitted
baselines, and compare the three training methods at one fixed seed. It does
not by itself establish memory benefit, calibrated real sensors, hardware
readiness or broad success rates.

Launcher: `scripts/run_go2_shared_recovery_transfer_development.py`.
Evaluator: `scripts/evaluate_go2_shared_recovery_transfer_development.py`.
Roots: `go2_shared_recovery_transfer_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.
Keep four GiB free before each recording. Completed tracking-control success
depth is retired under the existing retention policy; results and all failures
remain. Current new-layout recordings stay full through their five-arm
comparison and diagnosis, after which redundant success depth may be retired.

## Results in fixed assignment order

All ten assignments are complete and evaluated. The first reactive run on
new layout 0 verified a round trip: goal frame 1,472, home frame 2,236, both
with physical-radius and one-second quiet-dwell checks passed. There were
2,238 poses, zero contacts and maximum pose error 7.929 mm. The mission took
223.92 simulated seconds; 539 of 548 plans were on time. The owner completed
archival and exited 0 after 5:38.03, without swapping. Actual-treatment checks
confirmed no predictive outcomes were used.

It exercised one actual translation-veto recovery: a right arc was vetoed at
77.80 seconds, followed by a -45-degree target and a right-turn recovery plan.
There were 90 applied right-turn intervals during that recovery. Translation
resumed at 80.60 seconds and continued for 4,626 applied translation intervals
after recovery, with no further translation-veto recovery before both arrivals.
There were no auxiliary-only guard requests, so this remains no evidence of
degraded-perception recovery. Event details are in this run's
`signed_veto_view_recovery_diagnostic_v1.json`.

This is the first successful transfer of the shared setup to a new layout.
It does not by itself establish a comparative learned-model result.

JEPA seed 2026091001 also verified the layout-0 round trip: goal frame 2,368,
home frame 3,007, both physical-radius and one-second quiet-dwell checks passed.
There were 3,009 poses, zero contacts and maximum pose error 5.682 mm. The
actual learned XY/yaw forecasts, predictive selection and frozen model/fit
assignment were verified. It took 301.34 simulated seconds, with 615 of 704
selected plans on time. Owner archival completed and exited 0 after 7:10.09,
without swapping. Its full recording remains.

JEPA was slower than reactive in this single pair and recorded more late plans
(89 versus 9). Recorded stage medians were 58 ms tracking, 30 ms registration,
38 ms obstacles, 78 ms mapping and 78 ms planning; maximum planning service
was 934 ms and maximum planning completion age 1,452 ms. There were 23 skipped
planning records because the map was newer than the planning observation,
25 mission-hold records and no pipeline faults. The result recorded maximum
simulator lag 12,289 ms. These are measured-simulation timing observations;
the cause of the timing variation is not isolated and no hard real-time claim
is made. Details: `recorded_pipeline_timing_diagnostic_v1.json` in the JEPA run.

Direct prediction seed 2026091001 verified the layout-0 round trip: goal frame
1,714, home frame 2,356, both physical-radius and quiet-dwell checks passed.
There were 2,359 poses, zero contacts and maximum pose error 12.327 mm. Learned
XY/yaw forecasts, predictive use and the frozen model/correction assignment
were verified. Duration was 236.16 simulated seconds, with 435 of 580 selected
plans on time (145 late). The owner completed archival and exited 0 after
5:39.01, without swapping. There were no pipeline faults, two map-newer-than-
observation planning skips and seven mission-hold records. Planning's recorded
median service was 84 ms, maximum service 698 ms and maximum completion age
1,064 ms; maximum recorded simulator lag was 9,572 ms. The timing diagnostic
uses the same scope as JEPA and does not isolate a causal explanation.

| Layout 0 condition | Verified round trip | Contacts | Simulated seconds | On-time / selected plans |
| --- | --- | ---: | ---: | ---: |
| Reactive | Yes | 0 | 223.92 | 539 / 548 |
| JEPA | Yes | 0 | 301.34 | 615 / 704 |
| Direct prediction | Yes | 0 | 236.16 | 435 / 580 |
| Supervised prediction | Yes | 0 | 245.58 | 518 / 584 |
| Fitted motion | Yes | 0 | 235.86 | 423 / 574 |

Supervised prediction seed 2026091001 verified both arrivals: goal frame 1,636,
home frame 2,452, with the physical-radius and one-second quiet-dwell checks
passed. It had 2,454 poses, zero contacts and maximum pose error 12.281 mm.
Actual learned XY/yaw, predictive use and frozen model/correction binding were
verified. The owner completed archival and exited 0 after 5:50.83, with no
swapping. Its result recorded 66 late plans and maximum simulator lag 6,351 ms;
no hard wall-clock qualification is claimed.

Fitted motion verified the layout-0 round trip: goal frame 1,712, home frame
2,351, both physical-radius and one-second quiet-dwell checks passed. It had
2,354 poses, zero contacts and maximum pose error 10.006 mm. Actual fitted XY
and command-integrated yaw, predictive use and the frozen model/correction
binding were verified. The owner completed archival and exited 0 after
5:42.79, without swapping. There were 151 late plans; maximum recorded simulator
lag was 14,540 ms. All five layout-0 recordings remain full.

After all five layout-0 evaluations, run
`scripts/compare_go2_shared_recovery_transfer_development.py --layout-index 0`
and the existing native comparison plotter. The comparison includes every fixed
outcome, actual controller bindings, shared settings, common source identities,
timing and behavior metrics. Its output root is
`go2_shared_recovery_transfer_comparison_layout00_v1_attempt_001`; layout 1 uses
the corresponding `layout01` root when complete.

The layout-0 comparison and PNG/SVG figures are complete and visually inspected.
All shared settings and 157 common runtime-source identities match across the
five arms. All five independently verified the round trip with zero contacts.
Reactive was fastest in this single set; fitted motion and direct prediction
had similar durations. Path lengths were 23.95 m reactive, 27.12 m JEPA, 22.79 m
direct, 26.52 m supervised and 23.06 m fitted. Command-zero durations were
15.00, 69.10, 65.20, 53.86 and 71.28 seconds respectively; zero command is not
a claim of physical stationarity. Timing and exploration trajectories both
differ, so the mission-time differences do not isolate forecast quality.

On each controller's own matched executed windows, corrected XY endpoint RMSE
was 6.23 mm JEPA, 7.58 mm direct, 7.96 mm supervised and 7.28 mm fitted. These
populations differ and overlap internally; they are not independent matched
samples across controllers. Within each learned controller's same windows,
learned yaw endpoint RMSE was higher than the saved command-integrated
alternative: 3.58 versus 1.85 degrees JEPA, 3.04 versus 1.47 direct, and 3.21
versus 1.50 supervised. Fitted motion's applied command-yaw RMSE was 1.38
degrees. These are prediction diagnostics, not hypothetical navigation results.

Fitted motion on layout 1 independently verified the round trip: goal frame
1,017, home frame 1,580, both physical-radius and one-second quiet-dwell checks
passed. There were 1,582 poses, zero contacts and maximum pose error 8.121 mm.
Actual fitted XY, command-integrated yaw, predictive selection and the frozen
model/correction binding were verified. The mission took 158.36 simulated
seconds, with 377 of 388 selected plans on time. Maximum recorded simulator
lag was 553 ms. The owner completed archival and exited 0 after 4:20.20,
without swapping. This is a full retained recording.

| Layout 1 condition | Verified round trip | Contacts | Simulated seconds | On-time / selected plans |
| --- | --- | ---: | ---: | ---: |
| Fitted motion | Yes | 0 | 158.36 | 377 / 388 |
| Supervised prediction | Yes | 0 | 167.28 | 398 / 405 |
| Direct prediction | Yes | 0 | 170.36 | 395 / 417 |
| JEPA | Yes | 0 | 174.56 | 405 / 417 |
| Reactive | No arrival; budget exhausted | 0 | 480.88 | 1176 / 1200 |

Supervised prediction on layout 1 independently verified both arrivals: goal
frame 1,157, home frame 1,662, with physical-radius and one-second quiet-dwell
checks passed. It had 1,665 poses, zero contacts and maximum pose error 6.799 mm.
Actual learned XY/yaw forecasts, predictive selection and the frozen model/fit
binding were verified. The owner completed archival and exited 0 after
4:15.99, without swapping. There were seven late plans and maximum recorded
simulator lag 570 ms. Its full recording remains.

Direct prediction on layout 1 independently verified both arrivals: goal
frame 1,140, home frame 1,700, with physical-radius and one-second quiet-dwell
checks passed. It had 1,702 poses, zero contacts and maximum pose error 5.800 mm.
Actual learned XY/yaw, predictive use and the frozen model/fit binding were
verified. The owner completed archival and exited 0 after 4:20.60, without
swapping. There were 22 late plans and maximum recorded simulator lag 575 ms.
Its full recording remains.

JEPA on layout 1 independently verified both arrivals: goal frame 1,196, home
frame 1,742, with physical-radius and one-second quiet-dwell checks passed.
It had 1,744 poses, zero contacts and maximum pose error 7.825 mm. Actual learned
XY/yaw, predictive use and the frozen model/fit binding were verified. The owner
completed archival and exited 0 after 4:27.27, without swapping. There were
12 late plans and maximum recorded simulator lag 566 ms. Its full recording
remains. All three learned controllers have now verified both new mazes at the
fixed training seed; the fitted control also passed both.

The final reactive layout-1 assignment exhausted its budget without either
arrival. It had 4,805 accepted poses, zero contacts and maximum pose error
4.800 mm. Actual-treatment evaluation confirmed no predictive outcomes were
used. The owner completed archival and exited 0 after 11:28.03, without
swapping. This is a mission failure even though no runtime exception occurred.
Its full failure recording is retained.

The controller selected 1,065 consecutive hold plans from frame 544 through
4,800 because stored-map clearance was below 0.45 m, making every candidate,
including turns, ineligible. At the first blocked plan, stored clearance was
0.442937 m versus evaluator-only native body-centre-to-wall clearance 0.459309 m.
At frame 4,800 those values were 0.428780 and 0.443823 m. Translation requests
continued through 56.18 seconds after the first blocked planning observation
at 55.90 seconds. There were no actual translation-veto recovery events and no
auxiliary-only guard requests. Thus the signed-view intervention was not
exercised in this failure. The map discrepancy is not causally isolated, and
the final native clearance is itself below the nominal disk radius; do not
declare this merely a false map obstacle or assume a turn-only escape is safe.
Details: `reactive_stored_clearance_deadlock_diagnostic_v1.json` in the failure.

## Completed cohort and scientific interpretation

The layout-1 comparison and PNG/SVG plots are complete and visually inspected.
All shared settings and 157 common runtime-source identities match across all
ten assignments. The aggregate is
`go2_shared_recovery_transfer_complete_comparison_v1_attempt_001/result.json`.
No configuration was tuned between assignments.

| Controller | Goal and return | Contacts |
| --- | ---: | ---: |
| JEPA | 2 / 2 | 0 |
| Direct prediction | 2 / 2 | 0 |
| Supervised prediction | 2 / 2 | 0 |
| Fitted motion | 2 / 2 | 0 |
| Reactive | 1 / 2 | 0 |

All six learned-controller missions used verified action-conditioned forecasts
to select actions and completed independent physical arrivals. This demonstrates
the engineered learned-world-model navigation architecture in the stated
nominal simulation domain. It does not establish learning or JEPA superiority:
the fitted predictor also passed both layouts, and the reactive comparison
differs in predictive clearance and other recovery rules. The effective layout
sample is two, with one training seed, not ten independent mazes. Prior failures
remain visible in their original cohorts.

Next, isolate online predictive planning within the same frozen controller,
predictor, costs and action interface, rather than infer its contribution from
the differently structured reactive controller. Retain the sensor-based
dispatch guards and the observed unsafe-clearance failure. RGB dependence,
calibrated sensing and full-loop timing/hardware validation remain separate
unresolved requirements. No hard real-time, general terrain, sealed benchmark
or hardware readiness claim is made.

Retention after the completed comparison: redundant successful layout-1 fitted,
direct and JEPA depth was retired (10,056 leaves; 2,977,984,512 allocated bytes).
All outcomes, forecast evaluations, comparisons and non-depth records remain
unchanged. All five layout-0 full controller references, the layout-1 supervised
success and the complete reactive layout-1 failure remain full. Exact historical
depth replay of the three retired successes is unavailable. See
`docs/go2_development_artifact_retention_2026-09-14.md` for the inventory.

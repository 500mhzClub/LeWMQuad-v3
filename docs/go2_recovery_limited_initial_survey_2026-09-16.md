# Recovery-limited initial survey: exposed development experiment

**Completed:** the combined prompt-hold and recovery-limited survey controller
produced physically verified round trips with both JEPA (293.28 s) and
supervised rollout (296.86 s), zero contacts. The preceding prompt-hold-only
JEPA trial still failed tracking. This is a repair result on one exposed maze,
not an independent reliability result or a JEPA advantage. The original
20-run comparison remains unchanged.

The original five-arm comparison and the subsequent prompt-command-cancellation
JEPA trial all lost tracking during the startup panorama on replication layout
0, before any translation. Prompt cancellation was exercised six times but did
not prevent failure. The mandatory survey repeatedly requests a heading that
causes low visual support; recovery returns toward a supported view and the
same survey request resumes.

Test one further JEPA mission on this exposed layout. Keep prompt cancellation
enabled and change only the startup survey policy: when its own camera-time
weak-view recovery first interrupts the sweep, defer the remaining startup
views. Preserve `complete=false`, the genuinely completed view stages and an
explicit deferral reason. Do not mark the outstanding views observed or infer
free space from them. Existing outer visual recovery still takes precedence.
Once recovery clears, use the same observed-floor/frontier routing, forecasts,
clearance and dispatch checks as before. Later frontier views remain unchanged.

This tests whether the unconditional initial full sweep is the immediate cause
of repeated unsafe-for-tracking attempts. It does not claim that a partial sweep
certifies whole-body visibility, that every subsequent route is trackable, or
that passing the old failure point is navigation success. Retain every failure
and evaluate the full 4800-tick goal-and-home outcome.

Implementation: `lewm/recovery_limited_initial_survey_development.py`.
Launcher: `scripts/run_go2_nogil_recovery_hold_development.py --limit-initial-survey`.
Output: `go2_nogil_recovery_limited_survey_jepa_noise_2mm_native_layout00_4800_v1_attempt_001`.
Evaluate with the same flag plus `--evaluate` after owner exit/persistence.

The mixin is inserted at the initial-survey layer, underneath visual-recovery
routing. It bypasses only the interrupted initial sweep. Original source and
recordings remain fixed, including a source snapshot of the preceding launcher.
Same JEPA checkpoint, layout/appearance/physics seeds, six candidates, forecast
horizon, noise, camera scheduling treatment, CPU group, deadline and arrival
rules. Run sequentially without heavy concurrent analysis. This exposed trial
cannot establish independent-layout reliability or change the completed batch.

## First JEPA result and fixed supervised follow-up

The JEPA run completed a physically verified round trip in 293.28 simulated
seconds, zero contacts, 676/714 plans on time, and 33 recovery-active plans.
The survey was deferred at frame 60 after camera frame 58 triggered recovery;
only two observed views remained marked complete, and the sweep stayed
`complete=false`. The first deferred plan retained visual-recovery priority.
Both one-second quiet arrivals passed: maximum physical distances 19.5 mm
outward and 23.8 mm home, maximum 100-ms speeds 27.7 and 32.1 mm/s. The physical
return reversed eleven outward corridor edges, with no invalid graph transitions.

Six recovery publications cancelled nine command windows. No nonzero older
command occurred at a strictly later timestamp or after the request gate had
observed the new recovery threshold. Two requests share the exact virtual-clock
timestamp of a publication but record the previous threshold; those ties do
not determine thread order and are retained explicitly in the treatment receipt.
No exact zero-latency intervention is claimed. Full depth is retained as the
first successful repair reference. Three focused survey tests passed; the
actual runtime method order also preserves outer visual-recovery routing.

Next run one same-layout supervised-rollout mission using the identical combined
prompt-hold and recovery-limited-survey controller. Keep all sensor, model-training,
seed, deadline, candidate and arrival settings from its original comparison arm.
This checks whether the observed repair carries across the two trained models;
it is another exposed-layout development trial, not an independent test or a
revision of the completed 20-run comparison. Preserve its outcome either way.
Use `--limit-initial-survey --arm supervised_rollout`, with `--evaluate` afterward.
Output: `go2_nogil_recovery_limited_survey_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.

## Supervised result and combined interpretation

The supervised follow-up also passed both physical arrivals, in 296.86 simulated
seconds, with zero contacts, 693/720 plans on time and six recovery-active plans.
Maximum physical target distances during the quiet dwells were 13.1 mm outward
and 20.6 mm home; maximum 100-ms speeds were 23.3 and 17.1 mm/s. Outward/home
arrival frames were 1941/2964. The return reversed eleven outward corridors,
with no invalid graph transitions. The startup survey was deferred at frame 60
with two observed stages, and stayed explicitly incomplete. Its first deferred
plan still prioritized visual recovery. One recovery publication cancelled one
older command window; the treatment receipt found no later old-command request.

| Development condition on the exposed maze | JEPA | Supervised |
| --- | --- | --- |
| Original completed comparison | Tracking failure before translation | Tracking failure before translation |
| Prompt cancellation alone | Tracking failure before translation | Not run |
| Prompt cancellation plus recovery-limited startup survey | Verified round trip, 293.28 s | Verified round trip, 296.86 s |

Both successful runs used the original model state for their respective arm.
The repair changes the shared perception/control behavior; it neither retrains
JEPA nor demonstrates that JEPA is superior. Their small total-time difference
comes from different trajectories: JEPA arrived outward 31.3 s earlier, then
took 27.6 s longer returning. One execution per model cannot establish a speed
advantage. On each run's own executed forecast windows, fitted pose-command
position prediction remained more accurate than the neural prediction: JEPA
11.91 versus 7.11 mm RMSE, supervised 10.26 versus 7.21 mm. These overlapping
700-ms windows do not measure unexecuted candidate outcomes or constitute a
same-trajectory comparison between the two neural models.

Combined machine-readable evidence, including the original two failures and
the failed cancellation-only trial:
`go2_nogil_recovery_limited_survey_comparison_v1_attempt_001/result.json` under
the existing navigation artifact root. Per-root treatment, physical arrival,
forecast, source/timing and corridor readouts are preserved. Nine existing
commitment/pulse tests and three focused survey tests passed. The full native
trials, rather than these component tests, provide the navigation evidence.

The supervised launch initially stopped before creating an experiment root
because free recording space was below four GiB. Retiring the completed old
renderer-fix JEPA reference's depth reclaimed 1,137,614,848 allocated bytes;
all 4218 non-depth identities and 39 JSON hashes matched. This ends that older
depth pin; it does not remove its outcome. After diagnosis of the new supervised
success, its redundant depth was retired, reclaiming 1,852,432,384 bytes while
preserving all 6010 non-depth identities and 41 JSON hashes. Keep the new JEPA
repair success, original fresh-maze reference and current failures in full.

The next scientific work is prospective validation of this repair, with matched
controls, and a separate investigation of JEPA's layout-1 prediction/reserve
turn loop. Those mechanisms should not be conflated. Visually dependent
forecast accuracy, multiple training seeds, realistic sensor/timing conditions
and bounded physical-platform evidence remain necessary for the broad goal.
No expanded-candidate or broader environment-type test was run here.

# Causal residual final-goal native result

Both fixed development cases completed collection and raw audit. **0/2 fully
verified arrivals.** The full JEPA case ended 7.8184906 mm from the instructed
goal and passed the native one-second arrival/quiet subcheck, but terminated
with `MISSION_TICK_BUDGET_EXHAUSTED`. Direct remained unsuccessful at
1.101164799 m. Neither case is an independent maze evaluation.

The only prospective controller change was causal observed residual correction
of the existing final-goal utility. Original model forecasts, first-step surface
checks, all eight nominal path checks, mission budget, arrival thresholds,
camera calibration and model weights were retained. Native state was used for
evaluation, not correction or command selection.

## Native and matched-readout evidence

Artifact base:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

| Artifact | SHA-256 |
| --- | --- |
| `go2_causal_residual_final_goal_probe_v1_attempt_001/launch.json` | `29e898c6fab4287857713c0f28913df1b44681010bffbd10b7e5b35fb572e89f` |
| Same native root, `result.json` | `42211a96a2be46429b1e6cb92872acc033f84573411968d04d0d5ea8afc2c484` |
| `go2_causal_residual_final_goal_readout_v1_attempt_001/launch.json` | `531443cb038a3ca512db10e011a04d6e25cc8aaeb0fc890ca5f8a7d983f5d8b8` |
| Same readout root, `result.json` | `2c9f30fd78434d87cb5bac8f5ffb2d18faaf59051165601e9338d596c3886087` |

The completed readout binds 1,373 source files and authenticates both native
attempts, including the preceding executed-horizon attempt. Both current cases
passed raw sensor reconstruction, model-command replay, unchanged model state,
and strict physical visibility, with no hard measurement failure frames and no
physical or acquisition stops. All 311 auxiliary observations passed visibility
checks. These are measurement checks, not navigation qualification.

JEPA produced 253 completed commands, 254 paired observations and 13,400 physics
samples, including ten terminal zero commands. Direct produced 56 completed
commands, 57 paired observations and 3,550 physics samples; its eleven consecutive
infeasible selections ended at tick 46 after ten active zero waits.

Against the previous same-model JEPA run, the first changed command was tick 174.
All 175 observations through that decision have exactly matching physics, policy
histories, gyro histories, primary RGB, auxiliary public depth/mask, observer/map
receipts and model forecasts. The correction changed 15 JEPA selections relative
to their uncorrected final-goal scores. Direct's full 57-observation execution
remained exact, with zero corrected selections. That duplicate is not independent
replication. Between the current JEPA/direct cases the first changed command was
tick 26; their 27-observation physics/sensor/map prefix matches, while their
forecasts differ as expected for different models.

## Why the close JEPA result remains unsuccessful

At tick 241 the observed goal distance was 41.1003 mm. Tick 242 entered the
40 mm controller arrival region at 30.0560 mm and requested zero, with zero
completed quiet intervals. Tick 243 measured 23.0642 mm and one quiet interval,
then hit the unchanged global deadline. The controller therefore never emitted
`OBSERVED_GOAL_CANDIDATE` and stopped observing during its terminal drain.

The native evaluator later measured terminal initial-frame position
`[1.1984859786687223, -0.007670497682940791]` m and distance
`0.007818490602127996` m. Its one-second native radius/speed/zero-request subcheck
passed. Terminal drain observations do not retroactively complete the controller
dwell, and this failed attempt is not resumed or relabeled. A longer prospective
maze mission needs its own declared budget and continuous observation audit.

Maximum observed/native XY discrepancy was 2.21036 mm for JEPA and 1.39570 mm for
direct. Under two-case concurrency, median acquisition-plus-control time was
796.550 ms for JEPA and 772.448 ms for direct. Median full command iterations were
826.740 and 802.806 ms respectively; every command iteration exceeded 100 ms.
Physics was paused during computation. No real-time or hardware qualification
follows from this run.

Next work is the explicit variable-goal outbound/return controller and bounded
native maze collector/auditor. Independent maze navigation, actual backtracking,
matched reactive/non-predictive controls and attribution experiments remain
uncompleted requirements of the active goal.

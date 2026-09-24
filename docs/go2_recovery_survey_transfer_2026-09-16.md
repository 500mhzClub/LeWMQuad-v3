# Prospective transfer of the recovery-limited startup survey

Completed: all four fixed missions are evaluated. The combined repair completed
2/2 round trips; the original completed 1/2. All four had zero contacts.

| Fresh maze | Original | Combined repair |
| --- | --- | --- |
| 0 | Verified round trip, 406.80 s | Verified round trip, 237.26 s; survey deferral not exercised |
| 1 | No arrival; budget exhausted, 481.00 s | Verified round trip, 429.96 s; survey deferred after two views |

This is a two-maze development comparison with one execution per condition,
not a reliability estimate or a JEPA comparison. The repair succeeded on an
independent maze where it activated, but the original also completed its startup
sweep there and failed later. The experiment does not isolate survey deferral
from prompt cancellation, trajectory differences or timing. Both conditions
showed long turning delays. Runtime and model sources stayed frozen throughout.

Run four fixed missions on two fresh mazes using the frozen JEPA model:

| Assignment | Fresh layout | Controller |
| --- | --- | --- |
| 1 | 0 | Original |
| 2 | 0 | Prompt recovery cancellation plus recovery-limited initial survey |
| 3 | 1 | Prompt recovery cancellation plus recovery-limited initial survey |
| 4 | 1 | Original |

This tests transfer and possible regressions of the combined survey repair after
its two exposed-maze successes. It does not isolate prompt cancellation from
survey deferral; the earlier cancellation-only failure remains recorded. An
unexercised survey change cannot establish its efficacy. Keep every outcome and
do not change controllers, replace layouts or retry failures within the batch.

Construction seed 2026091623 selected two layouts from three structural candidates.
Both abstract topology and grid embedding differ from the explicit 92-layout
development registry and from each other. Selection used no navigation outcome.
The layout inventory, assignments, model identities and executable source hashes
were frozen in `go2_recovery_survey_transfer_plan_2026-09-16.json` before mission 1.
These remain development mazes in the same family, not sealed evaluation or
broader environment-type tests.

Primary outcome: physically verified goal-and-home round trip without disallowed
contact. Also record failures, tracking loss, actual survey deferral/cancellation,
physical return corridors, simulated time, planning deadlines and executed-action
forecast errors. Compare each pair's complete outcomes, not only passing times.
Two layouts, one execution per controller/layout and one training seed limit
reliability claims; this is not a JEPA-versus-supervised comparison.

Both controllers retain six candidates, a 0.8-s forecast, 300-ms planning deadline,
20-ms additional publication delay, 4800-tick mission budget, original inference
batch, the camera drawing scheduling repair, persistent observed map, ideal gyro,
2-mm synthetic depth noise and unchanged arrival/clearance rules. Runtime truth
remains evaluator-only. Model fitting, candidate expansion and broader environment
testing are not included. This is measured simulation, not hardware qualification.

Launcher: `scripts/run_go2_recovery_survey_transfer_development.py --assignment N`.
Use the existing native environment, CPU group `0-7,16-23` for layout 0 and
`8-15,24-31` for layout 1. Evaluate each owner after exit/persistence using
`--assignment N --evaluate` before starting the next. No other native job is
active at preparation; recording headroom is about 4.7 GiB. Run sequentially
without heavy concurrent analysis because the comparison includes deadline
behavior. Diagnose completed recordings and retire redundant depth under the
standing policy, retaining active failures and the current repair reference.

## Assignment 1: original controller, fresh layout 0

Verified round trip in 406.80 simulated seconds, zero contacts, no pipeline
faults. Physical one-second quiet dwells had maximum target distances 11.24 mm
outward and 14.62 mm home, with maximum 100-ms speeds 33.45 and 20.94 mm/s.
Both legs traversed eight unique directed corridors; all eight return edges
reversed observed outward edges, with no invalid transitions. The original
startup survey completed all nine views. Seventy plans used visual recovery.

692/1008 plans met the deadline (68.65%). The added publication wait was exactly
20 ms throughout and crossed the deadline for only six plans. Return routing
cost increased markedly: frame bins 2500–2999 and 3000–3499 had median route
wall times 270.2 and 241.9 ms, versus 3–21.5 ms in the early outward bins.
Neural forward computation remained about 8 ms. This points to route computation
as a substantial contributor; it does not isolate every source of latency.
Keep this outcome and finish the fixed comparison before changing runtime code.

On 693 executed windows, neural XY RMSE was 15.87 mm versus 7.33 mm for the
fitted pose-command reference; neural yaw RMSE was 2.71 degrees versus 0.80
for command history. These are trajectory-conditioned, overlapping windows,
not counterfactual navigation results. Full physical, timing, survey, forecast
and corridor readouts are saved. Redundant depth was selected for retirement
under the standing policy; all non-depth evidence and existing repair/failure
references remain. Aggregate: `go2_recovery_survey_transfer_comparison_v1_attempt_001/result.json`.

## Assignment 2: repair controller, fresh layout 0

Verified round trip in 237.26 simulated seconds, zero contacts, no pipeline
faults, 573/584 plans on time (98.12%). Maximum physical distances during the
quiet arrival dwells were 20.55 mm outward and 8.79 mm home; maximum 100-ms
speeds were 0.23 and 18.55 mm/s. The return reversed all eight outward
corridors, with no invalid transitions. Actual additional publication delay
was exactly 20 ms, crossing the deadline for six plans.

The startup survey completed all nine views and was never deferred. Thirteen
later recovery publications cancelled twelve older command windows. No older
nonzero request occurred at a strictly later clock timestamp. Three clock ties
record the gate's old recovery threshold; their within-tick order cannot be
inferred from equal timestamps. Complete receipts remain in the readout.

Both controllers completed this maze. The repair run was 169.54 s faster,
including 43.1 s earlier outward arrival and 126.4 s less return time. Its route
component medians stayed at 2.6–20.2 ms across the recorded 500-frame bins.
Different trajectories and maps, the original's expensive return routing,
and one execution per condition prevent attribution of this time difference
to a repeatable repair benefit. This pair did not exercise survey deferral
and cannot demonstrate that component's efficacy.

On 554 executed windows, neural XY RMSE was 11.07 mm versus 6.17 mm for
pose-command; neural yaw RMSE was 2.43 degrees versus 0.65 for command history.
The model is unchanged. Two assignments on the second fresh maze remain,
with condition order reversed as fixed before the first outcome.

## Assignment 3: repair controller, fresh layout 1

Verified round trip in 429.96 simulated seconds, zero contacts, no pipeline
faults, 1024/1067 plans on time (95.97%). Maximum physical dwell distances were
15.40 mm outward and 9.31 mm home; maximum 100-ms speeds 6.58 and 11.10 mm/s.
Twelve unique directed edges were traversed outward; all eight return edges
reversed observed outward corridors, with no invalid transitions.

Survey deferral activated at frame 68 after recovery triggered at frame 67
(8.2 s on the measured clock). Only two views were complete; the survey remained
explicitly incomplete. One recovery publication cancelled one older command
window, with no later old-command request and no equal-clock tie. This is a
prospective successful navigation episode that exercised the combined repair;
the original-controller comparison on this maze remains pending.

A long outward delay remains scientifically important despite completion.
Over frames 800–3099, all 575 plans routed toward a frontier, with 289 left
turns, 284 right turns and two holds. 558/575 were on time. There were 87 new
alternative-turn latches, 66 preferred-heading releases and 37 hold-relative
recovery reselections. Movement resumed after this interval; outward arrival
was frame 3667 and home frame 4296. This resembles the earlier prediction/
clearance/recovery turning problem and was not mainly a deadline outage.
It is not yet a causal diagnosis of the unexecuted alternatives.

Neural XY RMSE was 9.99 mm versus 5.79 mm for pose-command on the saved executed
windows; neural yaw RMSE was 2.83 degrees versus 0.99 for command history.
Keep this full recording as the fresh-maze repair and turning-loop reference.
The current comparison still uses unchanged frozen runtime and model sources.

## Assignment 4 and completed interpretation

The original controller exhausted its budget at 481.00 simulated seconds with
no arrival, zero contacts, intact tracking, no pipeline faults and 1179/1200
plans on time (98.25%). Its initial survey completed all nine views. It spent
a long period turning toward a frontier, then increasingly requested hold and
additional views. Over frames 1600–3099 it selected 362 turns in 375 plans;
368 were on time. Frames 3100–4800 contained 237 turns and 189 holds.
The full failure recording remains retained alongside the repaired counterpart.

Its neural XY RMSE was 8.61 mm versus 5.10 mm for pose-command over 1175
executed windows; neural yaw RMSE was 3.14 degrees versus 0.96 for command
history. Pose-command XY and command-history yaw were more accurate on each
of the four recordings' own windows. Different trajectories and overlapping
windows preclude treating these as counterfactual navigation outcomes.

The combined repair has now succeeded on both new layouts, but useful
reliability remains unproven: one repair success spent about 230 s repeatedly
turning, and the original failed on that maze despite timely planning. The
next bounded mechanism test concerns early reversal of a clearance-recovery
turn. The paired event diagnosis is saved in
`go2_turn_release_cycle_diagnosis_v1_attempt_001/result.json`; its script is
`scripts/read_go2_turn_release_cycles_development.py`.
The complete four-run aggregate is reproducible from the saved per-run
readouts using `scripts/read_go2_recovery_survey_transfer_development.py`.
Six candidates, one model seed, synthetic noisy depth and ideal gyro remain;
broader environment types and candidate expansion remain deferred. No hardware
validation or overall goal completion is claimed.

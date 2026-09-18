# Test early reversal of clearance-recovery turns

Completed: both assigned missions passed physical round-trip checks, but neither
contained an eligible early reversal. The intended ablation was unexercised in
both. These results do not demonstrate a fix or justify promoting the variant.

The completed four-run survey-transfer comparison yielded repair 2/2 round
trips and original 1/2, zero contacts. Turning delays persist independently of
startup survey completion. A bounded post-hoc event analysis now identifies a
specific recurrent decision in three full recordings:

| Recording | Early releases with a later re-latch | Median time to re-latch | Median XY displacement | Median release reserve margin | Median re-latch reserve deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Earlier replication JEPA layout-1 failure | 87/87 | 2.0 s | 9.87 mm | 1.79 mm | 2.22 mm |
| New repaired layout-1 success | 68/69 | 1.8 s | 10.24 mm | 2.13 mm | 2.58 mm |
| New original layout-1 failure | 61/61 | 2.0 s | 10.08 mm | 2.77 mm | 2.49 mm |

Every paired event re-latched after the same preferred pure turn was blocked
again. Median changes in the recovery target heading were 0.0048, 0.0121 and
0.0102 radians. Both plans were on time in 84/87, 66/68 and 60/61 pairs.
These are event-conditioned, potentially overlapping post-hoc observations,
not independent trials or a counterfactual success claim. In particular,
the released action being clear now does not mean it will remain clear while
approaching the desired heading. Reversal can undo long-way recovery progress.

Hypothesis: suppressing only the early preferred-heading release will reduce
repeated reversal and permit useful progress. This is an ablation of an earlier
controller rule, not a new reserve threshold, candidate bank or trained model.
It can also make useful turns longer or expose tracking trouble; retain any
negative outcome. Existing measured-heading completion, translation-progress
release, hold-relative recovery, blocked-direction switching, visual recovery,
clearance, stopping and terminal-arrival rules remain active.

Run two fixed exposed development cases, sequentially on CPU 8–15,24–31:

1. Earlier four-maze replication layout 1, original JEPA controller with only
   early heading release suppressed; compare to its preserved budget failure.
2. New survey-transfer layout 1, repaired JEPA controller with the same ablation;
   compare to its preserved 429.96-s success and long turning interval.

Keep each reference's geometry/appearance/physics/noise seeds, frozen model,
six candidates, 0.8-s horizon, 20-ms extra publication wait, 300-ms deadline,
4800-tick budget and original inference computation. No retuning, retries or
replacement layouts within these two cases. A single exposed run per case is
a mechanism test, not independent validation or a speed-benefit estimate.

Primary outcome remains the full physically verified goal-and-home mission
without contact. Also inspect actual suppressed-release receipts, turning
events, translation, tracking, deadlines, forecasts and contacts. Improvement
on saved predictions alone is insufficient. A result with no suppression
activation cannot establish this rule's effect.

Implementation: `lewm/no_early_heading_release_development.py`; launcher:
`scripts/run_go2_no_early_heading_release_development.py --assignment N`.
Frozen plan: `docs/go2_no_early_heading_release_plan_2026-09-16.json`.
Eleven focused tests passed, including unchanged clearance/hold, measured turn
completion, translation-progress release and actual method-layer composition.
Preparation also checked placement in both complete runtime classes. One initial
test collection exposed a closure restriction in the reused binding helper;
the implementation now uses an explicit method-layer call. No native attempt
was started with that failed implementation.

The broad goal remains incomplete. This study does not establish JEPA benefit,
sensor realism, real-time performance, other-environment transfer or hardware
readiness. Expanded candidate counts and broader environment types stay deferred.

## Assignment 1: successful navigation, intended intervention unexercised

The replication-layout-1 ablation completed a physically verified round trip
in 282.46 simulated seconds, zero contacts, no pipeline faults and 675/697
plans on time. Physical quiet-dwell maximum distances were 11.08 mm outward
and 15.66 mm home; maximum 100-ms speeds 29.20 and 11.96 mm/s. Its return
reversed all eleven outward corridors with no invalid transitions. Actual
additional publication delay was 20 ms for 696 plans and 22 ms for one.

There were **zero eligible early releases to suppress and zero alternative-turn
latch events**. This outcome does not demonstrate that suppressing early
reversal repairs the old failure. The original reference still has its recorded
480.90-s budget failure, no arrivals and 1176/1200 plans on time.

The first common-tick requested-command difference was at measured time 12.2 s,
well before the old prolonged turning interval. Both selected a right turn at
frame 104, observation time 11.9 s. The old plan completed at 12.130 s and
dispatched at 12.2 s. The ablation plan completed at 12.204 s, four milliseconds
beyond its deadline, and requested hold with `NO_ON_TIME_PLAN`. This changed the
subsequent trajectory. A single changed episode cannot separate scheduling
variation from intervention overhead or establish the intended behavioral
mechanism. Do not replace the old failure or count this as a successful causal
repair test. The successful mission and failure both remain in the evidence.

The second fixed case, on the new survey-transfer maze with the combined
startup repair, launched after the first owner exited and evaluation finished.
Its result is pending. Current aggregate:
`go2_no_early_heading_release_comparison_v1_attempt_001/result.json`.
Reader: `scripts/read_go2_no_early_heading_release_development.py`.

## Assignment 2 and final interpretation

The survey-transfer/repaired ablation completed a physically verified round
trip in 175.60 simulated seconds, zero contacts, no pipeline faults, 410/431
plans on time. Physical quiet-dwell maximum distances were 19.46 mm outward
and 12.57 mm home; maximum 100-ms speeds 9.48 and 18.45 mm/s. The return
reversed eight observed outward corridors, with no invalid transitions. All
431 extra publication waits were exactly 20 ms. Startup survey deferral
activated at frame 68 after two completed views, matching the reference's
recorded trigger and deferral frames.

Again there were zero eligible early releases, zero suppressions and zero
alternative-turn latches. The first command difference from the 429.96-s
reference was at measured time 12.6 s: both selected a right arc at frame 108,
observation time 12.3 s. The reference plan completed at 12.558 s and dispatched;
the ablation plan completed at 12.648 s, missed its deadline by 48 ms, and
requested hold. Both complete runtime method orders were checked after these
unexpectedly unexercised outcomes; they preserve every reference class in the
same order, inserting only the new class and the intervention mixin. The
recorded native outcomes and the unsuccessful causal test remain separate.

Neural position RMSE on this second run's 404 executed windows was 12.32 mm
versus 8.10 mm for pose-command; neural yaw RMSE was 1.99 degrees versus 0.68
for command history. No predictor advantage or counterfactual outcome follows.

The fixed two-case study is complete: two verified navigation successes, zero
contacts, **zero exercised behavioral ablations**. Do not replace the old
failures, advertise a 2/2 repair rate, or infer that the saved turning cycles
have been fixed. Scheduling differences can change early actions and later
trajectories even with the same scenario and model seeds. Intervention overhead
and host scheduling have not been experimentally separated.

Next scientific priority is a small balanced repeated comparison with fresh
control runs, using the same computation wrapper for enabled/disabled early
release, fixed before outcomes. Keep each execution and measure both activation
and the complete mission. This must precede any claim that the turn-rule change
improves reliability. No large training job or candidate-count expansion is
justified by these unexercised tests. Both native owners exited and physical
evaluations are complete. After the bounded diagnosis, redundant depth from
these two unexercised successes was retired under the standing policy. Results,
first-difference records and all non-depth data remain; consult each root's
depth-retention marker before replay. Active failures and the exposed/fresh-maze
repair references remain full. The subsequent four-run protocol is in
`docs/go2_turn_release_repeatability_2026-09-16.md`.

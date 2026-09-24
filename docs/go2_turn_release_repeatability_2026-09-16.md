# Balanced repeated comparison of early recovery-heading release

The previous two ablations completed round trips but never exercised the intended
change. Their first command differences from the older references were early
deadline misses. Comparing one asynchronous mission to an older failure cannot
establish a controller improvement. Preserve those unexercised outcomes.

Run four fixed missions on survey-transfer development layout 1, using the
combined startup-survey/prompt-cancellation repair and frozen JEPA model:

| Assignment | Early heading release | Repetition |
| --- | --- | --- |
| 1 | Enabled | 1 |
| 2 | Disabled | 1 |
| 3 | Disabled | 2 |
| 4 | Enabled | 2 |

Both conditions execute the same wrapper and compute the original release
proposal. When eligible, the enabled condition applies it; the disabled condition
keeps the incoming recovery action/state. Both record eligibility and treatment.
No eligible proposal means identical decision output in both modes. Existing
clearance, blocked-direction switching, measured turn completion, translation
progress, visual recovery, deadline, stopping and arrival rules remain active.
This does not equalize host scheduling or guarantee deterministic trajectories.

The enabled–disabled–disabled–enabled order balances first/last placement within
this small sequential study. Keep the same scenario, appearance, physics and
2-mm depth-noise seeds, CPU group 8–15,24–31, original model inference, six
candidates, 0.8-s horizon, 20-ms publication wait, 300-ms deadline and 4800-tick
budget. No training or parameter changes, outcome-conditioned retries, extra
repetitions or replacement layouts within the batch. All outcomes remain.

Primary outcome is the complete physically verified goal-and-home mission without
contact. Report every failure, actual release/suppression activation, turning,
translation, quiet arrivals, physical backtracking, tracking faults and planning
deadlines. Compare the two fresh control executions with the two disabled
executions and retain the older references separately. Inspect first command
differences without claiming a unique timing cause from a single trace.

This is a bounded repeatability/mechanism study on one exposed maze and one model
seed. It cannot establish population reliability, a JEPA advantage or hardware
readiness. If the rule remains unexercised, report that limitation rather than
continuing until a favorable case appears. A later independent validation would
be a separate decision, not an extension of this four-run batch.

Launcher: `scripts/run_go2_turn_release_repeatability_development.py --assignment N`,
then `--assignment N --evaluate` after that owner exits and finishes recording.
Prepare the fixed source/case plan before assignment 1. Run sequentially because
the comparison measures deadline behavior; no heavy parallel analysis during
timed execution. The two matched-wrapper tests and three earlier composition/
clearance tests passed. Only diagnosed redundant depth may be retired under the
standing policy, preserving all outcomes and current active failure/reference
inputs. No sealed material or hardware execution is involved.

## Completed result

All four fixed assignments completed and were physically evaluated. No extra
repetitions or replacement layouts were run.

| Assignment | Condition | Physical round trip | Simulated seconds | Plans on time | Eligible releases |
| --- | --- | --- | --- | --- | --- |
| 1 | Enabled, repetition 1 | Yes | 243.70 | 575/599 | 0 |
| 2 | Disabled, repetition 1 | Yes | 193.60 | 439/465 | 0 |
| 3 | Disabled, repetition 2 | Yes | 175.28 | 410/430 | 0 |
| 4 | Enabled, repetition 2 | Yes | 172.38 | 408/424 | 0 |

Zero contacts occurred. Both quiet physical arrivals passed in every run. Each
return traversed eight unique corridor edges reversing observed outbound edges,
with no invalid graph transitions. Overall, 1832/1918 plans met their deadlines
(95.52%). These are measured-simulation deadlines, not real-time qualification.

There were zero alternative-turn latches and zero eligible early-heading
releases in all four runs. The intended intervention was therefore unexercised.
The fastest execution was an enabled control; differences in completion time
do not establish a benefit or harm from disabling the rule. These four successes
are repeated navigation evidence on one exposed development maze with one
training seed, not four independent maze successes or a JEPA advantage.

Pairwise request records locate the first command divergences. At 8.38 seconds,
enabled repetition 1 still requested a turn while the other three held with a
latched command-window veto; its recorded visual-recovery minimum had not yet
updated. At 61.80 seconds, disabled repetition 1 executed a forward plan while
disabled repetition 2 and enabled repetition 2 held for a missed deadline. The
same frame-600 forward plan finished at 61.762, 61.814 and 61.820 seconds,
respectively. The latter two executions first differed at 65.36 seconds when
one held for stale/unavailable observation and the other continued forward.
These are recorded differences in publication/observation timing, not proof of
a unique host-level cause or of which later trajectory was counterfactually
necessary. Both same-condition repetitions diverged without a release decision.

On the executed forecast windows in every run, the fitted pose-command baseline
had lower XY RMSE than JEPA (7.45–8.32 versus 11.60–12.99 mm), and command-history
yaw had lower RMSE (0.65–0.73 versus 2.00–2.20 degrees). These are prediction
comparisons on the recorded trajectories, not alternative-policy outcomes.

Each root retains `heading_release_repeatability_readout_v1.json`. Aggregate
`result.json` and `complete_scientific_readout_v1.json` are under
`go2_turn_release_repeatability_comparison_v1_attempt_001` in the navigation
development artifact volume. The latter includes all six pairwise first-command
differences and nearby plan publications. Its reproducible reader is
`scripts/read_go2_turn_release_repeatability_summary_development.py`.

Close this four-run experiment without promoting the disabled rule. Keep the
original release behavior for the next independent maze comparison of the
startup-repaired planner and matched baselines. Earlier failures and long
turning episodes remain separate preserved results. Broader reliability,
multiple training seeds, realistic sensing and hardware evidence remain open.

After evaluation and diagnosis, all four unexercised-success depth recordings
were retired under the standing retention policy: 15,680 native depth leaves,
4,729,438,208 allocated bytes reclaimed. Every non-depth file and JSON result was
preserved and checked; per-root `depth_retention.json` and `DEPTH_RETIRED` record
that full historical sensor replay is unavailable. The earlier exposed and
fresh-maze JEPA repair references, and all failure recordings, remain full.
Inventories are under `.generated/depth_retirement_unexercised_heading_repeatability_assignmentNN_2026-09-16/`.

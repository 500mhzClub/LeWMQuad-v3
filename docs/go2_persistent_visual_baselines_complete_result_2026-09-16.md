# Complete current-controller baseline comparison

All twelve fixed baseline missions are complete and physically evaluated. The
instantaneous-ranking controller completed 4/4 round trips, reactive feedback
3/4, and current-reserve feedback 0/4. All twelve recorded zero contacts.
Every failed mission remains in the result population. The broader navigation
goal remains incomplete; this study does not establish a JEPA advantage.

Together with the sixteen preceding model comparisons on the same four
development layouts, the complete record contains 28 missions, 22 verified
outbound arrivals and 21 verified round trips, with zero recorded contacts.
There are four layouts and one execution per controller/layout, not 28
independent environments. Models were frozen before layout construction;
the layouts were already exposed before this baseline batch.

| Controller | Verified goals | Verified round trips |
| --- | ---: | ---: |
| Supervised world model | 4/4 | 4/4 |
| Pose/command predictor | 4/4 | 4/4 |
| Instantaneous ranking with predictive checks | 4/4 | 4/4 |
| JEPA world model | 3/4 | 3/4 |
| Direct prediction | 4/4 | 3/4 |
| Reactive heading feedback, prediction off | 3/4 | 3/4 |
| Current reserve feedback, prediction off | 0/4 | 0/4 |

## What the controls establish

Instantaneous ranking replaces forecast-based main utilities with the current
waypoint cost derivative. Predictive clearance, recovery, arrival and stopping
checks remain. Main action preferences differed on 678/2,199 plans; downstream
predictive checks overrode the instantaneous preference 24 times. The treatment
was exercised, but this is not a prediction-free controller.

The full supervised planner finished faster in all four recorded pairs:

| Layout | Supervised simulated seconds | Instantaneous simulated seconds |
| --- | ---: | ---: |
| 0 | 160.92 | 198.86 |
| 1 | 204.88 | 255.60 |
| 2 | 148.42 | 199.98 |
| 3 | 201.68 | 244.82 |

This is a consistent observed efficiency pattern worth replicating. It does
not establish a reliable speed benefit: actual deadlines differed, the
controller was developed with the supervised model, and there is only one
execution per pair. No successful-only average is used to rank failed arms.

Both prediction-off controls compute the same supervised forecasts for
workload control but do not consume them for action selection. They retain
tracking, mapping, routing memory, observation-age and actual stopping guards.
Their selector costs and feasibility/recovery rules differ from the full
planner. Reactive feedback uses nominal clearance and heading-first terminal
feedback; reserve-only feedback demands an additional 3 cm for movement and
has no predictive escape from a current reserve deficit. These are controller
package comparisons, not isolated removal of every geometric prediction.

Reactive feedback's three successes establish that forecast-free selection
can navigate these layouts with the current perception and memory package.
Its layout-0 failure was sustained terminal turning: 973 right-turn selections,
one forward pulse and 14 holds, without the required quiet dwell. It was not
primarily a missed-deadline pause.

All four reserve-only failures were persistent clearance-rule holds. Blocked
plan counts were 1,090/1,065/1,022/1,053. Layout 2 briefly recovered after its
first rejection, then held for 1,021 consecutive plans. The first stored
clearance rejection occurred above the physical 0.48-m threshold on layouts
1–3, but not layout 0. The map-discrepancy cause and safe turn-only escape
remain unisolated. These failures do not demonstrate learned-model necessity.

The preceding failures also remain: JEPA lost tracking on layout 3, reproduced
at the same frame by exact sensor replay; direct prediction reached layout 0's
goal but exhausted its budget before returning. On every one of the 28
executed recordings, the pose/command forecast had lower same-window
XY RMSE than the neural forecast. The windows overlap, differ between
trajectories and do not establish alternative-policy navigation outcomes.

## Scope and next work

This is development simulation with RGB, synthetic 2-mm depth noise and ideal
gyro, substantial simulator lag and one neural training seed. It is not sealed
final evaluation, calibrated real sensing, real-time qualification or hardware
evidence. The earlier routing-memory study separately supported accumulated
spatial routing evidence (4/4 versus 0/4); it did not isolate learned internal
memory or return-specific memory.

Next, examine a bounded failure-triggered old-view fallback on the retained
JEPA/layout-3 sensor recording, preserving measurement acceptance limits and
recent references. The existing exact replay establishes the failure, not that
this proposed fallback works. Any justified repair then needs prospective
closed-loop verification. Replication, learned internal-memory attribution,
realistic sensing/timing and bounded real-platform evidence remain outstanding.
Broader environment-type tests remain deferred at the user's request.

Authoritative aggregate:
`go2_persistent_visual_baselines_complete_v1_attempt_001/result.json` under the
navigation artifact root. It contains all 28 individual outcomes, treatment
receipts, failure diagnoses, prediction readouts and timing comparisons.
The four `go2_persistent_visual_baselines_comparison_layoutXX_v1_attempt_001`
roots contain complete baseline/reference trajectories and inspected PNG/SVG
figures. See [the journal](go2_persistent_visual_baselines_2026-09-16.md).
Completed diagnosed depth is retired under the standing retention policy;
all non-depth evidence and active failure-replay inputs remain.

Follow-up: [the retained-failure probes](go2_recent_local_view_bank_probe_2026-09-16.md)
found no eligible old view at the failed frame; all sixteen normal stored-view
camera fits failed. A newest-reference-per-heading-bin replay also failed at
the same frame despite changing selected references. Neither proposed lookup
repair was adopted for native navigation; feature/visibility and recovery
commands are the next diagnosis.

Later [recovery-timing follow-ups](go2_visual_recovery_dispatch_hold_2026-09-16.md)
added two physically successful layout-3 missions with zero contacts. The
experimental hold never activated in its run; an original-controller repeat
also succeeded. Early deadline/freshness decisions changed the command paths.
These exposed follow-ups provide variability evidence, not a demonstrated
repair, and do not revise this fixed study's outcomes.

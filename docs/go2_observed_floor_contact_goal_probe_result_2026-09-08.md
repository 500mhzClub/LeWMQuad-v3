# Observed floor-contact native result

The declared four-foot measured-floor contact policy enabled the fixed corrected
JEPA model to continue around the obstacle and approach the instructed goal.
It did not complete the controller's arrival dwell. Both methods retain
**verified_goal_reached = false**, so the paired result is **0/2**. All sensor
and exact fresh controller replay gates passed.

| Outcome | JEPA | Direct |
| --- | --- | --- |
| Terminal decision tick | 226 | 46 |
| Active zero-wait ticks | 216–225 | 36–45 |
| Commands including terminal drain | 236 | 56 |
| Paired frames | 237 | 57 |
| Native physics samples | 12,550 | 3,550 |
| Minimum native goal distance, m | 0.04630226449055245 | 1.1011647990168776 |
| Terminal native goal distance, m | 0.05555654737606409 | 1.1011647990168776 |
| Native 6-cm one-second quiet-arrival condition | pass | fail |
| Full controller-and-native goal gate | fail | fail |
| Maximum observed XY error, m | 0.002210362649595023 | 0.0013957027851192636 |
| Median complete iteration, ms | 818.8344440000001 | 810.1179215 |
| Maximum complete iteration, ms | 1197.838355 | 1056.575278 |

Neither case reported a physical or acquisition stop. All 294 paired frames
passed the measurement gates, and all 292 complete command iterations exceeded
100 ms. Physics remained paused during computation. Neither terrain support
nor hardware readiness is certified by the explicit floor-contact policy.

JEPA first executed a different command at tick 42 (right_arc instead of zero).
The first 43 observations exactly matched its predecessor in native physics,
policy/gyro histories, primary RGB, observer/map and forecasts. Direct matched
all 57 observations, every command and terminal state of its predecessor. The
new JEPA/direct pair first differed at command 26; all 27 shared-prefix native,
public, RGB, auxiliary depth/mask and observer/map records were exact, while
model forecasts differed. The contact-policy improvement is not evidence of
a JEPA training advantage or an independent-maze result.

The new JEPA trajectory had observed distance 0.9160454121171319 m at tick 100,
0.531948658249341 m at 150, and 0.058851057021948885 m at 200. Its closest
observed sample near arrival was 0.04744238319893259 m at 213, outside the
unchanged 0.04-m controller radius. At 216 its observed nominal radius crossed
the wall-cell constraint; it waited ten commands and terminated at 226. All
six terminal surface checks passed, but nominal first/eight-step checks failed
against cell [13,-2], with first-step clearances 0.43903420137575855–
0.44627653660346056 m against the unchanged 0.45-m requirement.

The completed readout also exposed a final-target mismatch. The selector still
aims at the goal cell centre [1.175,0.025], while the mission arrival check uses
the instructed initial-body point [1.2,0]. Near arrival the recorded proposal
was OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL with complete route-floor coverage, yet
intermediate_target_is_mission_goal remained false. Next is an explicit final
target correction: retain intermediate/frontier waypoints, and target the
instructed point only when the goal-cell route reaches its final waypoint and
the actual final connector is fully observed and nominally clear. Keep both
arrival thresholds, all collision/nominal checks, models and budgets unchanged.
Its source candidate passed five connector tests (0.15 s) and an inherited
scoring/execution scope test (1.71 s); no native outcome from it is established.

A separate training-only plan inventory authenticated windows.json SHA-256
c293e454ac391da377a282c61dc30dd6abbdd5274d253c963e4837a78e8f7811.
Among 408 available training contexts: 248 eight-step plans had no zero command,
30 had five moving commands then three zeros, 96 eight-step plans began with
zero, and 34 five-step known plans began with zero. The specific one-moving-
command/seven-zero stopping pattern is not present. This is a coverage check,
not stopping accuracy or a reason to change this completed trial.

The native run bound 1,313 sources, ran two one-thread CPU scenes, and took
400.285280801123 seconds after launch. Hardware admission recorded
82,422,005,760 bytes available RAM and 66,203,332,608 bytes free artifact storage.
The readout bound 1,316 sources and its two comparison tests passed in 2.05 s.

| Artifact | SHA-256 |
| --- | --- |
| Native launch | 1ccf500f854f42e4976353ea9c9f33f4026d58a566ea14c2c736f7c3febb9e6d |
| Native result | aab193a96edcee6fa2e15a62aab333821f6688995ca8f9e68cdc8887eaf403d9 |
| Readout launch | 8f38d2246824a84084600014e7bc58bd98d9feb23d2288ca229cf20c2a7248b4 |
| Readout result | aaa0b6202883a22b46acdfe9873414cf76c4b159767bebea1e20263ad9f476dc |

Roots: go2_observed_floor_contact_goal_probe_v1_attempt_001 and
go2_observed_floor_contact_goal_readout_v1_attempt_001. The full goal remains
active: independent mazes, verified arrivals, useful backtracking and matched
reactive/nonpredictive/planning/memory comparisons remain unestablished.

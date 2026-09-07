# Paired inner-arrival room-return simulation V1

Run one fresh physical-simulation batch to isolate the consistent internal
position-arrival region change from 6 cm to 4 cm. Reuse the exact scene,
starts, physics/appearance seeds, original multi-reference tracker, RGB-D/body
sensors, gait checkpoint/gains, nominal pulse table, budgets and external
acceptance of the completed intent-return V1 batch. That baseline is 0/3 full
returns; a passing final-home hold does not erase its failed intermediate hold.
This is a paired development intervention, not an independent-layout test,
JEPA experiment, autonomous maze demonstration or hardware evidence.

## Fixed factor and conditions

Use InnerGoalRoomReturn with its consistent 4 cm planning/ranking/arrival and
settling region. External native position tolerance remains 6 cm. Signed yaw
tolerance remains .05 rad. Keep all seven scripted stages, persistent return
intent and one uninterrupted visual frame. No balanced-feature frontend,
relocalization/reset, online model fit, native pose or friction input to control.

Run nominal_left, nominal_right, lower_friction_left once in that order.
Keep the exact frozen intent-room specification and session implementation:
left (.05,-.38,.375)/yaw+.055, physics2026090701/appearance2026090705;
right(-.05,.35,.375)/yaw-.065, physics2026090702/appearance2026090706.
Low friction shares left starts/seeds/geometry and changes robot/floor friction
from1 to.15. Do not omit it or interpret good tracking as good dynamics transfer.

The table still uses only two older nominal samples per command-duration cell.
Commands remain forward.20 and yaw+/-.45 for2/5ticks. Preserve first-pulse
feedback, minimum20 braking ticks,10 quiet intervals, maximum40 braking ticks,
35 pulses/1000ticks per leg,1m anchor excursion/.08m target overshoot,
140 pulses/3600ticks/36legs per mission and10 terminal-zero drain ticks.
The 4 cm region is an engineering reserve, not a certified pose-error bound.
Recorded-state/model-matched candidates do not guarantee this changed trajectory.

## Integrity, pairing and resources

Exclusive new output:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_inner_arrival_room_return_v1_attempt_001`.
Pair against the exact `go2_intent_room_return_v1_attempt_001` in the same parent.
Bind its launch/result/raw audit and every listed collection artifact through
the separate external-artifact guard. Bind all inherited source/input/native
dependencies plus new collector/auditor/controller/tests and this protocol.
Pin the completed balanced-replay result and recorded-inner-arrival diagnostic
as identity evidence only. No old outputs are mutated, relocated or resumed.
No source export or sealed access. Freeze sources at launch; no retries or
acceptance/seed/budget changes within this attempt. Preserve partial failures.

Keep the15GiB three-trial estimate plus40GiB storage reserve: require55GiB
initially,45GiB before each trial, record per-decision free bytes and stop
below40GiB through the guarded terminal-zero path. Native physical stops end
simulation immediately. All data is new; no deletion to make room.

## Independent audit and reported outcomes

Audit all actual raw sensors, depth, contacts, gains, geometry, time stamps,
commands/slew/phases, complete runtime decisions and memory. Additionally
require exact matching of all750 setup samples for base pose/twist, joints,
requested/applied commands and the first RGB identity against the baseline.
A mismatch prevents a paired causal claim; preserve and report the failure.
Do not require matching after the intervention changes commanded actions.

Reuse the baseline score functions unchanged. Every completed local goal and
the final independent home hold must pass all501 native poses over1s:
position.06m, yaw.05rad, speed.02m/s, yawrate.05rad/s; separately verify signed
winding. Full return requires all stages, all local holds/winding, final home
and no native stop. Report both individual failures and aggregate outcomes,
pulses, path/time, tracking error/coverage and compute cost for both arms.
Success in one room or one paired batch is not reliability or generalization.

If collector artifacts are complete, start the separately frozen raw auditor
once; report scientific failures even when acquisition/audit integrity passes.
Keep ideal hidden-robot RGB-D, level-floor support, uncalibrated uncertainty,
paused-compute timing and missing body-sweep safety explicit. The full goal
still requires useful learned prediction, observed branching and online memory,
independent mazes/seeds, matched training/rollout/memory studies, realistic
sensing/deadlines and bounded hardware evidence when available.

# Anticipate stopping projections during action selection

## Verified native outcome

The exposed learned maze-3 follow-up completed a physically verified round trip
in 227.56 simulated seconds, with 2,274 accepted poses and no disallowed contacts.
Goal/home events occurred at frames 1420/2272; maximum physical quiet-dwell
distances were 10.602/14.741 mm. Both passed zero-command and measured quiet-motion
checks. Median/maximum pose errors were 1.709/5.486 mm. Path length was 27.623 m,
with 543 of 561 selected plans on time. Owner exit was 0 after 5:42.79 including
recording, maximum RSS 13,500,512 KiB, no swaps. All four evaluations are saved.

The stopping-projection check changed 12 selected actions. Actual stopping-margin
vetoes fell from 25 to zero. Turn-only command time fell from 317.90 to 61.50 s;
translation time was 151.88 s and zero-command time 13.94 s. All 19 completed
view tasks ended in actual patch observation; the longest was 14.4 s, versus
76.8 s in the predecessor. There are 150 identical common source bindings and
matching shared settings. Comparison and PNG/SVG:
`go2_planned_stopping_projection_layout03_summary_v1_attempt_001`.

The 539 overlapping executed windows had corrected XY RMSE 8.154 mm, maximum
21.876 mm and no errors over 30 mm. Learned yaw RMSE was 3.235 degrees versus
the saved command alternative's 1.515 degrees. Forecast loss did not improve
uniformly; the native result supports the planning intervention on this exposed
case, not a learned-model advantage or general reliability. The next fixed
fresh-maze comparison is documented in `go2_stopping_projection_transfer_2026-09-15.md`.

The local-reference learned maze-3 run reached a verified goal but exhausted
its return budget. Its longest frontier-view task lasted 76.8 seconds
(46.3–123.1 s). Replaying 303 mapping updates matched all 192 episode plan map
counts. Every plan remained outside the selected camera-viewpoint arrival radius
(1.222–1.384 m away). There were 132 routing plans and 60 post-translation-veto
view plans, with 93 right turns, 84 left turns and 15 right arcs. Twelve actual
stopping-margin vetoes triggered the fixed leftward recovery view, opposing the
rightward route alignment. This is an approach/dispatch loop, not interrupted
committed viewing.

The dispatcher projects each translation from its latest observed body pose
using remaining command duration, observation age and a 0.5-second stopping
allowance. The planner previously checked the predicted 800-ms motion path
against stored obstacles without this additional requested-speed projection.

`lewm/planned_stopping_projection_development.py` evaluates those projections
from candidate predicted poses and headings against the existing observed fine
map. It includes possible camera observations from 0.1 s after planning through
the last camera tick before expiry, allowing the existing 0.2-second observation
age. The projection duration is expiry minus observation time plus 0.5 seconds.
Terminal translation pulses retain their 0.1-second duration. No future depth
or native state is supplied, and the stopping allowance remains uncalibrated.

An offline replay of the same 192 plans anticipated all 12 actual stopping
vetoes, but also blocked three other selected translations. All 15 translations
were blocked; this extra conservatism could impede navigation. Replay took
33.06 seconds. It does not establish an alternative navigation outcome.
Result is the retained run's `planned_stopping_projection_replay_v1/result.json`.

The runtime changes only a selected translation whose stopping projection is
blocked. It then chooses a turn or hold using the existing utilities and
prediction-clearance eligibility; it does not authorize a previously blocked
turn or substitute another translation. Actual dispatch guards, floor rejection,
arrival checks and physical limits remain unchanged. Four focused tests passed
in 1.83 seconds, covering stopping extension versus short pulses, predicted
heading, map translation and preservation of blocked-turn eligibility.

Fix one exposed learned maze-3 follow-up using
`scripts/run_go2_planned_stopping_projection_development.py`, output
`go2_planned_stopping_projection_learned_noise_2mm_native_layout03_4800_v1_attempt_001`.
Keep the local-reference tracker, committed camera views, model/fits, learned
XY/yaw, disabled contact score, layout, sensors, timing and 480-second budget.
No floor-reacquisition change is included. After owner exit evaluate physical
arrivals, prediction accuracy, actual vetoes, changed actions and elapsed view/
return behavior. Retain any failure. An exposed improvement cannot establish
fresh-maze reliability or learned-model advantage.

The native follow-up is running in session 23921, PID 3646004, on CPUs
8–15,24–31. Its actual launch record confirms learned XY/yaw, the local-view
reference tracker, committed views, stopping-projection selection, unchanged
dispatch guards, no floor reacquisition and one planned assignment. No final
outcome is available yet. The geometric replay function was unchanged when
the separately tested selection/runtime wrapper was added.

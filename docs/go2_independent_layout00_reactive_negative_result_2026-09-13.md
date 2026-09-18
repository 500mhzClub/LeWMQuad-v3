# Reactive negative result on independent development maze 0

The reactive case completed operationally with a negative navigation result.
It stopped at decision 348 with
`NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY`, before either goal arrival.
The final result has `verified_round_trip: false`. The original failed
trajectory and all audit outcomes are retained; there was no retry.

The controller recorded 11 consecutive infeasible observations against a
10-command wait allowance. Its current nominal clearance requirement is 0.45 m.
After a forward command at 336 and a turn command at 337, measured clearance
fell from 0.45142 m at 337 to 0.44570 m at 338, below the cutoff. Zero commands
followed; clearance settled around 0.44419 m. The current surface-intersection
flag remained false, but the nominal clearance gate also blocked the requested
view-acquisition turn. This identifies the recorded stopping mechanism, not a
proof that changing the cutoff or allowing a recovery motion would be safe.

The audit independently found a strict visibility failure at decision 305:
one auxiliary-camera boundary ray failed, with maximum strict error 0.404636 m.
All 4,290 sampled stable interior rays passed, with maximum error 0.00004412 m.
The boundary remains uncertified and the strict failure is not waived. The
observed XY pose error at that frame was 0.00084784 m. These observations do
not establish whether the boundary mismatch caused the later navigation stop.

Raw sensor reconstruction, auxiliary RGB reconstruction, controller replay and
command audit pass. There were two valid outbound cell crossings, zero recorded
contact flags over 18,650 physics samples, and no physical or acquisition stop.
Collection plus audit took 469.36 seconds. Both sampled resource phases pass.
Median observation-plus-control time was 612.09 ms; all 359 intervals exceeded
100 ms. The run remains unqualified for real-time or hardware use.

| Completed case on layout 0 | Outbound goal | Return home | Raw replay | Strict visibility | Verified round trip |
| --- | --- | --- | --- | --- | --- |
| Full-RGB JEPA | Yes | Yes | Pass | Pass | Yes |
| Reactive | No | No | Pass | Fail at 305 | No |

This is one maze, with a comparison of complete methods. It does not isolate
JEPA training, predictive ranking or memory. Both controllers use the same
command definitions: straight forward is 0.20 m/s, arcs are 0.16 m/s with
0.45 rad/s yaw, and in-place turns use 0.45 rad/s yaw. The reactive rule selects
forward or in-place turns; the learned planner evaluates six candidate plans.
There is no forward-speed mismatch. The earlier conversational suggestion of
one was corrected after checking both source paths.

The companion JSON records the final readout, exact stopping timeline,
measurement failure and final artifact identities. Final result SHA-256:
`6e2be5c282b41e1dca52bf6966a8a9bc182b2f2d9f5e0b61273ec669c13003e0`.
The supervised-rollout case automatically started next with
`seed_2026091001_full_supervised_rollout`, corrected state SHA-256
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
Its owner is PID 3149828, creation time 1789261694.22, session 74097.
Direct and nominal-forecast cases remain queued. The original experiment
settings and strict evaluation criteria are unchanged.

# Exact instructed final-target prefix result

Both fixed corrected seed-2026091001 models passed two identical fresh replays
with unchanged model state. All compared observations, map receipts, learned
forecasts, surface checks and nominal first/eight-step checks were exact.
Only the declared final goal target and its existing utility scoring changed.

JEPA first used the exact instructed goal at tick 171. In the observed gravity
map this is [1.1997956694891152, 0], replacing cell centre [1.175, 0.025]. Every
selected exact-goal connector at ticks 171–176 had a completely measured closed
floor-cell supercover and passed the original 0.45-m continuous nominal check.
The first changed command was tick 176: hold [0,0,0] instead of left_turn
[0,0,0.45]. Replay stopped at that observation, after 177 frames, before
executing the changed command.

Direct reproduced its entire recorded 57-observation trajectory, including all
commands, waits and terminal state. It did not reach the exact-goal targeting
stage. Neither method had a controller failure. This result does not establish
native arrival, stopping reliability, independent-maze navigation or hardware
readiness. The 4-cm controller and 6-cm evaluator goal gates remain unchanged.

Five connector tests passed in 0.15 s; the inherited scoring/execution scope
test passed in 1.71 s; native collection/audit scope passed in 2.01 s; readout
comparison boundaries passed two tests in 2.03 s.

Root: `go2_exact_mission_target_prefix_v1_attempt_001`.
The run bound 1,324 source files and took 272.5956959249452 seconds after launch.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | 71bd17d9c7c9f9ee43c94a4185390cb6d4e125e47b0c00010f57173fe33ffee7 |
| seed_2026091001_full_jepa_decisions.json | 7b680dbcf04defc6beac93c8c687b525cdf4043408b8176d9edaf6ef35953f4c |
| seed_2026091001_full_direct_decisions.json | 612c079dbbc8a04980473e6e87cc560aded6907ed30d5f82eb266ee7ad285b79 |
| result.json | 72a0743a1432ac4fadd57c27fe4edead554a5545ae399355b6834e1a3498787a |

# Chained tracking native attempt launched with its actual replay boundary

The original launch invocation passed admission and native serialization,
created its exclusive output root, froze `launch.json`, and spawned one fresh
worker. The emitted launch SHA-256 is:

`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`.

Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_chained_maze02_v1_attempt_001`.

The saved launch SHA, all 2,627 bound source hashes, the recorded boot and
the original live parent's exact creation time and command were independently
checked after launch. Parent: PID 2992412, creation time 1789193084.73,
tool session 69592. Fresh spawned worker: PID 2994743, creation time
1789194027.81. Its parent identity and spawn command were checked. PID 2994742,
with the same creation timestamp, is only Python's resource tracker, not a
second scene worker. Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.

The launch binds the completed chained waiter result
`0a41c3177c2696c86d4b8d21a56ed67baba936e8999184a105f7b804462b494b`
and controller replay result
`9703206c207fbc642164c1972a6eabdc70ebab3f1ced0c2df33c0ad74101eb72`.
It uses the actual completed replay boundary: 3,114 observations, intervention
frame 3113, 156,400 preintervention physics samples, and both command and
terminal-state changes. The original learned native input remains result
`4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18`.

The candidate is `MeasuredPlaneChainedAnchorController`, using unchanged
corrected model state
`56799c99e2bb1fd5e5a591009b931c5b2e04741d3a3744e2346ce9896a2160dd`.
The run retains the 4,000-navigation-tick budget, at most 4,013 commands and
4,014 observations, a 14 GiB collection allowance, the original robot and
physics loop, complete raw controller auditing, and the original image,
floor, temporal, conflict and bridge gates. No single-pass timing change is
adopted. The controller receives no native pose; this remains the reused
development maze 02 with physics paused during computation.

At this registration the worker was live in startup validation. A spawned
worker and a launch file do not establish scene initialization or any completed
physical observation. Continue monitoring the same parent and worker through
startup, collection and the complete raw audit. Authenticate the actual saved
terminal result before accepting navigation success or diagnosing a failure.
Do not restart on a quiet log or observation timeout; preserve terminal
failures and never overwrite this root.

The native result must reconstruct the matched physical/public prefix through
the actual intervention and the candidate command's 50 physical steps. A fresh
round trip, strict visibility and physical settling remain to be established.
No independent-maze, JEPA/memory advantage, real-time, hardware or deployment
qualification follows from launch. The full thread goal remains active.

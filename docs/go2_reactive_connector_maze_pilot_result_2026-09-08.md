# Reactive nearer-route connector native result: navigation failed

Native root: `go2_reactive_connector_maze_pilot_v1_attempt_001`.
Result SHA-256: `d17f634e17f049caf40a0bbcbb65e0300594c1a234cc07f32838df30ecec0d70`.
Launch: `e9d885a0540f4179b9966cb69429cf7d9b036ceae524243129fc1b90500449ab`,
1,477 sources. Session 32876 completed in 418.629936547 s.
Readout root: `go2_reactive_connector_maze_readout_v1_attempt_001`.
Result: `5b625ed72d1b419e489cb50f57c10ae0e3d06e6c8a3de6b88971db4d791f7dc8`.
Launch: `9e04a34b09a1a18615846783cf0f7d1e6ef914ad1fece2cac363a3834154e43b`,
1,481 sources. Readout session 99413 completed.

There were 283 complete commands, 284 paired observations and 14,900 physics
samples. The controller stopped at tick 273 for
`NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY`, followed by ten zero commands.
No physical/acquisition stop, arrival or return occurred. Raw sensor/controller/
command and strict primary/auxiliary visibility checks passed; hard-measurement
failed frames are empty. This is a navigation failure with valid recorded
execution evidence, not a successful arrival or comparison win.

Native path length was 1.9133489139592943 m; minimum goal distance
2.808442892484211 m; terminal distance 2.8167497506607484 m. Two valid open-edge
crossings were [-1,0] -> [0,0] at sample 3924 and [0,0] -> [0,-1] at 13067.
There were no invalid crossings or out-of-maze positions. Maximum observed XY
pose error was 0.0019620821057381986 m.

The physical/public/observer/map/mission prefix matched exactly through all 91
observations, including tick 90 before its changed command. Physical prefix SHA:
`97f34f60a64ed90bc02d1e6171ad2a0cad4579c85e63eb5253d801b692dd8445`.
The nearer-route fallback was used in 77 executed intervals. Its first command
at 90 was forward [0.2,0,0]; actual body XY displacement over that 100 ms was
[0.003792339345135666, -0.0032878354770822303] m. This replaced a blocked
connector with an observed clear connector at the same 0.45 m radius.

The final limitation differs from the predecessor connector stall. Current
nominal clearance at 273 was 0.4460644547176329 m, below 0.45 m, with nearest
occupied cell [13,-13]. The articulated current surface check was clear, but
the current nominal gate correctly prevented fallback. No alternate command
outcome is inferred. The reactive controller used no high-level model or future
outcome forecast, while retaining the same pretrained low-level locomotion.

Selected actions: forward 97, right_turn 94, left_turn 54, no action 26. Median
observation/control time 656.6730435 ms; command-inclusive 688.114535 ms;
receipt-inclusive 689.322512 ms. All intervals exceeded 100 ms with physics
paused, so this is not real-time operation.

This completes six policy attempts on the same development maze: four learned
and two reactive, all without verified arrivals or round trips. No other maze
layout has been executed. Differing complete controllers, predictive constraint
gates and shared persistent memory prevent attribution to learned prediction,
JEPA training or memory from these paired outcomes alone.

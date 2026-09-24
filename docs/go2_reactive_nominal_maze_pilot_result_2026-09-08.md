# Reactive nominal-route baseline: one edge, no arrival

The completed native attempt is `go2_reactive_nominal_maze_pilot_v1_attempt_001`,
result SHA-256
`1a3bf1e0f796d8fe5ae7b0b11feb837b7299064f83f6b9f44457d0040a229dd6`.
Launch SHA-256 is
`a2098de91926d68d3ab37cd7ec963f0a483001dce02baec5a159aa8f71f71520`,
binding 1,462 sources. Session 2205 completed; total post-launch wall time was
244.115 s. No high-level world model was used; the same pretrained PPO locomotion
policy, public sensors, scene, mission budget and persistent observed memory were
retained.

The controller stopped at decision 141 with
`NO_REACTIVE_ACTION_SATISFYING_CURRENT_GEOMETRY`, followed by ten zero drain
commands. All 151 requested command intervals completed, with 152 paired
observations and 8,300 physics samples. No physical or acquisition stop occurred.
Raw sensor reconstruction, fresh-controller command replay, independent command
audit and physical visibility passed; there were no hard measurement failures.
Both prospective physical/public prefix comparisons passed: eight observations
through changed command 7 against original nominal recovery, and four through
changed command 3 against executed-waypoint navigation.

Native traversal crossed one valid open edge, `[-1,0] -> [0,0]`, at sample 3924,
with no invalid crossings or out-of-maze positions. Native XY path length was
1.131608657 m. Minimum and terminal goal distance were 3.101834808 m, with terminal
initial-frame XY [0.994452552, -0.214102581] m. There were no observed arrivals,
native arrival windows or return traversal. The audited round-trip result is
false. This is the fourth native policy attempt on the same single development
maze, adding zero independent layouts and zero verified arrivals/round trips.

The completed readout is `go2_reactive_nominal_maze_readout_v1_attempt_001`, result
SHA-256 `4aeea872d44ec90f93d5ff29ce6e144af4b08add3394db0c94bd006c3593f2af`,
launch SHA-256
`c68bf6a2fdae96cd917c35f42418bf2cd04268a50550d6fb9727d7e50ab69edb`,
binding 1,465 sources. Session 61061 completed. Its paired scene comparison
preserves both audited failures: the earlier predictive controller crossed three
edges with a 4.909109277 m path and minimum goal distance 2.850807224 m; this
reactive controller crossed one. Geometry decision gates differ and both retain
memory, so this single reused-maze comparison does not isolate predictive ranking,
JEPA training or memory benefit, and establishes no generalization advantage.

At the first infeasibility (tick 90), current nominal clearance was 0.467990728 m
and the current surface check passed, but the measured waypoint connector had
only 0.446583058 m clearance. At the final infeasibility (tick 141), current
clearance remained 0.485307661 m with a clear current surface, while the waypoint
connector had 0.448060239 m clearance, below the unchanged 0.45 m radius. Heading
error was -0.097721127 rad, inside the 0.1 rad turn threshold; the connector veto
therefore prevented forward motion. No unknown connector floor cells were listed.
This identifies a reactive local-target/connector limitation; it is not an
executed result for any alternative waypoint or command.

Selected actions comprised 60 forward, 33 right-turn, 11 left-turn and 35
infeasible selections. Maximum observed/native XY pose error was 1.216 mm.
Median observation/control time was 655.365 ms, command-inclusive time 686.900 ms
and receipt-inclusive time 688.123 ms. Every cycle exceeded the 100 ms simulated
control interval while physics paused; no real-time or hardware qualification
is established. The prepared translating view-recovery experiment is next.

# Reactive nominal-route prefix: completed

Root `go2_reactive_nominal_route_prefix_v1_attempt_001` completed with result
SHA-256 `0653ca664495f8113296e0958bd2a1d2a3b2b6e3da536516e7efa70f17f8394a`.
Launch SHA-256 `44c720a314181b5f213ebac95438bb13d2ff85da48687f11b60b9054010ed5f4`
binds 1,435 sources. Decision stream SHA-256:
`3ef01b800d2c0549dffbadcb5031241f04fa3953af4a1614780f38ba94e2c058`.
Session 29906 completed. Replay and final authentication took 15.987856 s after
launch. Five focused successor tests passed in 2.04 s.

All eight public observations reproduced the original causal observer, map,
auxiliary partition and mission. The first command difference was at tick 7:
reactive forward [0.2, 0, 0] versus original learned left arc [0.16, 0, 0.45].
There was no terminal policy difference and all command outputs matched before
intervention. Replay stopped before consuming the unexecuted forward outcome.
No model or candidate future outcome was evaluated.

At the changed decision, observed heading error was 0.046343704 rad and current
nominal clearance 1.851455792 m. Unknown connector cells [0,0] through [8,0] were
recorded and explicitly permitted without being added to observed floor or called
certified free space. Current footprint and known-obstacle connector gates passed.
Those reactive geometry gates differ from learned predicted future gates; this
is a whole-controller baseline definition, not a clean ranking-only ablation.

Post-replay hardware: 76.281 GB RAM available, 51.894 GB artifact space free,
CPU 3.3% busy; the separately owned waypoint native worker remained active.
There are still zero native reactive-baseline executions. The next baseline
step needs a separate native collector and independent raw sensor/command/
mission/backtracking audit with an explicit resource envelope. Neither this
prefix nor synthetic tests establish arrival, navigation or memory advantage.

# Executed-waypoint prefix: completed, no new outcome inferred

Root `go2_executed_waypoint_maze_prefix_v1_attempt_001` completed with result
SHA-256 `5ab4f64ba8e528d55236f8ca6ab995c338e2272be95efebd4ef12192f517e37f`.
Launch SHA-256 `31b075b3fa03d991f8680b6979c5fccbe425c773966ea389029bc9964bb607f0`
binds 1,429 sources. Decision stream SHA-256:
`42f9228a275cb95e837b1974bcfad63648f7fe7f397e1391e92cae166f521c00`.
Tool session 5357 completed. Replay plus final verification took 13.612250 s
after launch; preceding dependency authentication is outside that interval.

Four observations matched the original causal observer, map, mission, residual
state, raw forecast and original constraints. At tick 3, the new intermediate
waypoint score first requested left arc [0.16, 0, 0.45], versus original forward
[0.2, 0, 0]. No terminal policy differed. Other original decision fields matched
before intervention; new scoring metadata was checked against the exact pure
transformation. Replay stopped before reading any outcome of the changed command.

The current residual sample count was zero at this first intervention, so the
change here comes from pose scoring horizon, with unchanged full-plan contact
cost. New forward utility was -0.008371305 m and left arc -0.001128347 m. Original
surface/path gates remain intact; these are model scores, not executed outcomes.
No native progress, arrival, recovery, or planning advantage is claimed.

The fresh JEPA model retained state SHA-256
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`
with no gradients. Eight policy tests passed in 1.69 s. Native source-scope and
existing physical/public prefix-comparison checks passed seven tests in 1.74 s.
Post-replay hardware: 82.700 GB RAM available, 55.263 GB artifact space free,
CPU 0.3% busy, both GPUs idle and no competing experiment recorded. Native
launch must reassess the explicit 10+1 GiB envelope over the 40 GiB reserve.

The full navigation goal remains unachieved; this is one compatibility prefix
on reused development maze 0. All bound sources and artifacts are now frozen.

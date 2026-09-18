# Variable mission controller compatibility replay V1

Execute `scripts/replay_go2_round_trip_controller_compatibility_v1.py` once at its
exclusive `go2_round_trip_controller_compatibility_v1_attempt_001` artifact root.
Bind the completed causal-residual native result
`42211a96a2be46429b1e6cb92872acc033f84573411968d04d0d5ea8afc2c484`
and matched readout
`2c9f30fd78434d87cb5bac8f5ffb2d18faaf59051165601e9338d596c3886087`.

For each of the two fixed native cases, load a fresh unchanged assigned corrected
model and instantiate `ObservedRoundTripController` with the original `[1.2, 0]`
goal, 240 navigation ticks, and return disabled. Reconstruct every recorded
primary policy/RGB-D/fast-sensor packet and calibrated auxiliary public depth.
Require exact equality with every original decision field except the explicit
controller identity. This covers commands, terminal policy, causal observer/map
receipts, raw forecasts, candidate scores, constraint checks and residual state.
New mission fields are additional receipts. Any mismatch terminates the replay;
later observations cannot label outcomes after an intervened command.

One fresh replay per model is compared with its actual executed and raw-audited
native recording. This is not a second native replicate, a longer-budget trial,
or a return/backtracking test. No native state reaches controller inputs and no
model training or native execution occurs.

Use deterministic single-thread numerical settings and one replay worker to
bound memory/storage and avoid duplicating scene or model state. Inspect hardware
before work: require at least 8 GiB available RAM and 256 MiB output headroom
above the unchanged 40 GiB reserve. These are admission allowances, not enforced
OS limits. Independent model cases could use separate processes, but one worker
is adequate for this bounded correctness check; native scene scaling is not
inferred from replay. Record hardware before/after, actual elapsed time, source
closure and exact output hashes. Preserve any failure at the exclusive root.

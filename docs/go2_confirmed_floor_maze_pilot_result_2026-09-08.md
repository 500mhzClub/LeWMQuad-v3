# Confirmed-floor maze pilot: completed failure

Native session 53952 completed, root
`go2_confirmed_floor_maze_pilot_v1_attempt_001`.
Result SHA-256:
`5ee4ef051e1a506f205aae51610deece18f755fb5440c40b1113e22e5ba317ee`.
Launch SHA-256:
`f0486f8eabaf8c0ac857d5c85cd9b9c40f72cb47de815696736bbb2f5cd13f7a`.
Total wall time 3763.926885 seconds; worker 3741.715567 seconds.

The single case, full JEPA on reused maze 0, collected 1547 commands, 1548 paired
observations and 78100 physics samples. It stopped at frame 1537 with
`NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS`, then issued ten
zero commands. There were no observed or native goal arrivals, no return,
and no physical/acquisition stop. Full raw sensor, model and command replay
passed; model weights remained unchanged. Strict primary visibility failed
at frame 909; the hard measurement failure list was empty. Navigation failed.

Audit SHA-256:
`810cd40922c821c3a0ba523048dfa442ef8ba4b0070b7e6bb8ff1a1f57dab283`.
The actual physics/public prefix matched the predecessor for 1483 frames,
through the observation preceding the changed command at 1482. Physics-prefix
SHA-256:
`42f715936bc7144adbd5525cce80676ebe332a80795903eaef858fd79e459973`.
Original visual evidence, raw maps, mission, forecasts and original constraints
also matched. This verifies the declared auxiliary-floor intervention prefix;
it does not supply an unexecuted alternative trajectory.

Completed readout session 89285, root
`go2_confirmed_floor_maze_readout_v1_attempt_001`, result SHA-256
`fd632bdca173ec2ea82be2b2d11f364c90746f01dfabd0d8d93c2be92a449bc8`,
launch SHA-256
`d240b42c6e50edf412bd6b7f42d9c6c5f1b37703ce4aac787fa88740965a1b72`,
1482 bound sources.

- Native XY path length: 7.256963716639283 m.
- Minimum native outbound goal distance: 1.1474975121459456 m.
- Terminal native outbound goal distance: 1.1482243239084386 m.
- Maximum observed-pose XY error: 0.008985761007324095 m.
- Nine open-edge crossings, five edges after removing loops; no invalid
  crossing or out-of-maze position. Return was never entered.
- Forty complete, zero censored intervals selected a candidate whose original
  auxiliary contact check blocked and confirmed-floor check allowed. These do
  not account for every later policy difference.
- First intervention at 1482: left arc `[0.16, 0, 0.45]`; actual 100 ms body XY
  `[0.004806358067816313, 0.0035872758027524453]` m, forecast XY error
  0.0033289164630069257 m. Only the executed command receives this outcome.
- Median observation/control wall time 1093.313 ms; command-inclusive median
  1126.718 ms; receipt-inclusive median 1137.006 ms, for 100 ms physics steps.
  Physics pauses during computation; this is not real-time evidence.

The terminal floor diagnosis remains documented separately in
`go2_confirmed_floor_terminal_provisional_diagnosis_2026-09-08.md`. Its immutable
collection identities are now contained in this completed native result.
The remaining original-height seed dependency explains why the local
classification correction eventually lost current floor evidence despite
coherent measured planes. A separately typed floor-registered pose candidate
has been implemented for prospective replay; the predecessor failure is retained.

Seven native attempts on maze 0 have now completed, all failures. There are
zero verified arrivals/returns, no execution on mazes 1–3, and no timing or
hardware qualification. The full thread goal remains active and unachieved.

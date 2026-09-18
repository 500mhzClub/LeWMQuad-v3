# Sustained-turn native waiter is running

The one-shot waiter was tested and started after verifying both original
prerequisite owners were still live. It has not started a native child or
created the planned native output. Source closure: 2178 paths. Focused waiter
tests: 39 passed in 2.30 seconds, session 2328, exit 0. Preparation/source
preflight session 82908 exited 0.

Preparation record:
`docs/go2_sustained_hold_reorientation_native_wait_preparation_2026-09-11.json`
SHA-256 `575baccaa817af83ba16be4b601b566908b496bfaafba6341bf23d09c938dd08`.

Execution record:
`docs/go2_sustained_hold_reorientation_native_wait_execution_2026-09-11.json`
SHA-256 `bd412ebd14060a16ce8f82e7bca015dd1e420ed5420da6bf6ff55f9926ac5cca`.

Waiter owner: PID 2817601, creation time 1789117418.98, command
`.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/await_go2_sustained_hold_reorientation_maze02_native_v1.py`.
Boot ID `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Tool session 41700.
Output `go2_sustained_hold_reorientation_maze02_native_wait_v1_attempt_001`.
Launch SHA-256 `679d0519e3eb5ddacf5cc3708254de12a117551725c43c0136141e43ce64263e`.
Execution verification session 39771 exited 0 and verified actual process
identity, absence of children/native output, source bindings, original owner
roster, first waiting event and both currently live prerequisite processes.

The exact raw replay remains PID 2813368, creation time 1789115323.91,
session 87754, launch
`6f68017be0f68198ab08af97fb4d90f33ebbdc2b1b56f8da123e8b34b4dd89df`.
Latest observed progress was frame 250 of 407 observations (indices 0–406).
It must finish both model replay and final original-input verification before
its result can be admitted. The budget waiter remains PID 2793505, creation
time 1789096721.36, with its original full queue unchanged.

The contact native worker completed collection and raw audit with no round
trip; its parent PID 2808232 is still verifying the original full inputs.
At the check there was no final contact result or failure. Tracking and budget
native runs remain behind that completion. Do not restart or replace any
original process on a timeout or quiet log. All source paths in the launched
closures remain frozen.

Next: poll the existing raw and queued native owners, authenticate completed
results once they are terminal, and diagnose the completed contact sensing
failure from its bound evidence. The sustained-turn waiter will automatically
perform its one native launch after both original chains complete. Its result
is a reused-layout diagnostic, not independent navigation or hardware evidence.
The overall navigation goal remains active and unachieved.

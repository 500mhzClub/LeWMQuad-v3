# Controller comparison verified; fresh native episode queued

The measured-plane trained-controller prefix completed at frame 122, with 123
observations and 120 paired raw forecast comparisons. Every complete original
decision reproduced the original recording; candidate visual and floor evidence
matched the previously verified full observer history. Both trained model
instances retained the original state. At frame 122 the original controller
requested hold `[0, 0, 0]`, while the candidate requested right turn
`[0, 0, -0.45]`. Neither controller had a terminal or failure at that boundary.
No following observation was consumed after this first changed command.

- Controller result SHA-256:
  `8385e643b776865a44d9271404e8c05a8acc46b37e7ff9bc4b8bf48396e93047`.
- Controller process/session 29653 exited 0.
- Completion receipt:
  `docs/go2_measured_plane_controller_prefix_completion_2026-09-11.json`,
  SHA-256 `ca03a74ce91eba199ae485acc6ee5729872557e428b42e8ef805a99135dc4bbf`.
- Completion checker session 87146 exited 0, with 2,442 source bindings. It
  checked the full output, reconstructed every consumed public packet and
  comparison, and rechecked source/input hashes. It did not rerun model
  inference or visual fitting; model-state assertions are from the authenticated
  frozen runner.

This establishes a real perception-to-controller effect on the same trained
model. It does not establish that the new command succeeds physically or that
navigation has recovered. The fresh simulation is needed for that evidence.

The simulator composition is
`scripts/measured_plane_extended_maze_development.py`. It uses the original
4,000-tick collection and full raw audit function code, with private function
globals substituting only MeasuredPlaneResidualController. The physical-prefix
checker is `scripts/measured_plane_native_prefix_development.py`: it requires
the first 123 fresh decisions to match the prospective candidate replay, the
shared 6,850 physics samples to match before the changed command, and the fresh
run actually to complete that different command. It does not compare following
physics to the predecessor or infer a navigation outcome.

The full launcher is `scripts/run_go2_measured_plane_maze02_pilot_v1.py`, under
`docs/go2_measured_plane_maze02_pilot_v1_2026-09-11.md`. Admission requires the
exact completed controller receipt, original ended native worker and completed
original chained-anchor queue. Queue completion is scheduling evidence and does
not require a successful predecessor navigation outcome. Unrelated training
audits are not repeated. The new episode retains its own complete raw
sensor/model/command audit, physical/contact/timing readout, original sensing
and physical round-trip success gates, and all negative results.

Focused tests, all exit 0:

- Simulator composition: 4 passed, 1.75 s, session 88167.
- Physical prefix and changed-command execution: 10 passed, 2.34 s, session 15844.
- Input and queue admission: 12 passed, 2.17 s, session 4568.
- Native launcher, final version including environment checks: 17 passed,
  2.12 s, session 32136.
- Queue waiter: 3 passed, 2.20 s, session 64002.

The final waiter preflight passed with 2,549 source bindings, session 82096,
exit 0, while the exact original chained-anchor queue owner was live. It checks
the final launcher source, deterministic CPU and original renderer environment,
resource capacity, and existing owners. No native scene was started by preflight.

The new waiter is now registered:

- Source: `scripts/await_go2_measured_plane_maze02_native_v1.py`.
- Protocol: `docs/go2_measured_plane_maze02_native_wait_v1_2026-09-11.md`.
- Root: `go2_measured_plane_maze02_native_wait_v1_attempt_001` under the
  existing navigation development artifact volume.
- Launch SHA-256:
  `b7991218f7dd150900201fb98a62e2d1ddd0dcd81e369420d6746d31dd81a9a5`.
- PID 2908919, creation time 1789159050.26, session 97204.
- Boot `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`; argv is the original environment
  interpreter, `-B`, and the waiter source path.
- 2,549 frozen source bindings. Do not edit those sources/tests/protocols while
  the waiter or its future child remains active.
- Confirmed sleeping on exact original chained-anchor owner PID 2845479,
  creation 1789129072.88. No measured-plane native output root exists yet.

It automatically launches one fresh native child after that exact original
queue owner ends with a verified completed result. The child root will be
`go2_measured_plane_maze02_pilot_v1_attempt_001`. The waiter preserves failures,
does not restart anything, and verifies the child's complete output on exit.

Separately, the old extended-budget native parent ended with exit 0 and produced
result `c92f0bdf5cc8ebb9e513470492578ec3457d02196dfc7238e2f6b12b9c621b27`,
status `NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_PILOT_V1_COMPLETE`, 23,128 artifacts
and zero round trips. Its original waiter PID 2793505 was actively performing
its final verification; its completion was not yet claimed. The preserved
execution order is extended-budget, sustained-turn, contact/flow, chained-anchor,
then the newly queued measured-plane episode.

The full goal is active and incomplete. No successful round trip, independent-
maze reliability, JEPA/planning/memory advantage, real-time performance or real
hardware validation has been established by this work.

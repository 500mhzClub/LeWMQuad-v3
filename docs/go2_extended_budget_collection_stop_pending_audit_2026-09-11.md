# Extended-budget collection stopped; terminal audits still running

At 2026-09-11 17:34 UTC, the original extended-budget launcher and worker were
both live with authenticated boot, PID, creation time and command identities.
The last complete decision-stream timing row was tick 3847. The worker log
already reported collection completion and required terminal auditing.

This corrects the earlier interpretation of the unchanged timing counter:
the worker is no longer collecting further navigation observations. Its log
reports `schedule_terminal: SENSOR_OR_MODEL_FAILURE`, 3847 completed command
ticks, 3848 observations and ten terminal zero-command ticks. It reports no
physical or acquisition stop. The precise sensor/model failure cause has not
yet been established by the terminal audit.

The last logged mission receipt is frame 3836 in phase `RETURN`, with observed
distance to home 2.595780787639278 metres. It records an outbound arrival at
frame 2935. Those are preliminary collection/mission observations: the log
explicitly does not qualify navigation or a verified round trip. No prefix
identity, strict sensing pass or terminal scientific outcome is claimed here.

The worker has written collection artifacts, including the case-level
`result.json`. The root-level result and failure files were absent at the last
check, and the original worker remained live. Do not confuse that case-level
collection receipt with completed worker/launcher verification. Do not restart,
replace or advance the original native queue based on this observation.

- Native root: `go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001`.
- Case: `no_rgb_direct_extended_budget_anchored_maze_02`.
- Launcher: PID 2843773, creation 1789128335.77.
- Worker: PID 2867880, creation 1789139673.31.
- Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
- Original owner record:
  `docs/go2_extended_budget_native_worker_launch_observation_2026-09-11.json`,
  SHA-256 `ed8d0c8a6a98f8de2c9335e0847c8adbf9642adbd216d18a95b3954cd80326ed`.

The original worker source performs artifact binding and the original raw
audit after collection, then writes its worker terminal. Its launcher and
waiter perform their further checks before the queued sustained-turn,
contact/flow and chained-anchor diagnostics can proceed. Preserve this order.

## Independent profiling progress

The owned-child external marker probe completed successfully; its predecessor
failure remains preserved. See
`docs/go2_body_projection_owned_profiler_admission_2026-09-11.md`.
Fourteen focused admission tests passed. The read-only actual input-admission
probe was live at the last check:

- Session 7158, PID 2883977, creation 1789147987.36.
- Execution record:
  `docs/go2_body_projection_external_profile_actual_admission_execution_2026-09-11.json`.
- Execution SHA-256:
  `cd1ee0aea24857295f2a72db24b1ea3df388ad75c8b91cb9033ddece2f7a172b`.

It authenticates the completed body-projection replay in memory without
rerunning its controller or overwriting its completion receipt. Do not treat
the probe as completed until its actual result or failure and ended owner are
checked. A full external controller profile has not launched. The broader
navigation goal remains incomplete.

At 17:35:20 UTC the admission probe completed successfully in 129.81745
seconds. Session 7158 exited zero, and PID 2883977 was subsequently confirmed
ended under its original boot/creation identity. Its result SHA-256 is
`ca94e56b28c99a3b36904b25ebe5d6e76e28ba1e9f3cf90c65117ef9a1a5913b`.
The result and execution hashes and all four explicit new source bindings
were checked after completion. The unchanged original checker reconstructed
the full original completion witness (2376 original source bindings) and
reauthenticated 1428 rows plus raw/model bindings. No controller or profiler
executed in this admission probe. The earlier live observation above is
historical; the native terminal audits remain separate and pending.

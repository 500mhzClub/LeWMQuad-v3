# Chained waiter completed; actual native preflight started

The original chained-controller waiter completed successfully and ended.
Its child replay also ended successfully. The completed replay admitted a
nonterminal candidate at frame 3113, with a current primary-camera visual pose
and a right-turn command where the original controller stopped. See
`docs/go2_measured_plane_chained_controller_prefix_result_2026-09-12.md`.

Waiter root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_chained_controller_wait_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `result.json` | `0a41c3177c2696c86d4b8d21a56ed67baba936e8999184a105f7b804462b494b` |
| `launch.json` | `4e0ed3e7a14c61b8a31ded1c83f34e15643e40bcd7668840bf163bc485311416` |
| `events.jsonl` | `c9d66de56117124380f903149322f9f070ad72d53e5324a1eb64fe4e976adb6e` |
| `replay_stdout.log` | `7fface718c1607fb814f0aac4f1d94a2d558ac948ecf2e6fbfe845c4a1c28cdb` |
| `completion.json` | `9dae00d4f1905f0298fc6e0583dff0efb84b5e3e8e93466d8c0c3ede3ad6c2db` |

Status: `MEASURED_PLANE_CHAINED_CONTROLLER_WAIT_V1_COMPLETE`.
Original owner PID 2930187, creation time 1789170044.92, ended. Its receipt
binds replay result
`9703206c207fbc642164c1972a6eabdc70ebab3f1ced0c2df33c0ad74101eb72`,
learned native result
`4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18`,
and timing waiter result
`ffff577353d432eb5626d94733d0fa26e8b7a709a0228eca552b7d15d4803409`.
It reconstructed all 3,114 consumed rows and public packets, reauthenticated
the original raw artifacts, and did not reexecute the controller replay.
After completion all 2,597 bound sources and four artifact hashes were
independently rechecked, both receipt copies matched, and the original launch
identity and source map matched.

The already tested native launcher, prospective protocol and focused tests
still match the hashes in
`docs/go2_measured_plane_chained_native_launcher_preparation_2026-09-12.md`.
Actual native preflight was started using the completed waiter result:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/run_go2_measured_plane_chained_maze02_v1.py --chained-wait-result-sha256 0a41c3177c2696c86d4b8d21a56ed67baba936e8999184a105f7b804462b494b --preflight-only
```

At registration, the exact preflight process was live: PID 2990617,
creation time 1789192394.74, tool session 2555. It had accumulated 36.1 CPU
seconds. This preflight performs actual completed-input admission and model
checks; it is not merely source preflight. No native output root, worker or
scene had been created. Available RAM was about 78 GiB and artifact space
541 GiB when the preceding waiter ended.

Continue polling this same preflight owner/session. Once it exits successfully
with `CHAINED_NATIVE_PREFLIGHT`, invoke the identical command without
`--preflight-only`. That invocation repeats admission, checks native idleness
and current resources, freezes its own launch identities and starts one fresh
worker. An observation timeout is not permission to restart admission or an
attempt. Preserve any actual failure. No fresh round trip or navigation
qualification is established by either the replay or this pending preflight.

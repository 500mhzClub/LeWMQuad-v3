# Full external controller profile launched

The full 1428-observation replay is running with the unchanged
BodyProjectedTiledController, original source-bound inputs and external
sampling. Source/resource preflight passed with 2391 bindings. Twenty-five
launcher/report tests passed in 4.30 seconds (session 23219). The full synthetic
report also passed review after a JSON serialization round trip (49890).

The parent owns a new Python child and a new pinned profiler. The child
completed actual original-input admission and both readiness handshakes before
controller execution. Last checked completed replay frame was 16, beyond the
first fixed sampling window. This is progress, not a completed profile or
navigation result. Do not read the profile before both subprocesses end.

- Root: `go2_body_projected_external_sampling_v1_attempt_001`.
- Parent session: 88315; PID 2885315, creation 1789148597.78.
- Launch SHA-256:
  `89192077c33ab33bd1eb147fa4877f6e5d2f801c22633071be1897fd2b35d207`.
- Child: PID 2885366, creation 1789148600.92.
- Child execution SHA-256:
  `e3f2a842b40f58c1101315a0cc7104694d6c0de2ea9753db0973f62dbc4f28e0`.
- Profiler: PID 2885638, creation 1789148732.81.
- Profiler execution SHA-256:
  `65d98f434d54d8c95629ec4b3d6493772db791f7dc0e007ef65f6998cd0cc900`.
- Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.

Both subprocesses and the parent were confirmed live using their recorded
creation times and commands. Their execution records are inside the root.
Sources, original failure 1173 and negative qualification scope are preserved.

The completion watcher passed four tests in 2.32 seconds (75046) and is live:
session 34376, PID 2885587, creation 1789148712.18. Its execution record is
`docs/go2_body_projection_external_profile_completion_watch_execution_2026-09-11.json`,
SHA-256 `eb8179ca0b4f8ed267efd3b32bbd21e962eaaa7929731282e032d9b090a48a30`,
2393 bindings. It waits for the actual parent to end, then invokes the already
frozen completion checker once with the actual launch and result hashes.
Do not separately run that checker or restart any attempt on an observation
timeout. The full CPU replay slot is occupied.

The extended-budget native worker remains live in post-collection auditing.
Its preliminary return-leg sensor/model stop is recorded separately in
`docs/go2_extended_budget_collection_stop_pending_audit_2026-09-11.md`.
The original native diagnostic queue remains intact. Final profile evidence
and final navigation evidence are both pending; the broad goal remains active.

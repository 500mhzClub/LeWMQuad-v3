# Observer verified; trained-controller comparison launched

The full measured-plane observer replay completed all 3,838 recorded frames
with 3,837 refined image pairs and no candidate visual or floor failure.
The original observer and floor evidence reproduced exactly, including the
original floor rejection at frame 3,837. The completion checker reconstructed
all consumed public packets and checked the complete output stream, candidate
pose witnesses and floor composition, with source/input hashes before and
after. It did not reexecute feature fitting or refit floor pixels.

Verified identities:

- Observer result: `eec3f8376bad0e47f6fa8a476cfafe05ce6ac0d249f6bc3254f555741c6b00cc`.
- Completion receipt:
  `docs/go2_measured_plane_observer_history_completion_2026-09-11.json`,
  SHA-256 `7942e5ff6cab7a3bcd912b38e1cfaea5cc4cf474df6a9940e337815c5adea92b`.
- Completion waiter result:
  `14ddf01ff4309507c38d09eeba859d1e3c569abdf5c780027e8110d464da7be3`.
  Session 19672 exited 0. Both original owners ended.

This is a positive observation-history diagnostic on a previously executed
development trajectory. It did not select commands or demonstrate a recovered
navigation episode, a round trip, independent-maze generalization or a JEPA
advantage. The existing original global floor and temporal gates were retained.

The next actual model/controller comparison is running under
`scripts/replay_go2_measured_plane_controller_prefix_v1.py` and protocol
`docs/go2_measured_plane_controller_prefix_v1_2026-09-11.md`. Its output root is
`go2_measured_plane_controller_prefix_v1_attempt_001` under the existing
navigation development artifact root.

- Preflight: session 76944, exit 0, 2,440 source bindings.
- Launch SHA-256: `23a08dfaabd751f84aab26c066aac656c46571feb7a029df005c462a2a96913a`.
- Process: PID 2906447, creation time 1789157880.21.
- Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
- Live session: 29653. Frame 50 reproduced both required histories and the
  original complete decisions with no changed requested command yet.
- Resource check: 16 physical/32 logical CPUs, all 32 allowed, CPU 1.9% busy,
  82,115,719,168 bytes RAM available, both GPUs idle, artifact free space
  593,441,222,656 bytes and workspace free space 21,223,301,120 bytes.
- Subsequent live observation: about 2.08 GB RSS; no result or failure yet.

The replay uses two fresh instances of the exact originally assigned corrected
no-RGB direct trained model. It compares complete original decisions, raw model
forecasts when both controllers forecast, and candidate observer/floor evidence.
It stops before consuming an outcome from any changed command. An output
boundary is not evidence that the changed command succeeded physically.

Focused runner/comparison tests: 27 passed in 2.26 seconds, session 19843,
exit 0. They include a poisoned next-observation stream to detect reading past
the changed-command boundary and a model-state mutation failure.

The separate completion checker is prepared at
`scripts/verify_go2_measured_plane_controller_prefix_v1.py`. Its 10 focused
tests passed in 2.36 seconds, session 16415, exit 0. After the exact replay
owner ends, retain any failure; otherwise supply its actual completed result
SHA-256 to that checker once. No controller completion has been claimed here.
No automatic controller-completion waiter has been launched.

The old native launcher remained active in final verification, with its worker
ended and worker evidence admitted separately. Its root completion and existing
native queue have not been advanced by this diagnostic. The full goal remains
active and incomplete.

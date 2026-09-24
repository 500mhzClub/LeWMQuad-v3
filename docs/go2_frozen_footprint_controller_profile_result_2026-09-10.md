# Completed optimized-controller profile

The frozen-footprint controller reconstructed all 405 original observations and
402 forecast-bearing decisions. Both fixed profile windows completed, followed
by the full original input audit. Session 88157 exited successfully. This adds
no native episode or navigation success.

Root: `go2_frozen_footprint_controller_windows_profile_v1_attempt_001` under the
existing external navigation development artifact directory.

- Result: `c636eb55c13f02624b73680295ab3f70d7faac00680d7e66dd820b870cfb9866`.
- Launch: `41b564dae8ec07a43c0c9d23068010b8f520692bb49465afbeb7ca216ba0b6d1`.
- Comparison stream: `56f30e63631e78eb8dd9dc6884681301ad74700a543be764c33fd187db18c4dc`.
- Independent recorded verification:
  `docs/go2_frozen_footprint_controller_profile_verification_2026-09-10.json`,
  SHA-256 `bc432a67568521ee173057ecd6b0fda714c8281ca88480e203d4df439dafea56`.

All 2,008 source bindings and six output artifacts were independently rechecked.
All 405 original decision, complete candidate decision and public-input hashes
match the completed paired frozen-footprint replay. Both raw profiler files
reconstruct their function and exclusive-module summaries; their ten recorded
observations and timings match the comparison stream. The independent check
does not rerun inference, reload sensor packets or reconstruct hidden state.
Those execution checks belong to the completed original run and paired replay.

| Window | Observations | Total exclusive profiled time | Selected remaining cost |
| --- | --- | --- | --- |
| Early navigation | 3–12 | 12.104811289 s | Floor-index construction: 40 calls, 3.400738634 s cumulative |
| Repeated discretionary hold | 395–404 | 19.925995902 s | Retained floor-patch module: 2.761116956 s exclusive; copying module: 3.559314480 s exclusive |

The early copying-module exclusive time is 1.346417374 seconds. Repeated-hold
footprint queries still cost 10.858765720 cumulative seconds for 180 calls in
the original later-floor memory implementation. Its callers overlap with this
cost; cumulative times must not be summed. These measurements include profiler
overhead on a shared host. They do not replace the previous unprofiled paired
result: 13.50% lower total controller time and a 663.82 ms median, still above
the 100 ms command interval. Acquisition is excluded from these profiles.

Raw profile identities:

- Early `.prof`: `e06357c5825b1b0c8734d419b3740877868edc74edb20bf11ceb00e048432ae3`.
- Early `.json`: `fa9642f12783f9004fc1cc33224685cd953870e71b0b088000dac4c59612464d`.
- Hold `.prof`: `32bd4983ea64a1262e8f7517c645ae10c327e7be07f61a1ffa42076d5de36651`.
- Hold `.json`: `95af76726d29af89e8a34b6054139184151ff8babcd9ddf850766cabe3b31f97`.

Reported wall time is 1,310.5481513689738 seconds after initial admission,
including final input verification. It is not controller latency.

The next candidate reuses exact depth-mesh intermediate arrays within each
observation. Its synthetic checks and helper timing are recorded in
`docs/go2_reused_floor_mesh_v1_2026-09-10.md`. A complete paired raw replay is
required to assess full-controller performance and exact model-driven decisions.
The existing simulator queue remains unchanged. Real-time, independent-layout
navigation and hardware qualification remain unestablished.

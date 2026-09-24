# Scoped footprint replay completed

The original paired replay completed and its tool session 5161 exited 0.
Result SHA-256:
`b7d80d3d98f9c4d0be92ab3a91d5d92f86d1ec3a5b78add2e078cb71a630670a`.
Its original owner, PID 2754886 with creation time 1789077365.71, ended.

Completion authentication checked all 2,145 original source bindings and both
output bindings, all 1,428 ordered comparison records, all 1,425 forecast
comparisons, the unchanged model identity, seven recorded retained-state
checkpoints, negative sensing scope and independently reconstructed timing
summaries. The admission uses the prepared next replay's 2,153-source union.
It authenticates saved evidence; it does not rerun the model, controller or
full original raw-input admission. The original completed worker performed
its own full input admission before and after replay.

Verification record:
`docs/go2_scoped_footprint_late_history_replay_completion_verification_2026-09-11.json`,
SHA-256 `98f44dba3ae9ffb6b84630d391aeea3e4b6a383212ed4efff2c171405f2ba061`,
session 9217, exit 0.

| Window | Controller time reduction | Baseline median | Candidate median |
| --- | ---: | ---: | ---: |
| All 1,425 navigation observations | 15.29% | 1.553 s | 1.279 s |
| Early frames 3–12 | 2.54% | 0.761 s | 0.825 s |
| Repeated hold frames 395–404 | 9.99% | 1.118 s | 0.980 s |
| Late frames 1418–1427 | 17.53% | 2.457 s | 2.059 s |

All navigation controller calls still exceeded 100 ms. Early median latency
regressed despite a small reduction in that window's total time. These are
alternating-order controller-only measurements on a shared host; acquisition
is excluded and no isolated benchmark or real-time qualification is claimed.
Original visibility failure at frame 1173 and the original failed round trip
remain in the evidence. No new physical command was executed.

The prepared combined scoped-plus-batched replay was then started against
this exact completed result, to measure the incremental batching effect.

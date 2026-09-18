# Timing queue completed; chained controller replay launched

The original full-history timing waiter completed its verification and ended.
The existing chained-controller waiter then launched its registered child.
No replacement attempt or native scene was started.

## Completed timing waiter

Root: `go2_measured_plane_full_history_timing_wait_v1_attempt_001`, under
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

| File | SHA-256 |
| --- | --- |
| `result.json` | `ffff577353d432eb5626d94733d0fa26e8b7a709a0228eca552b7d15d4803409` |
| `launch.json` | `c63f9bccb0cdfc273b1d3fd721310da5fac384f1350dcfc4625098fd103184a4` |
| `events.jsonl` | `f4105f849725a178bf9d16f2857c3944336c76821852837544fd5c241f23e90a` |
| `replay_stdout.log` | `826534823d64fa13878b1917fb5f33b9966ab7ac9950ac14bff6503eb2a09fbd` |
| `completion.json` | `f817e4aea4914fbaa1ffc074c23367e249676095472bdd436bc8ccf62b138add` |

Status: `MEASURED_PLANE_FULL_HISTORY_TIMING_WAIT_V1_COMPLETE`.
Original owner PID 2924370, creation time 1789167167.42, ended. Its saved
completion receipt confirms reconstruction of all rows, state checks and
public packets without rerunning controller inference, and binds timing replay
result `094519edfa684a38ccf144644c7d3c5bc4f14c67b3c6cf660737d6c627ef7cc0`.
All 2,586 source bindings and four result-bound artifacts were independently
rehashed and matched. The result's report equals its saved completion receipt.
No failure was present.

The completed timing result and its limits are documented in
`docs/go2_measured_plane_single_pass_full_history_result_2026-09-12.md`:
53.1218% lower total controller-observation time with complete fixed-history
decision equivalence, but no real-time or navigation qualification.

## Live chained controller replay

Root: `go2_measured_plane_chained_controller_prefix_v1_attempt_001`.
Launch SHA:
`ddf403f8bd376f8db22a2e5fc88e3b45f816bf8091f9c73619fef1490f96fbc2`.
Original child PID **2963794**, creation time **1789182926.58**, was verified
live with the exact command:

```text
/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/replay_go2_measured_plane_chained_controller_prefix_v1.py --learned-result-sha256 4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18 --timing-waiter-result-sha256 ffff577353d432eb5626d94733d0fa26e8b7a709a0228eca552b7d15d4803409
```

All 2,594 child source bindings were verified and matched the original queued
parent's source closure. The launch retains the maximum 3,124 observations,
2 GiB output cap, first-changed-command-or-terminal stop, no native execution,
and no automatic retry. Its initial resource record at frame 0 was present.
The original parent waiter PID 2930187, creation time 1789170044.92, remains
live. The child had accumulated 75.1 CPU seconds at the final check, with about
75.5 GiB memory and 541.8 GiB artifact space available. No result or failure was
present yet. The resource log records progress every 100 frames and at stop;
its last logged frame is not a count of every processed observation.

Next, monitor this same child and parent. The completed replay must establish
its actual intervention and pass its full consumed-input/output checks. Only
then can the prepared `run_go2_measured_plane_chained_maze02_v1.py` be admitted
using the actual completed chained waiter result SHA. Its native attempt root
is still absent. Neither a live replay nor the earlier two passing pair fits
establishes tracking recovery or a completed round trip.

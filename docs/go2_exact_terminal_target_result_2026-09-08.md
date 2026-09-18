# Exact terminal target changes no action on the recorded approach

All 46 recorded selections were checked. Nine selected the terminal observed
goal-cell centre and were rescored against the exact mission point. Zero chosen
actions changed. Every model forecast, surface check, nominal check and observed
route was preserved exactly. The five focused tests passed in 1.66 seconds.

The correction is therefore not adopted for a fresh native run on this evidence.
The old controller's failure remains unchanged. At tick 228 its goal cell had
already become unavailable to the conservative grid route, so the exact-terminal
rule correctly did not change the frontier target. This diagnostic neither
changes that routing condition nor demonstrates an alternative executed outcome.

The result binds 1,028 source files. Preflight recorded 82.00 GB available RAM,
84.82 GB artifact storage free, 0.4% CPU utilization and idle GPUs. The small
46-context computation ran sequentially; post-launch work took 12.423 seconds.
No native execution, training or model inference occurred.

Artifacts in `go2_exact_terminal_target_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `b9a31721945d9fc52257d2aad28450142ad7067f644648cd196bf2be19919fb2` |
| `contexts.json` | `a8299195529b86c196eb0b8000f5b642b1f891ff5db330ee2c82abb69b607fb8` |
| `result.json` | `a6747830853d0b3c35abd2aeedc7d1a1925357f99a8e5b2fab40a4129e75c199` |

The next useful diagnosis is correspondence loss on the final near-wall view.
The accepted corner population fell from 397 at frame 198 to 70 at 223 and 50
at 228. All current depth pixels remained valid at 228–229. The saved frame-229
image contains large, sparsely cornered wall patches. The existing matcher
requires mutual SIFT ratio association before bidirectional optical flow, so
failure could occur before otherwise usable image tracks reach the rigid fit.
That remains a hypothesis requiring measured pair-stage counts.

Inspect the frozen prior gyro/previous-measured-pose flow diagnostics before
designing a distinct corner-feature comparison. Any new comparison must retain
current-image/depth measurement, unchanged rigid registration/gyro/increment
gates, complete failures and separate native evaluation. It must not replace
the failed observer's pose or resume the native run. Near-goal prediction errors
and the foot-surface veto on hold at tick 223 also remain unresolved. The full
end-to-end navigation goal stays active, with zero verified native goals.

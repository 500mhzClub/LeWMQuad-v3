# Verified perception timing and next replay

The density-routed full-controller replay completed and its completion checker
passed. Across the same 1,425 navigation observations, paired total observe time
fell from 1,006.076600 s to 902.450946 s (10.29997658524292%). Median observe time
fell from 672.074 ms to 598.603 ms. All 1,425 observations in both arms exceeded
100 ms. This is shared-host CPU timing, excluding sensor acquisition, not
real-time or navigation qualification.

All 1,428 original input/decision comparisons, 1,425 model forecasts and seven
retained-state witnesses passed. The original sensing failure at observation
1173 remains. No physics was rerun and no navigation controller was promoted.
The completion checker rebuilt every timing population and the whole report,
and reauthenticated original raw/model bindings; it did not independently
execute the controller or full training ancestry again.

- Result: artifact root `go2_density_routed_floor_late_history_v1_attempt_001`,
  `result.json` SHA-256
  `cc1680b8922273d809fd2663c9f39be967638dfd9757858025c89b8b22cc2420`.
- Completion: `go2_density_routed_floor_controller_completion_verification_2026-09-11.json`,
  SHA-256 `6cc04190aea13972ff2a6236dc510170d2cc458672d07df26345a743170ff370`.
- Original replay session 59005 and completion-checker session 16882 both exited
  zero. Original replay owner 2851226, creation time 1789131561.6, has ended.

The next candidate preserves the original first 32-frame patch projection batch
and uses 128-frame batches for subsequent history. Its four synthetic cases
retained identical complete receipts. Paired total times improved by 22.72%,
20.25% and 5.78% for invisible, sparse-visible and visible-uncovered histories;
immediate-witness time regressed 0.31%. Preserve the earlier wide-batch result,
including its 146.93% immediate-witness regression.

- Microbenchmark: `go2_progressive_batched_retained_patch_microbenchmark_2026-09-11.json`,
  SHA-256 `f3e5b7921fcf62a286129e6d0952626c3dd6132a5d9c49c08f0dc273691be82b`.
- Component tests: 79 passed. Controller integration: 10 passed. Full replay
  harness: 16 passed. Completion-checker logic: 18 passed.
- Protocol: `go2_progressive_batched_floor_late_history_v1_2026-09-11.md`.
- Preparation: `go2_progressive_batched_floor_controller_preparation_2026-09-11.json`,
  SHA-256 `511e69fa65e633296489cfad3c08d37a60a56baa668094e96a1096f89ef8b4eb`.
- Live replay: session 69904, PID 2856701, creation time 1789134105.22,
  boot `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Command:
  `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/replay_go2_progressive_batched_floor_late_history_v1.py`.
- Artifact root: `go2_progressive_batched_floor_late_history_v1_attempt_001`.
  Launch SHA-256 `fc7c80e9d9ed24c48e9f89b23afd8a7625a637a43717e3f908b8c09af7f79cb0`;
  2,315 source bindings. Original input admission and first paired frame passed.
- Execution: `go2_progressive_batched_floor_controller_replay_execution_2026-09-11.json`,
  SHA-256 `c7b73ff683ba557843418450deafae7e841f099c3095b3135ea824a0065d9da4`.

Next, monitor that exact owner without restarting it. Once it ends, preserve
any failure. For successful completion, compute its actual result SHA-256 and
run `scripts/verify_go2_progressive_batched_floor_controller_completion_v1.py
--result-sha256 <actual SHA-256>` in the original single-thread environment.
Whole-controller gains from this candidate are not yet established.

The extended-budget navigation launcher 2843773, creation time 1789128335.77,
remains active in nested input verification, with advancing CPU and read
counters and no native child yet. Roughly 564 GiB artifact space is available.
Preserve the queue: extended budget, sustained turn, contact plus direct flow,
then chained anchors. Keep these source-bound navigation trials unchanged.
After their actual outcomes, use the prepared eight-diagnostic review before
selecting any independent 32-case population definition. No verified round
trip, independent-maze advantage, real-time qualification or hardware success
has been added by this timing work.

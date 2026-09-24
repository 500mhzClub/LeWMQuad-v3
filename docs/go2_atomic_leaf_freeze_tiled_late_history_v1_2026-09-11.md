# Primitive-leaf freezing paired controller replay

Compare `TiledDensityProgressiveFloorController` with
`AtomicLeafFreezeTiledController` on all 1,428 original development observations,
with 1,425 forecasts and retained-state witnesses at frames 3, 12, 395, 404, 1173,
1418 and 1427. The only algorithmic change is primitive-leaf receipt freezing.
Both arms retain the original cached clone, tiled floor geometry, progressive
retained-patch queries, model, sensing, map, history, mission and recovery rules.

The baseline is bound to result
`9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307`,
launch `8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc`
and completion `543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556`.
The completed full-history profile motivates the intervention; its result and
source bindings are inherited through the checked synthetic component benchmark.
Preserve the benchmark's clone regressions: the proposed clone implementation
is not used in either controller arm.

Runner: `scripts/replay_go2_atomic_leaf_freeze_tiled_late_history_v1.py`.
Exclusive output: `go2_atomic_leaf_freeze_tiled_late_history_v1_attempt_001`
under the existing navigation development artifact root. No retry or resume.
The original full-history loop is reused with private controller and normalizer
bindings in `scripts/atomic_leaf_freeze_tiled_replay_development.py`. No imported
module globals change. Normalization removes only the declared new identity;
the original ten retained-state type paths remain unchanged. Selector class
identity is outside the original retained-state comparison scope.

Before execution require the ended original baseline owner on its recorded boot,
unchanged full source bindings, complete baseline result/report/timing/row
reconstruction, actual original raw sensor and assigned model input admission,
64 GiB available RAM, 41 GiB artifact space and four physical CPUs. Keep one full
CPU replay plus one native scene. Source-only preflight creates no runtime output
and performs no raw/model admission. The prior profile owner and watcher have
completed, leaving the full CPU replay slot available.

Every original public packet, full baseline/candidate decision, model forecast,
requested command, model weight and absent gradient must remain unchanged. Use
separate model instances and tensor storage. Execute both controllers on each
observation in alternating order. Preserve all 1,425 navigation timings, fixed
early/hold/late windows and all over-100-ms counts, including any regression.
Only controller observe is timed; normalization and input hashing remain outside
the interval. No cProfile instrumentation is enabled. Shared-host interference
remains, and component percentages must not be added to controller timings.

After execution reauthenticate sources, raw/model bindings and the complete
baseline. The completion checker must require the exact ended execution owner,
actual result and execution SHA arguments, all paired rows, full report and
seven original state witnesses, and unchanged negative sensing scope. It must
preserve failure 1173 and make no real-time or navigation qualification claim.

This recorded-history timing study executes no new native physics and changes
none of the queued native controllers. It cannot establish a round trip,
independent-maze reliability, predictive-planning advantage or hardware readiness.

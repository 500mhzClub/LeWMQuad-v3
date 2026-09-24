# Freeze-only paired replay and extended-budget native execution active

This follows `go2_atomic_leaf_freeze_tiled_controller_prepared_2026-09-11.md`.
The checked freeze-only controller now has a running full-history paired replay.
The original extended-budget native worker has also completed its input checks
and begun collecting actual observations and commands. Neither has a completed
or qualified new outcome yet.

## Freeze-only paired controller replay

The candidate retains the original cached clone; the regressing alternative
clone from the synthetic benchmark is not adopted. The paired loop and all
original input, command, forecast, model-state and retained-state checks remain.
No native controller or policy is changed by this timing replay.

- Runner: `scripts/replay_go2_atomic_leaf_freeze_tiled_late_history_v1.py`.
- Root: `go2_atomic_leaf_freeze_tiled_late_history_v1_attempt_001`.
- Owner: PID 2871712 / creation 1789141465.68, session 17139; original boot
  `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Exact argv is in the execution receipt.
- Launch SHA-256: `c8c962c7e38702fcea4e2bbf68b3fc6b5f512242f2c77fd2e5133b31601c060c`,
  2,368 sources. Full input admission passed. Latest bounded complete comparison
  row was frame 134, with no failure document. The required population is 1,428.
- Execution: `docs/go2_atomic_leaf_freeze_tiled_controller_replay_execution_2026-09-11.json`,
  SHA-256 `1fc710a719a81652af92199734196fad0f37aba15c838de33c6864ba5ca2899c`.
- Invocation: `3bab466853d659d699581786f55cb0e90b44725aedd6ef020611b9d4f6360e9c`.
- Harness: `scripts/atomic_leaf_freeze_tiled_replay_development.py`.
  Sixteen tests passed in 4.65 s, session 68507, covering all 1,428 synthetic
  paired observations, separate model/storage ownership and corruption rejection.
- Harness preparation: `eebd974238e1df486a3df4d0854e20cb81bd4cb53ef941c68726320948c914a7`,
  2,364 sources.
- Admission tests: 22 passed in 2.48 s, session 48909. The actual baseline
  report, timing rows and negative sensing scope must reconstruct before launch.
- Source/resource preflight: session 99972, exit zero, 2,368 sources; about
  73.5 GiB RAM and 562.9 GiB artifact space at preflight.
- Launcher preparation: `b72ce05af1b374b4846e51557accef8daa669e0de14998da9188105411370962`.

## Automatic completion verification

- Checker: `scripts/verify_go2_atomic_leaf_freeze_tiled_controller_completion_v1.py`.
  Requires actual `--result-sha256` and `--execution-sha256` arguments. Eighteen
  tests passed in 2.28 s, session 98206. The real execution/owner gate also
  rejected the live replay before result access in session 66821; no completion
  receipt was written by that check.
- Checker preparation: `042ac3780becff22acd20ec283aaebcdca8796957423a48d3ca4234f05b2a5cf`,
  2,371 sources.
- Watcher: `scripts/await_go2_atomic_leaf_freeze_tiled_completion_v1.py`.
  PID 2872194 / creation 1789141631.66, session 59978, already active.
- Watch execution: `docs/go2_atomic_leaf_freeze_tiled_completion_watch_execution_2026-09-11.json`,
  SHA-256 `eaab6377ceec49cc55d102f4473c4c78f4518d86ec90bcf2841098278f895458`,
  2,374 sources. It recorded the actual admitted launch and will invoke the
  checker once after the original owner ends, rejecting failure or missing
  results. Do not start a duplicate checker or another full CPU replay.

The completed tiled baseline remains result
`9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307`,
completion `543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556`.
Its verified median was 536.584334 ms, and every measured navigation decision
exceeded 100 ms. No additional full-controller speedup is established yet.
Preserve the original sensing failure at frame 1173 and all seven state witnesses.

## Native collection has begun

Original launcher 2843773 / creation 1789128335.77 still owns worker
2867880 / creation 1789139673.31. Its native root is
`go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001`, launch
`7380fbd8c83306dfe544fd9760e13a9eb8a0b70cb93ade06e1972096044c8113`.
The original no-RGB direct model and 4,000-tick budget are unchanged.

The episode `no_rgb_direct_extended_budget_anchored_maze_02` now exists. A bounded
read of the first complete `context_decisions.jsonl.gz` row confirmed tick 0,
pre-sample index 749, no terminal condition and the original controller identity.
Only that complete first row was read from the live compressed stream; the
growing stream was not hashed or read to EOF. The latest complete line of
`decision_stream_timing.jsonl` was tick 570. This is live collection progress,
not an independently audited physical prefix, arrival, round trip or timing pass.
No worker-terminal record exists yet. The frozen collector pauses physics during
controller computation, so collection progress does not establish real-time use.

Latest available resources were approximately 71.05 GiB RAM and 562.11 GiB
artifact space. Keep one native scene plus the current full CPU replay. The
downstream queue remains sustained turn, contact/flow and chained anchors.

Next, monitor these exact owners and the paired-replay completion watcher. Once
the timing comparison verifies, interpret all timing populations including any
regression. Once native collection and its full raw/model/prefix audits finish,
retain its actual result regardless of success and let the original downstream
queue proceed. The eight-diagnostic final review and independent-layout study
remain pending. There is still no verified round trip, independent-maze
reliability, predictive-planning advantage, real-time qualification or hardware
result. The full goal remains active.

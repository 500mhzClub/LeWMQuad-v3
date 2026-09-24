# Tiled controller verified; complete-history profile V2 active

The paired tiled dense-geometry replay and its automatic completion checker both
finished successfully. Across all 1,428 observations, every complete original
decision and public-input identity matched; 1,425 forecasts and all seven
retained-state witnesses were preserved. The original sensing failure at frame
1173 remains. This adds no native physics, navigation success or qualification.

Total controller time decreased from 937.2314236194361 s to
809.5831959927455 s: **13.6197127422098% reduction** in this paired run.
Median controller time decreased from 623.237213 ms to 536.584334 ms.
All 1,425 navigation decisions in both arms still exceeded 100 ms. The late
ten-frame median decreased from 782.124351 ms to 665.723704 ms. Shared-host
timings remain; do not add this percentage to gains from other paired runs.
The tiled controller has not been adopted into the already-frozen native queue.

- Replay root: `go2_tiled_density_progressive_floor_late_history_v1_attempt_001`.
- Result: `9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307`.
- Completion: `docs/go2_tiled_density_progressive_floor_controller_completion_verification_2026-09-11.json`,
  SHA-256 `543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556`,
  2,340 sources. Sessions 99411 and 57399 both exited zero. Original replay
  owner 2862056 and completion watcher 2862653 ended. Do not repeat the checker.

## Profile V1 admission failure preserved

V1 profile invocation 2866232 / creation 1789139036.53, session 22960, exited
one during admission, before output creation or controller execution. Its
read-only capture of the frozen completion checker received the receipt, then
the checker's final print evaluated `digest(OUTPUT)` on the unwritten capture
destination. The missing-file failure is an implementation error in the
profiling wrapper, not a failed tiled-controller comparison.

- Preserved failure: `docs/go2_tiled_density_progressive_floor_profile_v1_admission_failure_2026-09-11.json`,
  SHA-256 `038ea8fe55b38e7a6d7d68744423a558dbd69e417f7bef5efa6bc85b1673dde7`.
- Original invocation: `6e5837ddd185d231b6c545a6ec668798b209489c5854468589ffb6b2319e57c8`.
- Original preparation: `e4ab0d28ff4b5ef126ee305b48743fa09bf8ba64ce4f00d9e22e1503dda11d8c`.
- Original unused checker preparation: `674bc4a368bc3ea626ec5e8e7b1946f8557e1dcc4f1383ca210ca83549323bb9`.

Preserve these files and the V1 runner, tests, protocol and checker unchanged.
Do not retry that invocation or overwrite its absent-runtime-output history.

## Profile V2 executing

V2 fixes only the capture-destination logging digest. It serializes the captured
receipt exactly as the original JSON writer, including its final newline, and
delegates all other hashes to the original helper. Tests execute the formerly
failing final log line, check exact serialization and ordinary-hash delegation.
All original completion-checker admission checks remain in its unchanged body.

- Runner: `scripts/profile_go2_tiled_density_progressive_floor_late_history_v2.py`.
- Root: `go2_tiled_density_progressive_floor_late_history_profile_v2_attempt_001`.
- Owner: PID 2866966, creation 1789139321.48, session 29134, original boot
  `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Exact argv is recorded in execution.
- Launch: `5cc9f2d13def63a2e46a67db939767beda744a508e884cb543c46ac1ccd8a83f`,
  2,349 sources. Full original input admission passed; frames 0 and 50 printed.
- Execution: `docs/go2_tiled_density_progressive_floor_profile_v2_execution_2026-09-11.json`,
  SHA-256 `d7b996e8c3f81a51cb73a67c00c82101a4d8df6ad8e3d6a66d64413d0e1ca55f`.
- Invocation: `83b6bc951879b869fbaa00e3d27dbf931f94b0bbe7bbf4f696a4d4270f544ce7`.
- Preparation: `5ce910fa3542ccd18c84a590b985941d6066762173cefa797a6acd38f0a845a1`.
  Profiler tests: 31 passed in 2.12 s, session 29639. Source preflight session
  26380 exited zero, 2,349 sources, about 76 GiB RAM / 563 GiB artifact space.
- Checker: `scripts/verify_go2_tiled_density_progressive_floor_profile_completion_v2.py`.
  Requires actual `--result-sha256` and `--execution-sha256`. Tests: 38 passed
  in 2.16 s, session 16159. Actual frozen execution/owner rejection also passed
  in session 67044 before any profile-result read; no completion was written.
- Checker preparation: `9e11005a039dab0061a614edf9c0a72d7aa5a59c9f4f4e92f942aa860ad73ad3`,
  2,352 sources.

The automatic watcher is already active. It recorded the actual launch and will
invoke the prepared completion checker once, only after the original owner ends
and in the absence of profile failure. Do not manually duplicate that checker.

- Watcher source: `scripts/await_go2_tiled_density_progressive_floor_profile_v2_completion_v1.py`.
- Watcher PID 2867206 / creation 1789139457.92, session 80413.
- Watch execution: `docs/go2_tiled_density_progressive_floor_profile_v2_completion_watch_execution_2026-09-11.json`,
  SHA-256 `fd6310d295f010747b19b06effdd6f9352a26ca61c0de617f48f2500ca52242e`,
  2,355 sources. Poll interval 15 s; no retries.

The profile executes the full 1,428-observation causal history and profiles only
the original fixed windows 3–12, 395–404 and 1418–1427. Every decision must match
the completed tiled replay; model state and raw/input identities are retained.
The saved `.prof` files must reconstruct all three JSON summaries. Profiler
overhead is included; acquisition is excluded; cumulative times overlap. This
profile identifies remaining costs and cannot establish an additional speedup.

## Native queue and next work

Extended-budget owner 2843773 / creation 1789128335.77 remains live in repeated
nested input verification with no native child. Latest checked CPU time was
7,634.83 s and physical reads 3,472,924,114,944 bytes, both advancing. This is an
active expensive admission, not a storage failure or terminal job. Preserve its
frozen source and original queue: extended budget, sustained turn, contact/flow,
chained anchors. No native outcome has been added.

Next, monitor the exact profile owner and automatic watcher. After completion
verification, inspect exclusive costs in the new profile to choose the next
substantial timing change; avoid extrapolating old component profiles. Keep one
full CPU replay plus one native scene. Continue the native diagnostic queue and
the pending eight-diagnostic review before any independent-layout execution.
No verified round trip, independent-maze reliability, JEPA/planning advantage,
real-time qualification or hardware result has been established. The full goal
remains active.

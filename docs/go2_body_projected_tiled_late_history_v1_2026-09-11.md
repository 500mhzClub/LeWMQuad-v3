# Paired observation-local body-projection controller replay

Compare `TiledDensityProgressiveFloorController` and
`BodyProjectedTiledController` on all 1,428 original development observations,
with 1,425 forecasts and retained-state witnesses at frames 3, 12, 395, 404,
1173, 1418 and 1427. Both arms retain original tiled floor classification,
receipt handling, registration, map evidence, retained patches, model,
feasibility checks, mission and recovery rules. The sole algorithmic change
reuses identical body-coordinate projection expressions within a map observation.
Neither arm uses the separate atomic-leaf-freeze candidate.

The baseline is bound to result
`9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307`,
launch `8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc`
and completion `543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556`.
Controller and harness preparation is bound to
`78cb96402369a6739b4936364a21de73e72a65ccbc0b0d8e5e415efb9ef6db15`.
The synthetic component benchmark is bound to
`70560685bb2697ec9d4ffcc5b2f015d4063daca9b5e099d9a622cfdef924eaba`.
Its component timings are motivation, not integrated performance evidence.

Runner: `scripts/replay_go2_body_projected_tiled_late_history_v1.py`.
Exclusive output: `go2_body_projected_tiled_late_history_v1_attempt_001`
under the existing navigation development artifact root. No retry or resume.
The original full replay loop is privately bound to the two controllers in
`scripts/body_projected_tiled_replay_development.py`. Public normalization removes
only the declared controller identity and body-projection flag; original
normalizations and ten state type paths remain. Selector class identity is
outside the original retained-state comparison scope.

Source preflight verifies frozen source and preparation bindings and resources,
but creates no runtime output and performs no raw/model admission. Actual
execution additionally requires the recorded atomic-leaf replay owner and its
completion watcher to have ended on the recorded boot; both execution documents
are SHA-256-bound. Check that slot before raw input admission and again before
creating the output. Preserve one full CPU replay plus one native scene.
No atomic-leaf result is used to select this independent comparison.

Require the ended baseline owner; complete baseline artifact/report reconstruction;
original raw sensor and assigned model input admission; unchanged source bindings;
64 GiB available RAM, 41 GiB artifact space and four physical CPUs. Use fixed
single threads and deterministic model operations with separate model instances
and tensor storage. Alternate baseline/candidate order per observation.

Require complete original public inputs, both full decisions, all forecasts,
model weights and absent gradients to remain unchanged. Preserve all 1,425
navigation timings, early/hold/late windows and over-100-ms counts, including
regressions. Time only controller observation, with normalization and input
hashing outside the interval. No profiler is enabled; shared-host interference
remains. Do not add component or previous-run speedup percentages.

After execution repeat actual raw/model admission, source checks and full
baseline reconstruction. Completion requires a separately checked verifier using
the actual result/execution SHA values, ended owner, all paired rows, reconstructed
complete report and seven state witnesses. Preserve sensing failure 1173 and all
negative scope. This recorded-history experiment changes no queued native
controller and proves neither navigation, independent-maze reliability, predictive
advantage, real-time execution nor hardware readiness.

# Verified receipt-freezing result and active body-projection replay

The atomic-leaf-freeze replay and its automatic completion verifier both ended
successfully. Sessions 17139 and 59978 exited zero; replay owner 2871712 and
watcher 2872194 have ended. Do not rerun their verifier.

- Result: `cf6ae19d47133bf9e72cc11951818939215fb961211835ce740f5cf465aa2425`.
- Completion document:
  `docs/go2_atomic_leaf_freeze_tiled_controller_completion_verification_2026-09-11.json`,
  SHA-256 `5fbea2df7a4505936e6d4ee461144dec7ec62cd73be099294f598036154f5af3`,
  2,371 source bindings.
- Completion-watch result:
  `c65acb85efff1cf51e78fee3ef6094fd08830f46749b048acb50859fba187b1b`.

All 1,428 original inputs and decisions, 1,425 forecasts and seven original
retained-state witnesses were preserved. The verifier reconstructed the complete
report and timing populations and reauthenticated the actual raw/model inputs.

| Window | Baseline total seconds | Candidate total seconds |
| --- | ---: | ---: |
| All 1,425 navigation observations | 832.334480 | 831.412655 |
| Early ten observations | 4.527601 | 4.470837 |
| Repeated-hold ten observations | 6.085148 | 6.306657 |
| Late ten observations | 7.154780 | 7.707257 |

The total navigation reduction is only 0.11075175960938433%. The candidate's
median is 548.369624 ms versus 553.406335 ms for the baseline. Every navigation
observation in both arms exceeds 100 ms. Preserve the hold and late regressions;
this does not establish a useful general speed improvement. Sensing failure 1173
and the absence of a verified round trip remain. No native controller adopted
this change.

## Independent body-projection comparison

The body-projection controller and harness preparation remains
`78cb96402369a6739b4936364a21de73e72a65ccbc0b0d8e5e415efb9ef6db15`.
Launcher and completion preparation is
`35244eabdb1cd9c5fd5e6ec75956dc4457472436b6a40c4f5c16792f6cdb23c9`.
These record 121 passing geometry, controller, harness, admission and completion
checks. The automatic watch's nine additional control-flow checks passed in
2.12 seconds, session 33597, exit zero. Its preparation is
`7f747e8f7c8c00c683a73d13e40daed2812729bc165beca72b58cd8b071eb7f1`.

The first source preflight, session 10271, correctly stopped at the resource
threshold while the preceding replay was live; it created no runtime output.
After that replay ended, preflight session 47852 passed with 2,373 bound sources,
71,636,930,560 available RAM bytes and 599,197,417,472 free artifact bytes.
The launcher additionally checked that the prior replay and its watcher had
both ended before beginning its original raw/model admission.

Current replay invocation:

- Script: `scripts/replay_go2_body_projected_tiled_late_history_v1.py`.
- Owner: PID 2876376, creation 1789143860.0, tool session 48899.
- Boot: `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`.
- Invocation:
  `docs/go2_body_projected_tiled_replay_invocation_2026-09-11.json`,
  SHA-256 `90e4e89c417df4838e810888886e22172e320bf00079ad73c5a8f20e7512fb2f`.
- Reserved root: `go2_body_projected_tiled_late_history_v1_attempt_001`.
- Baseline: completed `TiledDensityProgressiveFloorController`.
- Candidate: `BodyProjectedTiledController`; no atomic-leaf-freeze adoption.

The owner was confirmed live with 138,953,242,801 logical bytes read. At that
observation the runtime root was still absent: input admission was in progress,
and no paired frame or speed result was established. Do not restart it because
the launch or progress log is quiet.

Automatic completion watcher PID 2876551, creation 1789143904.27, tool session
80027, is active. Its execution document has SHA-256
`941b495b77588555b74a550b1d822ca49ed31bb94a302377d202d424782514dd`
and 2,379 source bindings. It waits for the actual launch, records that launch
identity and original process, waits for the owner to end, and invokes the
checked verifier once with actual result/execution hashes. Do not run a duplicate
checker. Preserve any admission, runtime or verification failure.

The extended-budget native worker 2867880 remained live; latest complete timing
row was tick 2094. No terminal outcome has been established. Preserve the native
queue: extended budget, sustained turn, contact plus flow, then chained anchors.
This is a historical progress observation; recheck authoritative process
identities and completed artifacts before acting. The full unseen-maze
navigation goal remains active with no new navigation or real-time qualification.

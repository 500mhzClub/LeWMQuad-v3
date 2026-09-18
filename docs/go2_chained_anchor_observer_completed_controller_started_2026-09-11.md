# Observer recovery verified; full learned-controller replay started

The chained-anchor observer replay completed 854 observations from frame zero.
Its original observer reproduced all original recorded visual evidence. The
candidate matched the original fields for frames 0–852, then changed at frame
853: it replaced the first measured bridge with a qualified auxiliary-camera
measurement from retained reference 850. It checked the original qualified
increment for conflict and promoted the new view through the original
`accepted_support_margin` rule. The replay stopped there and read no subsequent
observation. This proves an observer intervention in recorded history, not
navigation recovery or a successful physical continuation.

Observer result SHA-256:
`5cbce75a34aa610d7cf227b772a757509fb213cfb1a0980d7f9ffb6cb7ac2c9e`.

Completion verification independently reconstructed all 854 consumed public
packets and original visual rows, recalculated their comparisons, revalidated
the serialized pose witnesses through the runtime contract, and checked the
complete report and artifact/source bindings. The first verification attempt
failed because JSON stores the episode tuple as an array. V1 and its failure
record are preserved. V2 adapts only the validated identity representation in
a private call view; it changes neither stored evidence nor the runtime
validator or observer. All other V1 checks remain in effect. The V2 verification
completed with exit code zero; its combined focused tests passed 18 cases.

- V1 failure record:
  `docs/go2_chained_anchor_observer_completion_v1_failure_2026-09-11.json`, SHA-256
  `37fdef6004d971fdb6e65b7f5ad73bf06c56984c381c0128ccc5de60315e50ba`.
- V2 completed verification:
  `docs/go2_chained_anchor_observer_completion_v2_2026-09-11.json`, SHA-256
  `3ba5a9cdf6bf2a1a4dbd46a95481872fd02aed1acbd7bea0a017612dc515575b`.

Implemented a separate `ChainedAnchorResidualController` and its full-prefix
comparison helper. The mapping, floor registration, mission, residual, action
selector and model remain inherited from the original direct-flow controller.
Twenty focused tests passed. They require complete earlier decisions and
forecasts to match, exact observer evidence, preserved original bridge evidence,
valid forecast-to-command selection, and a terminal zero command on downstream
failure. The comparison explicitly distinguishes anchor admission from recovery
of an original controller failure: the original controller was still live at
this boundary.

The full-controller source/resource preflight passed with 2,267 source bindings,
about 75.2 GiB available RAM, 564 GiB artifact free space and 19.8 GiB workspace
free space. Both preceding full CPU owners had ended. Launched one fresh full
controller replay at:

`go2_chained_anchor_controller_prefix_v1_attempt_001`

under the development artifact volume. It reloads the original assigned no-RGB
JEPA model through the existing snapshot/coefficient checks, requiring state
SHA-256 `fb6f1aba8830a53d67cd6c284fb24199966d5f0c63db3b2a107ab833c81c266f`.
It reuses the completed worker's model admission; it does not retrain or repeat
the training-data study. It will compare the 853 earlier complete decisions and
850 forecasts, evaluate the full controller at frame 853 and stop there. No new
command is executed and no later recorded observation is consumed.

- Launch SHA-256:
  `8eba2f8dfea706109f8cec4fcf55206f36fa3b9f588c0bb2c344b492e95269cd`.
- Preparation SHA-256:
  `ef606453699553f7fb8398aa48d68cbf2ab475c5d3fb8953b56fe644f042438b`.
- Tool session: `17157`.
- Exact PID/create-time/argv/boot identity is recorded in
  `docs/go2_chained_anchor_controller_execution_2026-09-11.json`.

The process reported frame zero, and its live identity and frozen source table
were checked. Its final controller result was not yet available at that check.
The new controller and replay sources/tests/protocol are now bound to the
running attempt and must be preserved.

The separate visibility-batching performance replay also completed. Result
SHA-256 `3aae2867faf3a127b3a42acf13c69514607ce15272d5a98ad4a87da8d32fddb4`.
Its report contains 1,428 observations, 1,425 forecast comparisons and seven
equal retained-state checkpoints. It reports controller totals of 1034.109 s
versus 1011.611 s, with every timed decision still exceeding 100 ms. A dedicated
completion verification remains outstanding; these timings are not real-time
or native navigation qualification, and this optimization has not been adopted
by the new tracking/controller experiment.

# Frozen-footprint replay: independent checker prepared and scheduled

Latest status: the scheduled independent checker completed successfully;
session 58639 exited zero and original waiter PID 2715834 has ended. Completed
verification SHA-256:
`4b574cf6a46943b7135ec11c472f7c1c73a01af48411178a9a4917e0dd5681c8`.
All 2,003 verification source bindings were rechecked. See
`go2_frozen_footprint_anchored_prefix_result_2026-09-10.md` for the result and
measurement scope. The scheduling details below are historical.

The independent recorded-evidence checker passed 38 focused tests in 2.26
seconds (session 46400, exit zero). Its recursive source preflight passed with
2,003 bindings (session 97912, exit zero), including all 2,000 prepared original
replay sources and the new checker, test and protocol. These sources are now
bound by the verification schedule and must remain unchanged while it runs.

An observation-only waiter is live under PID 2715834, creation epoch
1789056706.75, tool session 58639. It follows the exact original replay PID
2715071, creation epoch 1789056341.94, and its exact command from the original
execution record. The scheduling record is
`go2_frozen_footprint_anchored_verification_scheduling_2026-09-10.json`,
SHA-256 `51fd844d17d947803c263232e01608a8e99f86e67cc9be11ef9c7f409696355f`.

The waiter will bind the replay launch while its original owner is live, after
checking that the complete source map matches the pre-execution record. It
records that observation in
`go2_frozen_footprint_anchored_verification_launch_identity_2026-09-10.json`.
After the original process ends successfully, it runs the checker with explicit
expected launch and result hashes. It then rechecks the prepared verification
sources and the completed checker's binding. A missing launch, terminal replay
failure, changed source, reused PID, expired two-hour wait, or checker failure
stops this verification; no original process is restarted.

The checker reconstructs all saved decision hashes and original command
endpoints and recomputes the complete timing summaries. It checks the four
reported state hashes against the completed reference, without claiming an
independent neural rerun or hidden-state reconstruction. Only the original
paired replay owns that execution evidence. The protocol is
`go2_frozen_footprint_anchored_verification_v1_2026-09-10.md`.

At scheduling, the original replay was still verifying its inputs and its
output root had not yet appeared. Original native worker 2709978 and batch
parent 2659758 remained live. Available artifact storage was about 597 GiB.
There is no completed new replay, verified controller speedup, additional
navigation episode or hardware result at this stage.

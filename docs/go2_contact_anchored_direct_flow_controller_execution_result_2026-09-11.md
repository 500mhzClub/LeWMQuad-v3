# Full contact controller tracking-recovery replay is running

The completed observer replay recovered a current pose at frame 561. The next
experiment now runs the original contact-scoring controller and a separate
controller with only the existing direct-flow observer substituted. Each has
a fresh copy of the original full-supervised model. The full replay checks
floor registration, mapping, model predictions and selected commands through
the known boundary. It stops there even if the recovered request remains hold.

Fifteen focused tests passed in 2.17 seconds, session 6849, exit 0. They cover
complete preboundary decision comparisons, preserved negative floor outcomes,
actual controller-class integration, and a bounded synthetic execution of the
two-model replay loop. The first expanded-loop test run failed because its
fixture added selected_action only to the candidate. That fixture was corrected
without changing runtime code; the failure is retained in the preparation
record. Source preflight session 74927 exited 0 with 2049 bound source paths,
79,103,586,304 bytes available RAM and 608,526,266,368 free artifact bytes.

Preparation:
`docs/go2_contact_anchored_direct_flow_controller_preparation_2026-09-11.json`,
SHA-256 `82b55927ac7569eba8612771922da05c1b547d67df35724c07ca4654abc91b63`.
All prepared/launched source paths are now frozen.

The replay is PID 2822758, creation time 1789119437.63, command
`.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/replay_go2_contact_anchored_direct_flow_controller_prefix_v1.py`.
Boot ID `1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Tool session 35432.
Output `go2_contact_anchored_direct_flow_controller_prefix_v1_attempt_001`.
Launch SHA-256 `6c577445f7bff2e58aad960c6d26908c683a20b0d7a4fbb2344967b5072da9ad`.
Assigned model SHA-256
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.

Execution record:
`docs/go2_contact_anchored_direct_flow_controller_execution_2026-09-11.json`.
Session 62351 verified the exact live process, original boot, launch and source
bindings, fixed observer inputs, absence of terminal output, and the confirmed
first decision. The first original decision reconstructed, the candidate's
normalized first decision matched it, and public inputs remained unchanged.
Only the confirmed first row of the active gzip stream was inspected.

Next: poll session 35432 and the exact owner. Completion requires 562 original
decisions and 558 preboundary forecasts, unchanged states/gradients for both
models, the fixed observer evidence, and final worker/input artifact checks.
Preserve any full-controller floor or planner failure. No postboundary old
observation may serve as new candidate motion. A nonzero exit must be diagnosed;
do not restart on quiet output or an observation timeout.

The original native contact parent and contact/tracking/budget waiters were
still active at the prior check. The sustained-turn native waiter remains
behind that queue, with its raw prerequisite now completed and separately
verified. This new replay occupies the one full CPU replay slot and starts no
native scene. No new round trip or independent-policy selection is claimed.

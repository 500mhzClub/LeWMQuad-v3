# Queue the matched nominal episode after the exact learned native owner

`scripts/await_go2_nominal_measured_plane_native_v1.py` owns one exclusive
waiter root, `go2_nominal_measured_plane_native_wait_v1_attempt_001`, and may
dispatch the uncreated `go2_nominal_measured_plane_maze02_v1_attempt_001` once.

Wait on the exact learned parent PID, creation time, argv and boot recorded
in `scripts/nominal_measured_plane_native_inputs_development.py`. While live,
check only that original launch identity and sleep 30 seconds between polls.
Do not read a future result, infer completion from elapsed time or restart a
process after an observation error. No native scene is launched while waiting.

After the original owner ends, require its actual complete result and full
input admission. A fully audited failure to navigate is admissible; a failed
execution/audit is preserved without automatic bypass. Wait for a free native
slot before dispatching one child, and retain any new failure without retry.

The child's fixed science and complete audit are defined in
`docs/go2_nominal_measured_plane_maze02_v1_2026-09-11.md`. The waiter records
the actual child command/PID and waits for its terminal process status. After
successful process completion, authenticate every child artifact and worker
receipt, recheck nominal controller/audit attribution, reconstruct physical
prefix and contact/progress/timing readout, and preserve either scientific
outcome. Do not rerun the full model/controller audit or unrelated training.

Source preflight requires the prepared source/recorded-prefix bindings, current
resources and original owner observation. It creates no waiter/native output
and runs no scene or model. All source bindings are frozen at waiter launch.
This queue grants no navigation, independent-layout, real-time or hardware
qualification and never relabels earlier failed waiters as complete.

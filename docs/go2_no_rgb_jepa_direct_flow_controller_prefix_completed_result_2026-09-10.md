# Full-controller tracking recovery replay completed

The completed replay reconstructs all 860 observations through frame 859.
All 859 earlier complete decisions and 856 original forecasts match. At the
original tracking failure, the candidate retains a valid visual pose and
selects `right_turn`, requesting `[0, 0, -0.45]`. The assigned model state is
unchanged. No command from that changed decision was executed, and no following
observation was consumed. The original failed episode remains a failure.

The runner completed its original input admission before and after replay and
wrote terminal result SHA-256
`152da30ba5c142d8150dacd5865278f81d5db46c7603bd637eb8ec2eee292fdf`.
Its original process, PID 2749113, has ended. The completion check authenticated
all 1,947 source bindings, all three output artifacts, the result/report
agreement and the unchanged artifacts from the earlier independent boundary
receipt reconstruction. That completion check did not independently repeat
neural inference or the full training-input admission.

[Completion verification](go2_no_rgb_jepa_direct_flow_controller_prefix_completion_verification_2026-09-10.json)
has SHA-256
`5ff771915eb24483dd855af4a54176f3f1beb1189cda81426fe8b92f92ebd56f`.

The existing tracking-recovery waiter has accepted this exact result. It still
waits for the six-case batch, frontier pilot, hold-reorientation pilot and
contact-score pilot to finish and authenticate before starting its one fresh
simulator episode. Do not launch another copy of that episode. The replay
establishes recovery at the observed failure boundary; continued physical
navigation and a verified round trip remain unproven.

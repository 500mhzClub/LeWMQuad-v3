# Fixed observed-prefix diagnosis during the adapter batch

Snapshot exactly the first 1,000 complete decision rows (0–999) of the original
first adapter case, `all_phase_full_jepa_residual_maze_02`, launch
`97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a`.
The original case/batch may still be running; do not alter, stop or restart it.
Write a separate immutable compressed copy and compact summary. Re-read that
same original prefix and require its canonical row hash to match. Do not consume
additional rows or claim to bind the original still-growing whole stream.

Count requested commands, discretionary holds outside warmup/mission settling,
contiguous hold observation intervals and the corresponding observed pose/goal
estimates and waypoint targets. Summarize the saved phase, articulated-surface,
raw eight-step nominal-path gates and utilities. Distinguish feasible lower-scored
movement from higher-scored movement vetoed by its raw predicted path. Mark
recovery overrides separately; raw-gate eligibility is not a reconstruction of
those overrides or a clearance certificate. No model inference, policy alteration,
alternate trajectory or hypothetical success is produced.

Use observed-controller receipts only; do not read native pose/contact traces or
claim completed-command, physical progress, final navigation or full episode
audit. Those remain the original batch's responsibility. The fixed-prefix
diagnosis is evidence for subsequent development work, not an independent maze
trial or permission to mutate the existing queued frontier intervention.

Bind original and queued launch identities and unchanged inherited/new sources.
Use eight GiB available RAM and 40 GiB artifact reserve plus one GiB allowance.
Exclusive output `go2_adapter_hold_prefix_v1_attempt_001`; preserve failures and
do not silently replace a truncated or changed prefix.

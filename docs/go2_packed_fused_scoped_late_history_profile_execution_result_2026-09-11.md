# Remaining controller-cost profile is running

The completed packed/fused optimization preserved decisions while reducing
total controller time by 9.38%, but its median remains 665 ms and every planning
call exceeds 100 ms. The new profile rebuilds that same 1,428-observation
development history with the improved controller and profiles the original
three fixed ten-frame windows to identify remaining costs.

The original profile loop is reused with exactly four substituted globals:
controller constructor, complete-decision normalizer, output root and progress
label. All profiled inputs and decisions must match the completed packed replay.
No new native command or sensor episode is generated, and the existing failed
visibility and round-trip evidence remains part of the input scope.

All 35 focused profile/completion-checker tests passed in 2.22 seconds (session
66732, exit 0). Source preflight session 68447 exited 0 with 2,189 source paths,
72,218,243,072 available RAM bytes and 617,883,734,016 free artifact bytes. The
original one-model profile thresholds remain 48 GiB RAM, 41 GiB artifact space
and four physical CPUs. Preparation session 71645 exited 0.

The actual launch then authenticated the completed packed/fused reference and
rehashed the original raw worker artifacts and bound model inputs. It reused
the earlier full training admission without rerunning training ancestry.

Owner: PID 2792594, creation time 1789096258.61, tool session 67783.
Command: `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/profile_go2_packed_fused_scoped_late_history_v1.py`.
Root: `go2_packed_fused_scoped_late_history_profile_v1_attempt_001`.
Launch SHA-256: `b0033cf1aae03125f82fb04e69a9f7b528569b61c79c94734112aaaa8a87a377`.

Execution verification session 44076 exited 0 after checking the exact live
owner, source/launch/preparation bindings and a complete matching row at frame
101. No result or failure record existed at that check. This process owns the
single full CPU replay slot alongside the original sixth-case native audit;
do not restart it because an observation call times out.

Preparation: `docs/go2_packed_fused_scoped_late_history_profile_preparation_2026-09-11.json`,
SHA-256 `6b65a1e3c03b2da98e26c2667b004abf155c36affe9d19bf1a7c84575af78a2b`.
Execution: `docs/go2_packed_fused_scoped_late_history_profile_execution_2026-09-11.json`,
SHA-256 `37548b4f9792f08fb9d2d708a3af4caae4dcf554ff8739d6ff008ffdebc7de4c`.

Completion will require every original decision/input identity, fixed window,
model-state check and output binding to pass. Cumulative profile costs overlap,
profiler overhead is retained, and this will not establish sensor latency,
real-time execution or new physical navigation success.

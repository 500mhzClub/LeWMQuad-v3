# Chained-anchor prospective physical-prefix comparison

The completed controller replay is authenticated by
`go2_chained_anchor_controller_completion_verification_2026-09-11.json`,
SHA-256 `6b0c3298bb4df47aa59affc80d5dcb9236c1e4d9ae5cabd07550800c94d26425`.
It consumed 854 observations, preserved 853 earlier complete decisions and
850 earlier forecasts, and admitted a reacquired anchor at frame 853. Both
controllers requested the same right turn at that boundary. The original
controller was still live there. This is not evidence of recovered navigation.

`scripts/chained_anchor_native_prefix_development.py` evaluates a future fresh
simulation against that complete replay and the original failed tracking run.
It requires identical physical trajectory samples 0 through 43399, identical
854 public sensor packets, identical complete candidate decisions through frame
853, and completed actual command tapes through that frame. Both traces must
also contain the 50 physics samples for the boundary command. Those subsequent
samples are not required or claimed to match. No later outcome is inferred.

The fixed intervention is in anchor evidence at frame 853, not in the command
at that frame. All original requested commands through the boundary must match.
The helper reconstructs all saved replay comparisons and the entire report.
It uses native geometry only for evaluation after collection, never as a
controller input. It launches no simulation.

Synthetic tests cover all 854 rows, insufficient and excess populations,
changed physical samples, missing command samples, changed sensor identity,
changed full decisions and forecasts, incomplete commands, changed endpoints,
and false claims about navigation or command changes. The preparation checker
also reconstructs the actual completed prefix with this new helper and checks
the source and artifact bindings before and after that read.

The existing collector/auditor preparation remains unchanged. A source-bound
native launcher and ordered waiter remain to be prepared. The current queue
must complete through the extended-budget, sustained-turn, and contact-plus-flow
diagnostics before a chained-anchor simulation. Their actual outcomes must be
retained and reviewed; none of this preparation selects an independent-layout
study policy or grants hardware or sealed-evaluation authority.

# Partial-height native experiment preparation

Prepared a fresh maze 1 collection, independent complete raw audit, prospective
replay admission and physical-prefix comparison. Native execution has not been
submitted. It requires a completed positive first-change replay; the current
replay is still running under session 20949 / PID 2513816. No outcome has been
assumed and no command is selected in this preparation document.

30 tests passed in 3.85 s, session 28970 exit 0. The collector and evaluator
ASTs retain every original calculation apart from controller identifier and
collection status labels. Worker tests verify separate fresh collection/audit
models and retained evidence after audit, prefix or verification failure.
Admission tests reconstruct all saved decisions against the actual predecessor
stream and reject short/extra, altered-model/input/state/tracker/receipt,
unchanged-hold and negative replays. Physical comparison tests check 25,950
physics samples, 505 paired observations, 504 preceding commands and 501 model
banks; changed future physics is intentionally excluded, while a partially
executed changed command is retained as a negative physical outcome.

Source-only closure check 54421 exited 0: all 1,700 discovered/inherited source
paths match their current SHA-256 bindings. This did not admit an unfinished
prefix, verify all runtime ancestry, instantiate a model, or execute a scene.
Full --preflight-only remains pending until the replay result exists. No frozen
running source was modified.

Fresh hardware from that check: 16 physical/32 logical/all 32 affinity, CPU busy
9.6%; available RAM 73,263,136,768 bytes; artifact free 75,768,889,344 bytes;
workspace free 21,358,682,112 bytes. Both GPUs idle, card 1 VRAM total
34,208,743,424/used 1,398,722,560 bytes. Competing hold, support and height replay
RSS were 4,801,687,552 / 4,326,961,152 / 1,433,812,992 bytes. No native scene.
Resources must be refreshed after the positive replay before preflight/launch.
The pilot uses the original 32 GiB memory and 40+10+1 GiB storage admission.
It does not lower the unmet fixed supervised cohort's storage gate.

Frozen prepared files:

| Path | SHA-256 |
| --- | --- |
| scripts/partial_floor_height_maze01_episode_development.py | cae493e28177ab627dc1cbed767056e4768fae13677514ac413ba4dd6fe5c845 |
| scripts/partial_floor_height_maze01_audit_development.py | d2eb1e10df71530952b6c28324d02bbd2d08f3451a4afef600ae38dd132cdfae |
| scripts/partial_floor_height_maze01_native_prefix_development.py | 55731a21495108de58ec66f45c892832ab5c21cebab26b3c72c8dd86cc09aff1 |
| scripts/run_go2_partial_floor_height_maze01_pilot_v1.py | ade43cac807faa5e8faebbb3287fd75c0678b26b1829e50059ba0818d34c439c |
| lewm/tests/test_partial_floor_height_maze01_native_development.py | 4625142ed038b4e9024355564b212ddf49ad5b1663f16cab9bf5c0faf3cbf3dd |
| lewm/tests/test_partial_floor_height_maze01_native_prefix_development.py | bfc8066a851074e24775daf9846e5e4aa20f4b0ccc4a4069d4159cb57e8f87f2 |
| docs/go2_partial_floor_height_maze01_pilot_v1_2026-09-09.md | 09ee0a2dba21646756f2b96963ad621ca819766092cb08dbbe6fadf5285481c3 |

Next: admit the completed exact height prefix only if it recovered and selected
a changed command; run the new launcher with --prefix-result-sha256 and
--preflight-only, then the same command without --preflight-only after fresh
resource and single-native-scene checks. The protocol allows this new corrective
pilot in an idle native slot without changing either the fixed supervised
cohort or the prepared tracking maze 3 dependency. If the replay is negative,
preserve it and diagnose it instead. No cleanup, retry or outcome is authorized
by merely writing this preparation.

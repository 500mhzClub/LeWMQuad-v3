# V4 C0 oracle gate result: passed, 28 September 2026

**C0 completed 20/20 development round trips on the frozen harness `v4_completed_support` (`82b7b604…`), with zero disallowed contacts, zero hard-clearance violations and no C0 validity problem. The gate (at least 19/20, zero contacts, zero hard violations) passes.**

The harness is validated: with true motion in the prediction slot, the shared stack completes beacon-and-return missions in these mazes. Capability qualification started at 11:50 BST on this harness.

## How it ran

- **Episodes.** The first four gate episodes (00/0 to 01/1) were not rerun. The fifth, 02/0, reached a terminal round trip before the old checker's closeout error. It was re-evaluated from its preserved records under [the prefix erratum](go2_navigation_capability_oracle_prefix_erratum_2026-09-28.md) and was not rerun.
- **Frozen components.** The remaining 15 used the unchanged run owner, readers and assignments, run by `scripts/run_go2_capability_completed_support_v4_gate_erratum_continuation_development.py`. Four ran concurrently under handoff §5.7. Every C0 episode is prefix-qualified on its own.
- **Retention.** 00/0 kept full frames and passed a bitwise replay: 2,256 RGB and 2,256 depth packets matched and the native trace was exact. All later episodes kept hashes only.
- **Cohort record.** `cohorts/v4_completed_support_C0_gate/result.json`, SHA-256 `bd2da649…`. Per-episode data is in [the JSON summary](go2_navigation_capability_completed_support_v4_gate_result_2026-09-28.json).

## Results

| Measure | Value |
|---|---|
| Round trips | **20/20** |
| Disallowed contacts / hard violations / operating-margin violations | 0 / 0 / 0 |
| Unresolved native samples; FK interval robustness failures | 0; 0 |
| Minimum articulated separation (lower bound) | 56.2 mm (08/0). All other episodes were at least 101 mm. |
| SPL, mean outbound / return | 0.777 / 0.929 |
| Simulated mission time, median (range) | 154.4 s (102.9–234.7 s); no episode came near 480 s |
| Owner wall time per mission, median (total) | 1,004 s (5.64 h) |
| Planning decisions; comparable branch rows | 7,500; 44,994 |
| Comparable-prefix error, maximum | 0.0 mm, 0.0° |
| Decisions with no matching branch | 1 (02/0 at 119.9 s; missing current observation, then vetoes) |
| Vetoed selections | 35, in seven episodes (maximum 11, in 03/1) |
| Frozen checker passed unchanged | 19/20. 02/0 was requalified under the erratum. |

## Veto watch

Dispatch-layer substitutions occurred in seven episodes:

| Episode | Vetoed selections | Vetoed movement selections | Veto ticks | Missing-observation ticks |
|---|---:|---:|---:|---:|
| 00/1 | 7 | 3 | 64 | 1 |
| 02/0 | 3 | 3 | 34 | 1 |
| 02/1 | 2 | 2 | 19 | 1 |
| 03/1 | 11 | 0 | 88 | 2 |
| 04/1 | 5 | 0 | 45 | 0 |
| 05/0 | 1 | 0 | 5 | 0 |
| 09/0 | 6 | 4 | 40 | 0 |

All seven succeeded. No failure involved repeated vetoed selections, so no veto-loop mechanism arose. Override ticks (no on-time plan; settling or terminal holds) occurred in every episode, as they do for C1.

## Status and budget

- **Harness.** V4 is frozen and passed its gate, so the six-version cap is no longer binding.
- **Budget.** 71.59 of 160 hours used (the 70.19-h baseline plus 1.40 h of active job time). RecoveryStorage has 101.8 GiB free and the workspace 5.47 GiB, both above their reserves.
- **Next.** Capability qualification on validation episodes 10/0–29/0 under the [committed plan](go2_navigation_capability_qualification_plan_2026-09-28.md).

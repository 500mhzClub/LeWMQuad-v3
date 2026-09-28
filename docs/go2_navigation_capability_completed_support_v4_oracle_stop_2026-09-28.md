# V4 oracle gate stop — 28 September 2026

**C1 passed both screens, 20/20 safely. C0 passed its first four missions safely, then stopped on the fifth assignment, 02/0, at closeout because oracle prefix qualification failed. The gate is incomplete. No validation runs or official videos have started.**

| Stage | Result | Status |
|---|---|---|
| C1 first development episodes | 10/10 round trips | Passed |
| C1 second development episodes | 10/10 round trips | Passed |
| C0 00/0, 00/1, 01/0, 01/1 | 4/4 round trips; no contacts or hard violations | Individually qualified |
| C0 02/0 | Oracle prefix check failed | Preserved, unqualified |
| Remaining C0 gate assignments | 15 unstarted | Stopped |
| Capability qualification and official videos | Unstarted | Await passing gate |

The first corrected C0 recording passed command replay: 1,128 frame pairs, 2,256 RGB matches, 2,256 depth packet matches and all native trace values exact. Later oracle runs therefore legitimately used hash-only sensor retention.

## What failed

Of 2,328 branch-prefix checks in 02/0, 2,322 had a matching executed command prefix and all matched with **zero position and yaw error**. Six rows at one decision were rejected as `NO_MATCHING_EXECUTED_PREFIX`, not for a measured pose discrepancy. The frozen requirements remain 1 mm and 0.1 degrees.

At simulator time119.9s (118.4 mission seconds), all six branch tapes began with forward command `[0.2, 0, 0]`. The actual dispatch immediately requested and applied `[0, 0, 0]`, reason `CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE`; the following commands carried `COMMAND_WINDOW_VETO_LATCHED`. Twenty later request records were labelled with this prediction origin. Thus it is not sufficient to discard these rows as irrelevant: that prediction has no continuously matching executed prefix from its source boundary.

This narrows the issue to the relationship between the assumed committed prefix, live dispatch and qualification. It does not demonstrate a restoration pose error. The cause of the observation-unavailable veto is not yet established. No thresholds, checker, oracle or controller have been changed.

The owner completed158.52 simulated seconds and recorded an observed round-trip candidate before its closeout check failed. The physical arrival/clearance reader was not run for02/0 because the cohort stopped first; it is **not counted as a verified fifth success**, and no full-trajectory safety claim is made for it. All logs, native trajectories, hashes and failure evidence are preserved.

## Disposition required

The explicit instruction to stop at a fidelity/approval boundary is in force. No retries, remaining gate assignments, validation episodes or new physics were launched after this failure. Five of six outcome versions have been consumed; the latest completed-cohort projection was102.98/160h, with storage fitting. This is a fidelity stop, not a version, storage or wall-budget stop.

Proposed next step for approval: a bounded investigation of the recorded prefix/dispatch discrepancy, then a concrete correction proposal if required. Preserve this attempt and distinguish a checker accounting defect from a genuinely invalid prediction prefix; do not bypass qualification by dropping the six unmatched rows. A new run or resumption plan must explicitly address this failed assignment.

Frozen harness source:`7da82b23`; C1 second-screen report commit:`6daedc6e`. Numerical evidence and file hashes are in [the stop record](go2_navigation_capability_completed_support_v4_oracle_stop_2026-09-28.json).

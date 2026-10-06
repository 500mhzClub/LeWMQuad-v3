# V4 completed-support first-episode screen

> **Version-ledger correction (28 September 2026):** the six-version cap counts V0 (the pre-registration governs). Read this record's "N of six" as N+1 of six, including V0. V4 is the sixth and last version. See [the version ledger](go2_navigation_capability_version_ledger_2026-09-28.md).

**10/10 beacon retrievals and returns. Zero disallowed contacts, hard-clearance violations, operating-margin violations or unresolved native clearance samples.** The first C1 screen passes; the same frozen harness is now running the ten second-episode checks. Five of six outcome versions have been used.

The single change makes visual recovery use the complete selected feature counts already used by the unchanged tracker, retaining its 48/96 thresholds. Tracking estimation, sensors, models, candidates and safety rules are unchanged. Exact pre-change command replays motivated the change; this is a development result, not a held-out capability estimate.

| Episode | Beacon time (s) | Return duration (s) | Total (s) | Round trip |
|---|---:|---:|---:|---|
| 00/0 | 87.20 | 43.20 | 130.40 | Pass |
| 01/0 | 126.50 | 84.70 | 211.20 | Pass |
| 02/0 | 146.30 | 63.00 | 209.30 | Pass |
| 03/0 | 87.20 | 63.30 | 150.50 | Pass |
| 04/0 | 98.10 | 40.40 | 138.50 | Pass |
| 05/0 | 103.60 | 50.10 | 153.70 | Pass |
| 06/0 | 108.80 | 42.10 | 150.90 | Pass |
| 07/0 | 92.50 | 90.30 | 182.80 | Pass |
| 08/0 | 124.80 | 72.00 | 196.80 | Pass |
| 09/0 | 146.50 | 62.40 | 208.90 | Pass |

Both remaining V3 failures, 01 and 09, now complete. All eight previous successes are retained. No pose loss or mission timeout occurred. Hold-rule counts and native clearance records remain in the accompanying JSON and original episode evaluations; planned arrival holds are distinguished from exclusions.

The cohort took 1.427 wall-hours including native evaluation. The measured projection is 103.61/160 hours with 15% contingency, including the second C1 screen, C0 gate, reduced capability set and videos. Projected additional retained storage is 46.88 GiB against 92.66 GiB usable after reserve. No concurrency gain is assumed.

Frozen source: `7da82b23`. Harness SHA-256: `82b7b6049a93c34cc00599db93a82fee0e7a882683af45ccaae47b5443c84871`. Full results: [machine-readable report](go2_navigation_capability_completed_support_v4_screen_result_2026-09-28.json).

Continue without tuning: second C1 screen must reach 9/10 safely; then C0 must reach 19/20. The first corrected C0 episode retains full RGB-D and must pass bitwise command replay before later C0 runs use hashes. Capability qualification and official videos remain pending.

# Exhausted-view V2 development screen result

> **Version-ledger correction (28 September 2026):** the six-version cap counts V0 (the pre-registration governs). Read this record's "N of six" as N+1 of six, including V0. V4 is the sixth and last version. See [the version ledger](go2_navigation_capability_version_ledger_2026-09-28.md).

**8/10 round trips and 10/10 beacon retrievals**, compared with V1’s 5/10 and 9/10. No pose losses, disallowed contacts, hard-clearance violations or operating-margin violations. The 9/10 first-episode threshold is not met.

| Episode | Beacon | Round trip | Simulated seconds | References retired |
|---|---|---|---:|---:|
| 00/0 | True | True | 130.42 | 0 |
| 01/0 | True | False | 480.00 | 0 |
| 02/0 | True | True | 159.62 | 0 |
| 03/0 | True | True | 463.12 | 0 |
| 04/0 | True | True | 132.72 | 0 |
| 05/0 | True | True | 165.72 | 1 |
| 06/0 | True | True | 160.92 | 0 |
| 07/0 | True | True | 187.72 | 1 |
| 08/0 | True | True | 301.62 | 2 |
| 09/0 | True | False | 480.00 | 0 |

The single shared recovery change turns 05 (return timeout), 07 (return pose loss) and 08 (outbound timeout) into successful round trips. Their unsuccessful attained-view references are retired once, once and twice respectively. This is development-screen evidence, not controller capability qualification or a JEPA comparison.

01 and 09 remain return timeouts at 480 s. Neither triggers the new reference-retirement rule. Their previously diagnosed turn-memory deadlock and prolonged turn/recovery oscillation remain the next diagnostic targets; no further controller change is selected or implemented in this result record.

The complete serial cohort took 2.15 wall-hours, including native evaluation. Three of six outcome-driven versions are consumed. No simulation is currently running.

The frozen owner’s post-cohort projection predates the second-episode amendment and assumes no additional tuning screen. It must be supplemented before another cohort; it is not admission for the next version. All programme caps and the 480-s mission budget are unchanged.

Next: diagnose the remaining failures and select at most one shared change. Once the first-episode screen reaches 9/10, run the same-harness second-episode C1 check and require 9/10 before C0. If that check fails, subsequent versions screen all 20 with an aggregate 18/20 threshold. Neither the second-episode check nor the C0 gate has started.

Evidence: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/cohorts/v2_exhausted_view_C1_screen/result.json`.

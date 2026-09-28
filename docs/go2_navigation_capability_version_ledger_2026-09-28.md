# Navigation capability harness: version ledger, 28 September 2026

This is the canonical count of harness versions against the six-version cap. Andrew Knowles ruled on 28 September 2026 that **the pre-registration governs**: "at most six versions including V0" (version IDs 0–5). V0 plus five charged changes makes six versions. **`v4_completed_support` is the last version.** If its C0 gate fails, work stops and is reported; Andrew then decides whether to allow another change.

| # | Version | Frozen | Harness SHA-256 | Change | Charge | C1 screen |
|---|---|---|---|---|---|---|
| 1 | V0 (`v0`, adapters r1–r4) | `523005c1` | — | Stack as deployed for the V4.2 source runs | Base version | Pilot only |
| — | `v0_task_c1` | `13ebc8e2` | `e4480ffa…` | Settled-start task transform; corrected arrival reader | Not charged: correctness | 3/10 |
| 2 | `v0_startup_c2` | `9ee22d80` | `b5807450…` | Map bound from generator envelope; startup floor fallback; bounded startup rotation | **Charged**: containment failed on stale route-grid offsets | 3/10, diagnostic only |
| — | `v0_grid_c3` | `f904a9b5` | `8ec038a8…` | Grid-index correction at all call sites | Not charged: correctness, containment exact | 3/10 |
| 3 | `v1_paired_floor` | `2469bb79` | `e069986c…` | Initial map floor from the qualified paired depth plane | **Charged**: normal-start initialisation changed | 5/10 |
| 4 | `v2_exhausted_view` | `bad3883a` | `83382ac9…` | Retire an attained view-recovery reference after 1 s of aligned accepted poses | **Charged** | 8/10 |
| 5 | `v3_live_turn` | `8c825807` | `7c458c5a…` | Release the turn-memory latch when its direction becomes ineligible | **Charged** | See `v3c1` |
| — | `v3c1_live_turn` | `4f503906` | `0c87318d…` | Same change, re-bound to the deployed memory class after a zero-decision startup failure | Not charged: implementation erratum | 8/10 |
| 6 | `v4_completed_support` | `7da82b23` | `82b7b604…` | Recovery uses the tracker's actual selected-feature count, including sparse corner completion | **Charged** | 10/10 first, 10/10 second |

That is five charged changes, or six versions counting V0. The cap is reached.

## Records corrected by this ledger

Earlier records counted charged changes against six without counting V0. Read each "N of six" as **N+1 of six, including V0**.

- **Given a correction banner:**
  - `…_paired_floor_v1_result_2026-09-27.md` ("two of six");
  - `…_exhausted_view_v2_result_2026-09-27.md` ("three of six");
  - `…_completed_support_v4_screen_result_2026-09-28.md` and `…_completed_support_v4_second_screen_result_2026-09-28.md` ("five of six");
  - `…_completed_support_v4_oracle_stop_2026-09-28.md` ("five of six").
- **Left unedited, corrected here only,** because the frozen V4 harness binds them by hash:
  - `…_grid_c3_failure_diagnosis_2026-09-26.md` ("two of six");
  - `…_paired_floor_leg_diagnosis_2026-09-27.md` ("third of six");
  - `…_v2_turn_diagnosis_2026-09-27.md` ("fourth of six");
  - `…_v3_support_diagnosis_2026-09-27.md` ("fifth of six").
- **JSON field.** The frozen JSON records' `outcome_iteration_versions_consumed` counts charged changes and excludes V0. Its value of 5 for V4 means six versions under the cap.

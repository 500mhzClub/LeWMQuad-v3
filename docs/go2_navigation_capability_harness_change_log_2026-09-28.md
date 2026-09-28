# Navigation capability harness: change log, 28 September 2026

Every harness version from V0 to the frozen `v4_completed_support`, with its diff, charge status and screen results. Charge status follows [the version ledger](go2_navigation_capability_version_ledger_2026-09-28.md): six versions including V0, so V4 is the last. Diffs are taken from each freeze record's hash bindings; the full source diff is `git diff <predecessor freeze> <freeze>` on the files named. All changes are shared identically by C0–C4. None of them touches the visual tracker's estimation, the learned models, the sensors, the candidate bank, the 480-s budget or the arrival rules.

| Version | Freeze commit | Harness SHA-256 | Behavioural diff: new or changed controller modules | Charge | C1 screen (safety) |
|---|---|---|---|---|---|
| V0 (adapters r1–r4) | `523005c1` | — | Deployed V4.2 stack. Adapters for episode binding, prediction-slot models, accounting and replay only. | Base (version 1 of 6) | Pilots: C0–C3 round trips; C4 home only |
| `v0_task_c1` | `13ebc8e2` | `e4480ffa…` | New `navigation_capability_target_reference` (one-time settled-start cue transform), sensor-hash retention and environment pin; evaluator reads fixed-world arrivals | Correctness | 3/10 (0 contacts) |
| `v0_startup_c2` | `9ee22d80` | `b5807450…` | New `navigation_capability_map_domain` (±8-m storage, ±7.9-m points) and `navigation_capability_startup_recovery` (downward-camera floor fallback, bounded rotation). Changed `axis_aligned_fine_connectivity`, `coverage_translation_view`, `clearance_preferred_route`, `fine_stored_obstacle_routing`. | **Charged** (2): containment failed | 3/10, diagnostic only (0 contacts) |
| `v0_grid_c3` | `f904a9b5` | `8ec038a8…` | Grid-index offsets corrected at every call site, in 11 routing and mapping modules including `fine_goal_route`, `cached_fine_connectivity`, `multirate_routing_map` and `exact_mission_target`; call-site audit and refined containment | Correctness (containment exact) | 3/10 (0 contacts) |
| `v1_paired_floor` | `2469bb79` | `e069986c…` | New `navigation_capability_paired_floor_start` (initial floor from the qualified paired depth plane) | **Charged** (3) | 5/10; 9/10 beacons (0 contacts) |
| `v2_exhausted_view` | `bad3883a` | `83382ac9…` | New `navigation_capability_exhausted_view` (retire an attained view-recovery reference after 1 s of aligned accepted poses) | **Charged** (4) | 8/10; 10/10 beacons (0 contacts) |
| `v3_live_turn` | `8c825807` | `7c458c5a…` | New `navigation_capability_live_turn_memory` (release the turn-memory latch when its direction becomes ineligible) | **Charged** (5) | Zero-decision startup failure on the wrong memory class |
| `v3c1_live_turn` | `4f503906` | `0c87318d…` | New `navigation_capability_live_turn_binding_c1` (the same change, bound to the deployed `InterruptedRouteTurnMemory`) | Implementation erratum | 8/10; 10/10 beacons (0 contacts) |
| `v4_completed_support` | `7da82b23` | `82b7b604…` | New `navigation_capability_completed_support` (recovery uses the tracker's actual selected-feature count, including sparse corner completion) | **Charged** (6, the last) | 10/10 first, 10/10 second (0 contacts) |

The C0 gate on V4 and its outcome are recorded in the gate result document.

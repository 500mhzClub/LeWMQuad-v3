# Current research brief

The authoritative reference is [LeWMQuad-v3 agent brief: navigation testbed and controller capability, 25 September 2026](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md).

It supersedes the decision-headroom programme. V4.2 is closed; its recommended study-design revision is not pursued. Follow the new brief's authorised scope, gates, budgets and stop conditions. Earlier protocols and results remain historical records.

## Active goal — explicitly started 25 September 2026

**Programme goal:** determine when JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory. Start with a validated testbed, then the E1 benchmark, then moving obstacles. E1 execution and moving obstacles require separate approval.

**Current goal:** establish a validated Go2 simulation navigation testbed. For each controller type (C0 oracle, C1 command history, C2 reactive, C3 JEPA, C4 supervised predictor), determine whether it completes beacon retrieval and return in unseen mazes, report against the pre-registered capability criterion, and produce one example video per controller (C0 optional under the brief).

**Status:** the unchanged corrected C1 screen completed: 3/10 round trips, zero disallowed contacts and hard violations. Five startup failures (four initial-floor failures and one map-bound rejection) are covered by the user's combined correctness amendment. All-180 structural checks and a bitwise first-frame regeneration of 01/0 are complete. Shared correction `v0_startup_c2` expands the generator-sized map domain and adds measured startup-floor recovery. Its full ten-episode screen is running, with exact containment required on all five original episodes that reached planning; divergence consumes a harness version. Projection: 73.5/160 hours (119.7-hour conservative grid-cost sensitivity), 46.9 GiB additional against 100.6 GiB usable. The oracle gate, capability and official videos remain pending. See [the startup erratum](go2_navigation_capability_startup_erratum_2026-09-26.md). The goal remains incomplete.

This replaces the 7 September navigation goal and the closed decision-headroom programme; neither is to be resumed. Machine-readable active-goal state is in [go2_navigation_capability_active_goal_2026-09-25.json](go2_navigation_capability_active_goal_2026-09-25.json). The thread goal tool refuses to replace its unfinished legacy goal; it must not be falsely marked achieved to bypass that limitation. This repository record is the current task tracker.

Interim containment update: 04/0 diverged, so the combined startup change counts as a harness version. A stale `+100` clearance-grid offset in fine-goal routing was identified; the screen remains unchanged, and this defect must be corrected before any later version or gate claim. See [interim results](go2_navigation_capability_startup_c2_interim_2026-09-26.md).

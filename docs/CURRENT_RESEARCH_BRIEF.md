# Current research brief

The authoritative reference is [LeWMQuad-v3 agent brief: navigation testbed and controller capability, 25 September 2026](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md).

It supersedes the decision-headroom programme. V4.2 is closed; its recommended study-design revision is not pursued. Follow the new brief's authorised scope, gates, budgets and stop conditions. Earlier protocols and results remain historical records.

## Active goal — explicitly started 25 September 2026

**Programme goal:** determine when JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory. Start with a validated testbed, then the E1 benchmark, then moving obstacles. E1 execution and moving obstacles require separate approval.

**Current goal:** establish a validated Go2 simulation navigation testbed. For each controller type (C0 oracle, C1 command history, C2 reactive, C3 JEPA, C4 supervised predictor), determine whether it completes beacon retrieval and return in unseen mazes, report against the pre-registered capability criterion, and produce one example video per controller (C0 optional under the brief).

**Status:** `v0_grid_c3` completed all ten C1 development episodes: 3 round trips (00,03,04), 3 pose losses (01,02,07), and 4 timeouts (05,06,08,09). Every episode initialized; there were zero disallowed contacts, hard or operating-margin violations, or unresolved sampled clearance. All five original planning episodes reproduced their entire native arrays, full requests and consumed sensor hashes exactly. The indexing correction is therefore a contained correctness version and adds no version charge; the defective predecessor C2 consumed one of six. The 9/10 navigation threshold was not met, so the oracle gate remains unstarted. Current work is regenerated-frame diagnosis of pose loss and the remaining navigation failures, before choosing one shared recovery/behaviour change. Tracker estimation changes require approval. The programme remains incomplete.

This replaces the 7 September navigation goal and the closed decision-headroom programme; neither is to be resumed. Machine-readable active-goal state is in [go2_navigation_capability_active_goal_2026-09-25.json](go2_navigation_capability_active_goal_2026-09-25.json). The thread goal tool refuses to replace its unfinished legacy goal; it must not be falsely marked achieved to bypass that limitation. This repository record is the current task tracker.

The completed diagnostic screen and current indexing/containment work are recorded in [the grid-correction progress report](go2_navigation_capability_grid_c3_progress_2026-09-26.md). Earlier interim reports remain historical records.

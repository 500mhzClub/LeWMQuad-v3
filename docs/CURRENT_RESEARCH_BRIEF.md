# Current research brief

The authoritative reference is [LeWMQuad-v3 agent brief: navigation testbed and controller capability, 25 September 2026](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md).

It supersedes the decision-headroom programme. V4.2 is closed; its recommended study-design revision is not pursued. Follow the new brief's authorised scope, gates, budgets and stop conditions. Earlier protocols and results remain historical records.

## Active goal — explicitly started 25 September 2026

**Programme goal:** determine when JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory. Start with a validated testbed, then the E1 benchmark, then moving obstacles. E1 execution and moving obstacles require separate approval.

**Current goal:** establish a validated Go2 simulation navigation testbed. For each controller type (C0 oracle, C1 command history, C2 reactive, C3 JEPA, C4 supervised predictor), determine whether it completes beacon retrieval and return in unseen mazes, report against the pre-registered capability criterion, and produce one example video per controller (C0 optional under the brief).

**Status:** resumed read-only diagnosis and completed all five command-tape sensor replays; stopped before a new cohort because full C0 RGB-D fallback projects 207.07 GiB against 101.32 GiB usable. C1–C4 passed bitwise sensor replay; C0 lacks original sensor evidence, although its native trace replay is exact. The corrected fixed-world reader changes no pilot verdict. All 180 registered episode geometries pass; the shared visual-start reference fix and post-fix test are still unfinished. No oracle gate or validation cohort has started. See [the 26 September checkpoint](go2_navigation_capability_reference_retention_erratum_2026-09-26.md). The goal is incomplete; gates and caps remain unchanged.

This replaces the 7 September navigation goal and the closed decision-headroom programme; neither is to be resumed. Machine-readable active-goal state is in [go2_navigation_capability_active_goal_2026-09-25.json](go2_navigation_capability_active_goal_2026-09-25.json). The thread goal tool refuses to replace its unfinished legacy goal; it must not be falsely marked achieved to bypass that limitation. This repository record is the current task tracker.

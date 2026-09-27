# Current research brief

The authoritative reference is [LeWMQuad-v3 agent brief: navigation testbed and controller capability, 25 September 2026](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md).

It supersedes the decision-headroom programme. V4.2 is closed; its recommended study-design revision is not pursued. Follow the new brief's authorised scope, gates, budgets and stop conditions. Earlier protocols and results remain historical records.

## Active goal — explicitly started 25 September 2026

**Programme goal:** determine when JEPA representations carry decision-relevant information for navigation beyond kinematic prediction, reactive control and explicit memory. Start with a validated testbed, then the E1 benchmark, then moving obstacles. E1 execution and moving obstacles require separate approval.

**Current goal:** establish a validated Go2 simulation navigation testbed. For each controller type (C0 oracle, C1 command history, C2 reactive, C3 JEPA, C4 supervised predictor), determine whether it completes beacon retrieval and return in unseen mazes, report against the pre-registered capability criterion, and produce one example video per controller (C0 optional under the brief).

**Status:** V4 passed the first C1 screen:10/10 round trips, zero contacts, hard/operating violations or unresolved native clearance. The same frozen harness is running the ten second-episode C1 checks; it needs9/10 before the C0 gate. Five of six outcome versions used. Updated projection103.61/160h includes15% contingency; storage fits. Continue in-session at milestones. See [the V4 screen result](go2_navigation_capability_completed_support_v4_screen_result_2026-09-28.md).

This replaces the 7 September navigation goal and the closed decision-headroom programme; neither is to be resumed. Machine-readable active-goal state is in [go2_navigation_capability_active_goal_2026-09-25.json](go2_navigation_capability_active_goal_2026-09-25.json). The thread goal tool refuses to replace its unfinished legacy goal; it must not be falsely marked achieved to bypass that limitation. This repository record is the current task tracker.

The completed diagnostic screen and current indexing/containment work are recorded in [the grid-correction progress report](go2_navigation_capability_grid_c3_progress_2026-09-26.md). Earlier interim reports remain historical records.


**Completed additional check (20/20 passed; five normal recordings exact):** after the unchanged C1 screen, execute the [startup-output contract](go2_navigation_capability_paired_floor_output_contract_plan_2026-09-26.json) on all 20 development starts, with numerical tolerance frozen from the old primary-path calibration before V1 comparisons. Compare complete 00/03/04/05/07 recordings with C3. Any failed or unresolved start must be reported before further changes. The current version also changes normal-start initialisation, so its charge stands regardless of the conditional correctness exemption.


**Gate-sequence amendment (27 September):** after C1 first reaches 9/10 on the ten first episodes, run the same harness on all ten second development episodes and require another 9/10 before C0. If that check fails, every subsequent version screens all 20 episodes with an aggregate 18/20 requirement. Existing safety rules and caps apply; the running V2 screen is unchanged. See [the approved amendment](go2_navigation_capability_second_episode_gate_amendment_2026-09-27.md). The old gate entry point must not bypass this requirement.

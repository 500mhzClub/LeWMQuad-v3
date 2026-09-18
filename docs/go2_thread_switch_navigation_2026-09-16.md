# Parent thread scheduling experiment

Run one complete supervised-rollout mission on exposed short-pulse maze 1,
changing only the parent Python thread switch interval from its recorded
default to 1 ms. Use the original obstacle observer, frozen pulse-trained
model, original CPU allocation, 4800-tick budget, 2-mm depth noise, ideal gyro,
300-ms dispatch delay, command cadence and all existing freshness/clearance
rules. Spawned worker interpreter settings remain at their defaults. No heavy
concurrent analysis during simulation. Preserve the result, including failure.

Compare against the completed fresh observer-grouping reference. Its median
planning service was 106.859 ms and obstacle service 28.386 ms; it had 726
stale triggers, 885/1200 on-time plans, no arrivals and no contacts. The packed
observer follow-up did not improve navigation. A public startup-decision
profile ran the planner in isolation in about 25 ms, including about 7 ms of
neural inference, while the isolated observer also ran appreciably faster
than native stage service. This is evidence motivating a contention test,
not proof of the GIL as the cause. The startup profile does not reproduce
later route state or concurrent host load.

Primary outcomes are physical arrivals, contacts, progress and actual command
continuity. Also compare measured stage service, acquisition, late plans and
original causes of latched stops. A speedup alone is not navigation success.
This single exposed layout and fixed historical-reference order cannot
establish reliability or isolate JEPA's contribution. Native timestamps and
measured-service charging remain unchanged; no real-time or hardware claim.

Launcher: `scripts.run_go2_thread_switch_navigation_development`.
Evaluate with `--evaluate` after actual owner exit and persistence. Output:
`go2_thread_switch_1ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.

Launched in session 93023, PID 3907601. The native launch message and live
owner were confirmed at 19 seconds elapsed. Available recording headroom was
5,057,183,744 bytes. No heavy concurrent replay or analysis is running. Poll
this owner/session; do not restart on quiet output. Evaluate only after owner
exit and complete persistence. The goal remains active and incomplete.

The retained startup profile's corresponding first native planning service
was 95.690 ms (the full-run median was 107.296 ms). The approximately 25-ms
isolated repeated profile includes profiling overhead and excludes live host
contention; it is not a bitwise recreation of the whole native controller.

Completed and physically evaluated after owner exit and full persistence:
no arrivals, no contacts, budget exhausted at 480.96 simulated seconds.
All 1200 selections used the supervised treatment; 919 plans were on time
and 281 late. There were 787 stale-triggered latched windows, 780 selecting
non-hold actions, accounting for 13,867 latched request intervals. Only 3324
request intervals were nonzero, versus reference's 3805. Minimum physical
goal distance was 1.581 m, final 1.726 m, final home distance 2.451 m, and
path length 5.236 m. The opening survey lasted 155.6 s.

Median obstacle/planning/acquisition service was 28.537/104.008/82.557 ms;
median obstacle observation-to-completion remained 114 ms (95th 124 ms).
The change did not improve navigation or command continuity. It does not
establish that every form of host contention is absent. Complete comparison
and original stop-cause readouts are saved in the run root as
`thread_switch_comparison_v1.json` and `dispatch_stall_diagnosis_v1.json`.
The next experiment explicitly tests a 250-ms obstacle age bound while
retaining actual-age charging in the stopping connector; see
`docs/go2_pipeline_age_navigation_2026-09-16.md`.
Of the 787 stale triggers, 786 occurred at age 220 ms and one at 300 ms.
Every plan still reported a clear candidate. All non-depth outcome/timing
records are retained; diagnosed raw depth was retired under the retention
policy after evaluation. No pending raw-depth replay uses this case.

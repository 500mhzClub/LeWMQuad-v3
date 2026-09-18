# Camera-cadence visual support recovery

Run one prospective full supervised world-model mission on exposed short-pulse
maze 0. Keep the failed prefix-aware attempt's model, sensors, tracker, support
thresholds, prefix-aware terminal transition, command/clearance checks, arrival
requirements and 4800-tick budget. No pose matching threshold changes.

Move the existing local supported-view state update from 400-ms planning
observations to every accepted 100-ms registered camera observation. Attach a
copied recovery state to that exact observation; a later update cannot affect
an earlier plan. References remain local (0.20 m) and recent (10 s) when a
recovery starts. Keep physical view memory across mission phase changes because
its measured initial-body coordinates do not depend on goal versus home.
An unavailable registered pose supplies no new view or recovery command.

The preceding public replay exactly reproduced 1,134 accepted raw poses before
failure at 1134. Both cameras were weak at 1127, but a transient count of 59
at planning frame 1128 hid it from the earlier implementation. The new rule
retains the 1127 warning in 1128's planning evidence. This yields an earlier
possible dispatch on the saved tape, not proof that an alternative action would
preserve tracking. The native experiment tests that hypothesis end to end.

Thirteen focused checks passed, including a warning between planning frames,
unchanged historical evidence after later recovery, future-evidence rejection,
local/recent view rules, terminal-prefix behavior and age/stopping gates.
Preflight: 16 physical/32 logical CPUs, 0.6% CPU utilization, 69.52 GB available
RAM, GPUs idle (discrete VRAM 1.84/34.21 GB), 4.576 GB artifact space. Run one
native owner on CPUs 0–7,16–23 using software EGL; pause heavy concurrent work.

Launcher: `scripts.run_go2_framewise_visual_support_development`.
Root: `go2_framewise_visual_support_supervised_rollout_noise_2mm_native_layout00_4800_v1_attempt_001`.
Preserve all outcomes, evaluate physical arrivals/contact and actual treatment
after owner exit and complete persistence. This is exposed-maze development;
new-layout reliability and causal JEPA/RGB advantages remain unestablished.

Launched session 84858, owner PID 3922725. Native launch and live owner
confirmed at 14 seconds elapsed. Poll this same session until persistence and
owner exit before evaluation. The full navigation goal remains incomplete.

## Completed failure

Owner exited 1 after tracking loss; all 948 camera acquisitions were saved.
Physical evaluation: no arrivals or contacts, 945 registered poses, median/max
position error 2.147/12.306 mm, 229/236 on-time plans. The attempt fails the
navigation objective despite accurate tracking before the loss.

Fifteen recovery triggers occurred across 36 planning observations. At the last
three planning frames, support counts were 24/57, 31/32 and 0/22, but no recovery
state was available. The last recorded recovery target was measured at 65.0 s;
it was 30.9 s old and 0.0711 m from the last accepted pose. The ten-second
eligibility limit would exclude this still-local target. The final ordinary
frontier selections continued alternating turns until tracking failed.

The next experiment removes only that age limit for a local view objective.
It does not treat the old image as current support or obstacle clearance;
current measured pose, fresh obstacle evidence and all existing forecast/dispatch
checks remain. Keep this failure and full sensors for further diagnosis.

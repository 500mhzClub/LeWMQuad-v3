# Obstacle grouping timing experiment

After all fourteen short-pulse comparison missions finish, run two sequential
missions on exposed short-pulse maze 1: original obstacle observer, then packed
cell observer. Both use the frozen pulse-trained supervised model, original
controller, 4800-tick budget, 2-mm depth noise, ideal gyro, 300-ms dispatch delay,
and the original maze-1 CPU allocation. Do not run spare-core replay, training
or performance analysis during either mission. Keep both outcomes, including
timeouts, tracking failures and contacts. No tuning between the two missions.

The candidate changes integer occupied-cell grouping only. It preserves the
observed cells, floor fitting, missing-camera rules and 200-ms freshness limit.
Its actual class reproduced all outputs/receipts on 2532 retained direct-run
frames and four synthetic primary-dropout frames. The reference is repeated
because the historical supervised run included concurrent spare-core analysis;
comparison with that historical run alone would not isolate grouping code.

Evaluate physical arrivals, contacts, progress, obstacle and planning latency,
late plans and the original cause of each latched command veto. The primary
question is whether faster equivalent obstacle processing improves actual
command continuity and closed-loop progress. Do not interpret lower component
latency or a successful replay as navigation success. A single execution per
condition on one exposed maze cannot establish general reliability or isolate
host/order variation. This is a repair experiment, not a JEPA attribution test.

Use `scripts.run_go2_obstacle_grouping_navigation_development --condition
reference` then `--condition packed`; use the same command with `--evaluate`
only after its native owner exits and recording persistence completes.
Both outputs are separate new development attempts. The fourteen-run launcher
and its original outcomes remain unchanged.

Recorded deadline diagnosis before the pair: across reactive, command-history,
pose-command, supervised and direct maze-1 runs, no stale dispatch had a newer
obstacle stage logged strictly before that request. Newer stages logged at the
same timestamp numbered 90/217/240/224/222 respectively; those ties do not
establish delivery before dispatch. Direct's next result arrived 2–6 ms after
538 of its 790 stale triggers; supervised's did so for 375 of 614. The remaining
cases include longer delays. This supports testing the several-millisecond
observer speedup while preserving the freshness rule. It does not establish
a queue-delivery bug or counterfactual navigation success. Per-trigger evidence:
`go2_short_pulse_navigation_direct_noise_2mm_native_layout01_4800_v1_attempt_001/`
`stale_obstacle_completion_comparison_v1.json`.

The fourteen-run comparison has completed: three round trips overall, zero
across the six neural missions. Both full result aggregates and the inspected
second-maze figure are saved. The reference condition launched next in session
82456, PID 3902304, with 4,745,621,504 bytes free. Its live process and launch
output were confirmed. No heavy spare-core analysis is running. Wait for actual
owner exit and recording persistence before `--condition reference --evaluate`;
the packed condition has not launched. The original comparison's failures and
remaining geometry/clearance diagnoses remain part of the evidence.

The reference owner has exited zero with complete persistence and physical/
actual-treatment evaluation. It exhausted the 480.92-s budget without arrivals
or contacts. All 1,200 selections matched the supervised treatment; 885 plans
were on time, 315 late. Minimum physical goal distance was 1.459 m, final goal
distance 2.611 m, final home distance 1.593 m and path length 6.738 m. Position
error was 1.362 mm median and 3.948 mm maximum. Maximum simulator lag was
143.682 s. No spare-core analysis ran during native execution; posthoc analysis
started after owner exit.

Initial panorama took 158.8 s, with 390/396 plans on time but only 1,013/7,940
nonzero request intervals. Afterward 495/804 plans were on time; every selected
plan still routed to or viewed frontiers. Eleven frontier events completed;
all 1,200 plans reported a clear candidate. All 726 latched command windows
began with stale observations at age 220 ms, explaining 12,763 stopped request
intervals; 720 selected non-hold actions. Median acquisition/obstacle/planning
service was 82.713/28.386/106.859 ms. The source and readouts are preserved as
`dispatch_stall_readout.py`, `dispatch_stall_diagnosis_v1.json` and
`phase_diagnosis_v1.json` under the reference root.

On 145 matched 700-ms executed windows, neural XY RMSE was 18.894 mm versus
unused pose-command 10.347, command-history 14.901 and nominal 27.770. These
are overlapping trajectory windows, not executed alternative policies. The
reference's unsuccessful outcome shows that removing concurrent analysis alone
did not produce a successful mission in this execution. It does not establish
the causal effect of background load from one historical/repeated pair.

Diagnosed reference depth and the completed reactive maze-1 reference depth
were retired under the retention policy to provide recording headroom. Exactly
14,760 leaves reclaimed 4,732,420,096 allocated bytes, with all 65 JSON hashes
and 14,844 non-depth identities preserved. No pending raw-depth replay or fit
uses either case. Outcomes, timing, RGB/body/commands, poses and physics remain
available for the paired comparison; full original sensor replay is unavailable.

The **packed** condition is now active in session 16740, PID 3904143, under
`go2_obstacle_grouping_navigation_packed_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.
Its live process and launch output were confirmed 39 seconds after launch;
available space was 5,726,306,304 bytes. Keep spare-core replay, training and
performance analysis paused during this mission. After actual owner exit and
complete persistence, evaluate with `--condition packed --evaluate`, then
compare both outcomes, actual stage timings and dispatch interruptions. Do not
restart on a quiet log or assume success from component equivalence. The broad
goal remains incomplete.

The packed owner has now exited and the complete persisted recording has been
evaluated successfully. It exhausted the 480.94-s simulation budget with no
physical arrivals and no contacts. Of 1,200 plans, 917 were on time and 283
were late, compared with reference's 885 on time and 315 late. The native
terminal report counted 760 stale triggers versus reference's 726. Neither
condition completed the mission. The outcome therefore does not establish a
navigation improvement. Actual obstacle-stage timing and treatment wiring
still need diagnosis before attributing any timing difference to the candidate.
The packed raw depth remains available for that diagnosis. Available disk
space after persistence was 1.76 GiB; no native mission is currently running.

Final paired readout is saved in
`go2_obstacle_grouping_navigation_comparison_v1_attempt_001/result.json`.
Reference/packed median obstacle service was 28.386/28.066 ms, planning
106.859/107.296 ms and acquisition 82.713/82.969 ms. Minimum physical goal
distance was 1.459/1.522 m; path length 6.738/6.612 m. Every one of packed's
760 latched windows began at obstacle age 220 ms; 754 selected non-hold
actions. There is no demonstrated navigation improvement.

Source inspection confirmed the selected initializer sets the observer used
by the process runtime; invoking it selects PackedCellObstacles. The launch
source hash matches. No actual worker-class receipt was recorded during the
mission, so this source/wiring check is not a contemporaneous type receipt.
Profiling 48 retained startup frames exercised the intended packed extractor;
its isolated total was 0.591 s versus original 0.659 s with profiling overhead.
The retained startup decision profile took about 25 ms per plan in isolation,
including about 7 ms neural inference. Neither profile reproduces native host
contention or later navigation state. A separate 1-ms parent thread-switch
mission will test the scheduling hypothesis with the original observer.

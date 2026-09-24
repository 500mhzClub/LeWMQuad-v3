# Fixed 20-ms planning latency stress

Run the original persistent-visual controller on exposed layout 3 with the
frozen JEPA model, then the frozen supervised model. Add 20 ms of simulation
time after measured planning work becomes available, before publishing the
plan. Preserve the 300-ms dispatch deadline, 400-ms planning cadence, 250-ms
observation bound, prefix checks, sensors, seeds and 4800-tick mission budget.
No prompt visual-recovery hold is enabled. The injected delay is applied once
per planning job, including jobs whose plans consequently miss their deadline.
Tracking, mapping, obstacle and registration publication are unchanged.

This intervention tests a specified additional planning latency, not calibrated
hardware latency or deterministic replay of the original failure. Actual host
service and acquisition timing still vary. Two missions on one exposed layout
cannot establish statistical reliability or a model ranking. Retain both full
outcomes, physical arrival/contact checks, actual added-delay receipts, missed
deadlines and visual-recovery activity. The original zero-added-delay outcomes
remain separate, including the JEPA failure and later successful repeat.

Launch sequentially with
`scripts/run_go2_visual_recovery_dispatch_hold_development.py --control --planning-extra-ms 20 --arm jepa`
then `--arm supervised_rollout`. Use the original layout-3 CPU allocation.
Evaluate each after owner exit and recording persistence with `--evaluate`.
No heavy parallel analysis during either mission. Hardware/load/storage are
recorded at each launch; current free artifact storage is about 4.82 GB.
Retire completed diagnosed depth between runs under the standing policy.

Outputs:
`go2_planning_latency_plus20ms_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`
and
`go2_planning_latency_plus20ms_supervised_rollout_noise_2mm_native_layout03_4800_v1_attempt_001`.
The learned checkpoints and action scoring remain fixed throughout the pair.
No threshold search, new environment type, sealed evaluation or hardware run
is part of this experiment.

## JEPA result

The first run exhausted its 4800-tick budget with no arrivals and zero physical
contacts. Tracking remained available. Only 56/1200 plans were on time;
22,875/24,025 requests held because no on-time plan was available. Nonzero
commands totalled 19.62 s, including only 1.60 s with translation requested.

All 1200 selected plans have matched actual delay receipts: 20–22 ms extra
publication delay (2-ms simulation step quantization). Median completion age
rose from 290 to 310 ms. On this recorded trajectory 1148 plans were ready
before the original deadline prior to the extra wait, and 1092 crossed the
deadline during the wait. This is a within-job timing diagnosis, not a claim
that removing the wait would preserve the trajectory or ensure navigation.
No visual recovery activated and no pipeline fault occurred. Owner exited 0;
physical outcome evaluation and saved forecast evaluation completed.

The root's `planning_latency_stress_diagnosis_v1.json` contains the actual
treatment and failure diagnosis. Completed diagnosed depth is retired under
the standing policy; all non-depth evidence and original tracking-loss depth
remain. Proceed with the already planned supervised comparison unchanged.

Before the supervised run, retiring diagnosed depth left only 4.16 GB free,
below the unchanged four-GiB launch headroom. Cleared stale, regenerable AMD
compiler objects in `~/.cache/comgr/llvmcache-*` (last modified September 11
or earlier). No model, dataset, runtime result or active CPU simulator cache
was removed. Exact file inventory is
`.generated/comgr_cache_cleanup_2026-09-16.json`.

## Completed paired result

The supervised run also exhausted its budget without an arrival, with zero
physical contacts and tracking intact. Its 1200 planning jobs all received
exactly 20 ms extra delay. Median completion age rose from 290 to 310 ms;
1075 plans crossed the deadline during the added wait. Only 56 plans were on
time. Nonzero commands totalled 19.60 s, including 0.40 s with translation.
Neither stress run activated visual recovery or reported a pipeline fault.

| Model | Initial-view plans on time | Ordinary routing plans on time | Goals / round trips | Contacts |
| --- | ---: | ---: | ---: | ---: |
| JEPA | 52/67 | 4/1133 | 0 / 0 | 0 |
| Supervised | 55/65 | 1/1135 | 0 / 0 | 0 |

Both physical evaluations completed after owner exit and recording persistence.
The aggregate is `go2_planning_latency_stress_complete_v1_attempt_001/result.json`.
Every original and added-delay outcome remains preserved. These are two
additional exposed-layout development missions, separate from the fixed
28-mission baseline comparison and subsequent zero-added-delay repeats.
There is no supported JEPA-versus-supervised ranking from this pair.

The shared pipeline has insufficient timing margin for this declared added
latency: during routing, the extra wait removes almost every usable command
window. This does not establish a calibrated deployment latency envelope or
prove that any one computational stage alone causes failure. Next, measure
the current planning path's main costs, reduce redundant work without altering
forecast/clearance semantics, then prospectively repeat this fixed stress
condition. Do not count component speedups alone as repaired navigation.

Both completed diagnosed depth recordings are retired under the standing
policy; all outcomes, RGB, poses, commands, physics, forecasts, source copies
and added-delay receipts remain. Original tracking-loss recordings remain full.

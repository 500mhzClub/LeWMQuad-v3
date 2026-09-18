# Prospective synthetic wall-stack sampler calibration

Calibrate `WallStackSampler` before any controller profiling. No active controller
process is attached or instrumented. This is a component diagnostic, not a raw
sensor replay, model experiment or native run.

Use three fixed workloads: 800,000 Python integer-update iterations, twelve
NumPy sine-and-sum passes over one million fixed float64 values, and a 50 ms
sleep. Each case receives two recorded warm-up pairs and twenty recorded
measurement pairs. Alternate original/sampled execution order per pair. The
sample interval is 10 ms, with at most 10,000 samples and 128 Python frames per
sample. Include thread creation and shutdown in sampled elapsed time.

Preserve every paired duration, returned-value equality check and complete
sample report, including empty samples and slower results. Compare exact return
values and unchanged array bytes. Report total-time overhead separately for each
workload; no acceptable-overhead threshold or controller speedup is inferred.
Keep actual sample offsets and capture durations rather than assuming a regular
sampling interval. Record the sampler thread's own CPU time separately.

The host is shared with one full replay and one native scene. These measurements
do not establish production overhead. Python/GIL scheduling can bias sampling;
samples include waits and Python locations suspended in native calls. They are
not native stacks, function-call counts or CPU self time. A later controller
diagnostic needs its own frozen protocol, completed baseline admission, original
model/input checks, explicit sampling windows, complete output equivalence and
ended-owner verification. Profiling timings must remain separate from the
unprofiled controller comparison.

Runner: `scripts/calibrate_go2_wall_stack_sampler_v1.py`. Exclusive output:
`docs/go2_wall_stack_sampler_calibration_2026-09-11.json`. Preserve failures;
do not silently replace a failed calibration or choose favorable repetitions.

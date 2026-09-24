# Fixed controller profiling windows on the completed JEPA case

Replay the unchanged ResidualAnchoredContinuationController and exact expanded
JEPA adapter state through original observations 0–404. Admit the completed
first worker terminal
`617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e`
through the existing full original input verifier before and after the profile.
Bind inherited sources to the native waiter's launch
`e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6`.
Never attach a profiler to or mutate a running/queued experiment.

The fixed windows are the first ten planned observations, 3–12, and the first
ten observations of the preserved long hold run, 395–404. These choices are fixed
before measuring function-level timings. Reconstruct every intervening raw
observation and complete original decision. The later window must contain the
original discretionary holds. Do not consume observation 405 or any candidate
trajectory. Use one fresh original controller/model and its original public
mission; no command, policy, input, model weight or gradient may change.

Enable cProfile only around controller.observe for those twenty observations.
Record unprofiled and profiled controller wall intervals separately per frame,
public packet identities and exact original decision hashes. Save both complete
profiler statistics and machine-readable function counts, exclusive time and
cumulative time. Cumulative times overlap and must not be summed. Grouped
module totals sum exclusive time only. Reject nonfinite/negative times and
protected file paths before serialization. No function arguments or sensor
values are included in profiler statistics.

This diagnoses controller hotspots under profiling and the shared workstation
load. It is not an isolated speed benchmark, and profiler overhead is not
removed. Differences between early motion and later holds are not causal
attribution to the action: map size and observation state also differ. Offline
packet loading is outside the profiled region and cannot estimate live renderer
or camera acquisition time. No real-time, hardware or navigation qualification
follows from the profile. Actual observed wall times are recorded separately in
`go2_adapter_first_1000_observed_timing_2026-09-10.json`.

Resource admission: at least four physical CPUs, 48 GiB currently available RAM
and 40 GiB artifact reserve plus one GiB output allowance. Use a single CPU
process with one Torch/OpenCV/BLAS thread and deterministic algorithms. The
current host has 16 physical CPUs; the original supervised native worker and
two-controller raw prefix are distinct active processes, so this bounded CPU
diagnostic can use otherwise available capacity without launching a scene.
Recheck resources after full input admission. Exclusive output
`go2_adapter_controller_windows_profile_v1_attempt_001`; preserve failures with
no silent retry. Source-only preflight creates no runtime output. All existing
experiment priorities and frozen sources remain unchanged.

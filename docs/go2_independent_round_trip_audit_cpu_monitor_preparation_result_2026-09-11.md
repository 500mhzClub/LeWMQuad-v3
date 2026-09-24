# CPU audit monitor prepared

Fresh interpreter inspection found Torch 2.12.0+rocm7.2 with CPU default
device, no initialized CUDA context and uninitialized Genesis, while OpenCV
reported OpenCL enabled. Inspection session 79855 exited 0; the same states
were observed again during preparation. This is an enabled runtime setting,
not evidence that a previous audit used GPU computation.

Reviewed audit paths show `renderer_audit` reads saved camera/depth/renderer
witness records rather than invoking rendering. Its sensor audit calls
`check_capture_domain`, whose body checks transforms/frustum bounds with NumPy.
The assigned checkpoint reader explicitly validates CPU model and optimizer
tensors. These inspected paths do not constitute a complete transitive runtime
qualification of the auditor.

Implemented `scripts/independent_round_trip_audit_cpu_monitor_development.py`.
The monitor disables OpenCL within one checked scope, inspects PyTorch tensor
arguments, results and device requests, and profiles Python/C calls to reject
Genesis activity, accelerator initialization, OpenCL changes and new Python
threads. Violations remain recorded even if the caller catches the first
exception. Scope exit checks state and restores the original OpenCL setting.
This is instrumentation of reviewed code, not OS device isolation.

**13 tests passed in 3.20 seconds**, session 3088, exit 0, with eight imported
Torch JIT deprecation warnings. Tests include real CPU arithmetic, rejected
meta-device requests, an intercepted actual Genesis scene entry without
initializing Genesis, caught violations and cleanup behavior. No trained model,
native scene or original raw audit was executed.

Preparation verified **2,037 source bindings**, session 86524, exit 0:
`docs/go2_independent_round_trip_audit_cpu_monitor_preparation_2026-09-11.json`,
SHA-256 `4b7877693272a42910f5b79d06d58b9e1a468ef7a6aad68dadf78e6b743ab97f`.
The existing driver and audit sources remain unchanged. Full raw-auditor
qualification and overlap permission remain false.

Next is a bounded replay of the existing four-arm factory startup under the
monitor, checking complete saved decisions, model outputs and state against
the original completed startup. That check must consume only the old four
packets per arm and execute no new command. Integration around the complete
separate audit and a source-bound overlap verifier still follow it.

At closing, original audit worker 2743870 (creation 1789071424.56) and combined
replay 2766980 (creation 1789083663.49) remained live. The combined replay was
still in full input admission, before creating its output root. No job was
replaced or restarted.

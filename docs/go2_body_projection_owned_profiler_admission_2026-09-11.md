# Owned-child external profiler preparation

The original `marker_smoke_v1` remains a terminal tooling failure: py-spy
returned 1 with `No child process (os error 10)`. Its failure receipt and all
outputs remain unchanged. It is not a successful profiling result.

An [upstream report](https://github.com/benfred/py-spy/issues/759) describes
competing waits for a child process as a cause of this error. That is consistent
with our failure but does not independently prove its cause. The
[pinned 0.4.2 source](https://github.com/benfred/py-spy/blob/v0.4.2/src/main.rs)
uses different lifecycle handling for `--pid` and command-launch modes.

The V2 synthetic probe uses a separate runner that launches and reaps both its
own Python child and its own profiler. It accepts no external PID. The child
uses Linux `PR_SET_PTRACER` to permit its parent and that parent's descendants;
it does not grant access to arbitrary processes or change system ptrace
settings. This scope follows the
[kernel's Yama documentation](https://www.kernel.org/doc/html/latest/admin-guide/LSM/Yama.html).
The profiler samples only the newly created child. The runner waits for both
readiness handshakes before starting the synthetic workload.

The successor also corrects the fixture: a Python `synthetic_observation`
function calls sleep below each observation marker. The original fixture called
sleep directly, which could leave only the marker itself in a Python stack.

The V2 probe completed with both exit codes zero, both children reaped, 324
total samples, zero reported sampling errors, and 296 samples within the 30
declared observation markers. All 30 contained the intended Python descendant.
The full parser validated the trace after both processes ended.

- Runner: `scripts/run_go2_external_marker_owned_child_smoke_v2.py`.
- Child: `scripts/external_body_projection_marker_smoke_child_v2.py`.
- Root: `.generated/tools/go2_py_spy_0_4_2_v1/marker_owned_child_smoke_v2`.
- Result SHA-256:
  `466c8f94e34616a3d67012bce9f498238e04f9c9bbe332000d0742c3d1016f3f`.
- Tool session 20653 exited zero. Commands, source hashes, process identities,
  boot, profiler binary hash, stdout, stderr and complete trace are retained.

This establishes synthetic marker coverage and successful process ownership on
this host. It does not qualify profiler overhead, unbiased attribution, actual
controller profiling, navigation, real-time operation or hardware deployment.
The earlier full CPU replay slot is free; no external full replay has launched.

## Admission integration

`scripts/body_projection_external_profile_admission_development.py` invokes the
unchanged body-projection completion checker with private CLI/output bindings.
It supplies both exact original result and execution hashes, captures exactly
one receipt in memory, and reproduces the original writer's logging digest.
Every original source, ended-owner, raw/model, report, full timing population,
and predecessor check still executes. The complete receipt must equal the
frozen original except its UTC timestamp. Returned artifacts are rehashed after
reading. The existing completion receipt is never overwritten.

Fourteen focused tests passed in 2.10 seconds, session 65941. They check private
bindings, both mandatory arguments, malformed/missing/duplicate receipt
rejection, original live-owner rejection, logging serialization, changed
negative evidence rejection and post-read authentication. These synthetic tests
do not themselves establish actual raw/model input admission.

The next actual admission check is read-only and does not launch a controller.
The eventual full replay still requires its own prospective execution protocol,
bound parent/child/tool identities, admission before and after execution, all
1,428 original decisions and inputs, 1,425 forecasts, seven state witnesses,
complete marked-stack coverage and terminal verification. Profiled elapsed time
must not be presented as an unprofiled speed comparison. Preserve sensing
failure 1173 and all negative qualification flags.

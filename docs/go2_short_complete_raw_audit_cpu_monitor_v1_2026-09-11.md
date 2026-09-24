# Complete old short-episode CPU audit probe

Replay the complete original `all_phase_full_jepa_residual_maze_02` audit from
`go2_all_phase_residual_maze02_matched_native_v1_attempt_001` under the frozen
AuditCPUMonitor. The fixed episode has 14 observations, 1,400 physics samples,
13 command intervals and the original SENSOR_OR_MODEL_FAILURE termination.
Its full saved audit, including failed navigation, must match exactly.

Authenticate the original completed result, launch, source bindings, all raw
outputs and bound model inputs before and after. Use the original assigned
AllPhaseTranslationBiasModel and call the original auditor directly. Model
loading, geometry construction, sensor reconstruction, controller replay,
command verification, physics, raster visibility, renderer witnesses and
round-trip evaluation all execute inside the monitored scope. No original
function globals, controller behavior or input artifacts are changed.

Use one fresh CPU process, hash seed zero and one OpenCV/BLAS/Torch thread.
Set OPENCV_OPENCL_RUNTIME=disabled before importing Python modules. Retain
the closed monitor even when the auditor fails, and retain any changed report
before rejecting it. A monitor failure is terminal for this exclusive output
root; no automatic retry or replacement is allowed.

This bounded replay may run alongside the existing native worker and the
single admitted full controller replay. It does not launch a native scene or
consume new independent-layout data. Require 32 GiB available RAM and 41 GiB
free artifact storage before launch.

Passing establishes monitored execution of this complete old audit only.
It does not execute the independent multiarm auditor, the reactive command
audit or successful learned planning on a long trajectory. The existing
four-arm factory startup supplies separate bounded evidence for actual model
and controller paths. Full independent-study coverage, monitoring integration,
source-bound overlap admission and final population policy review remain
separate work. This instrument is not an OS device-access sandbox. No claim
of navigation success, real-time performance, hardware qualification or
collection/audit speedup follows from this probe.

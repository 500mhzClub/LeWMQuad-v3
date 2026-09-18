# CPU audit runtime instrumentation

A fresh interpreter inspection found Torch's default device was CPU, its CUDA
context was uninitialized and Genesis was uninitialized, but OpenCV reported
OpenCL enabled. That flag does not prove any prior auditor used the GPU.
Explicitly disabling OpenCL is necessary for this monitor's scope claim.

The one-use monitor requires a fresh single-Python-thread scope without an
existing profiler, CUDA initialization or Genesis initialization. It disables
OpenCV OpenCL, installs a Python/C-call profiler and a PyTorch dispatch mode,
and restores the original OpenCL setting when the scope closes. Tensor inputs,
outputs and device requests must be CPU. Genesis runtime calls, CUDA/XPU lazy
initialization, native CUDA/XPU initialization, OpenCL setting changes and new
Python threads are rejected. A caught violation still fails scope exit.

This is instrumentation for the reviewed fixed runtime, not OS device
isolation or protection against arbitrary hostile native extensions. Native
CPU library worker threads are not individually profiled. The profile hook,
thread count, default device, accelerator, Genesis and OpenCL states are
checked at exit. Explicitly changing or removing the profile hook invalidates
the result. The recorded operation counts refer only to the monitored scope.

Unit tests exercise small real CPU tensor operations, rejected synthetic
Genesis calls and intercepted real Genesis scene entry without initialization,
actual meta-device requests, OpenCL changes, new Python threads
and profile removal. They do not execute an original raw audit or establish
overlap permission. A real bounded factory/controller replay, complete raw
audit integration, frozen source admission and actual overlap qualification
remain outstanding. Existing worker, driver and native queue sources are not
modified by this monitor preparation.

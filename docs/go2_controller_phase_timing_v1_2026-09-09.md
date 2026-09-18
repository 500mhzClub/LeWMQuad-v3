# Exact-prefix controller phase timing diagnosis V1

The live ninth controller is still well above100ms per observation. Earlier
cProfile measurements incurred per-function overhead and nested costs. Measure
the current later-floor-resolution controller with explicit coarse phase timers
to identify remaining latency costs before choosing an optimization.

Exclusive output go2_controller_phase_timing_v1_attempt_001. Reconstruct the
completed960-frame later-floor-resolution prefix, result
a4e34e4ca8b72422f8fc6c1ca54b33a80c9821ff076fbc431d0575f0a1701fbe, using its
original paired public packets from the completed eighth native attempt. The
fresh controller must match every complete reference decision exactly and stop
after frame959, before its unexecuted successor outcome.

Use separate subclasses and instance-local forwarding proxies. They time only
original calls and preserve original arguments, returns, writes and exceptions.
No global profiler, monkeypatch, source replacement or change to a running
controller. Timed phases include motion, registration, map observation/waypoint,
primary insertion/classification/coverage, auxiliary insertion/confirmation,
six candidate contact queries, selection, controller advancement and receipt
construction. One pair of hooks on the fresh model times its outer forward call
without touching tensors or parameters; always remove hooks, including errors.
Six focused tests passed, including exact synthetic public-packet decisions,
nested duration accounting, return identity and failure cleanup.

Record inclusive and child-subtracted exclusive durations and call counts.
Check that exclusive durations sum to the outer controller duration each frame.
Timer overhead remains included; this is phase attribution, not an uninstrumented
or controlled paired speedup measurement. JSON normalization/comparison and
packet reconstruction are outside the controller timer. No native acquisition,
command dispatch or physical outcome is timed by this diagnostic.

Bind prefix, original input artifacts, model admission and sources before/after.
Require exact unchanged model state and no gradients. Run --preflight-only and
inspect actual CPU affinity/topology/utilization, RAM, GPU/VRAM, competition and
storage. One CPU replay process and one numerical thread beside the existing
single scene;8GiB available RAM and128MiB output above40GiB reserve. No scene,
GPU switch, training, model selection, navigation or real-time qualification.
Freeze diagnosis sources at launch and preserve any failure.

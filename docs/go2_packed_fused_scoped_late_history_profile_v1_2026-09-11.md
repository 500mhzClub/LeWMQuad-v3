# Remaining packed/fused controller cost profile

The completed packed-index composition reduced total controller time by 9.38%
on 1,425 paired planning observations, with exact decisions and seven retained
state checkpoints. Median time remains about 665 ms; all calls exceeded 100 ms.
This profile identifies remaining costs before choosing another implementation
change. It is not an additional speedup experiment or a navigation trial.

Rebuild the same 1,428-observation old development history with one fresh
PackedFusedScopedController and the same frozen full-RGB JEPA model. Reuse the
original profile loop, changing only the controller constructor, complete
decision normalizer, output root and progress label in isolated function globals.
Keep cProfile enabled only for frames 3–12, 395–404 and 1418–1427. Capture the
existing bounded-depth state-size summaries. All actual profiled decision and
public-input identities must match the completed packed replay, and all model
weights and input arrays must remain unchanged.

Authenticate the completed packed replay and its fused/combined predecessors.
Rehash the actual old raw worker artifacts and bound model inputs before and
after. Reuse the completed original full training admission, without rerunning
training ancestry or consuming another native episode. Preserve the known strict
visibility failure at frame 1173 and the original failed round trip.

One full CPU replay slot is used, after packed replay owner 2786620 has ended.
The original one-model profile resource bounds apply: at least 48 GiB available
RAM, 41 GiB artifact space and four physical CPUs. Python hash and numerical
libraries use the existing fixed single-thread settings and OpenCL is disabled.
An exclusive attempt stores the launch, every comparison row and three pairs
of pstats/JSON summaries. Failures are terminal and preserved without retry.

Cumulative profile costs overlap and must not be summed. Profiler overhead is
not removed. Controller work alone is profiled; this is neither an isolated
benchmark nor complete sensor/command latency. State sizes are not additional
state-equality proof. No native commands, new sensor collection, policy change,
qualification, promotion, hardware evidence or goal completion follows.

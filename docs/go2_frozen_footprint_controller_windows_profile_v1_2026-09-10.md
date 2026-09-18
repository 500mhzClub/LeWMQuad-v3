# Profile the verified optimized controller at fixed windows

The completed frozen-footprint paired replay reduced total controller time by
13.50%, but its post-warmup median was still 663.82 ms and every measured call
exceeded 100 ms. Profile the remaining costs before choosing another
optimization. The existing native queue and all completed evidence remain
unchanged.

Use the exclusive root
`go2_frozen_footprint_controller_windows_profile_v1_attempt_001`. Admit the exact
completed paired result
`4fb9363938101682473115563f4aca2f48a68a8a6d8e9ead2cf3a8ea2ddae772`
and independent verification
`4b574cf6a46943b7135ec11c472f7c1c73a01af48411178a9a4917e0dd5681c8`,
including their complete source and artifact bindings. Use the same completed
original full JEPA adapter case, model and 405 raw observations, indexed 0–404.
The candidate implementation is the exact completed frozen-footprint controller.

Enable cProfile only around controller observe for observations 3–12 and
395–404. Disable it before normalizing the two declared controller metadata
fields, serializing records, hashing evidence or comparing decisions. Record
both the complete candidate decision hash and original normalized decision
hash. Require every normalized decision, original completed command endpoint,
public sensor array and model state to match, with no gradients and 402
forecast-bearing observations. Reconstruct the discretionary holds in the
unchanged later window. Consume no observation 405. Any mismatch is a terminal
failure, with no automatic retry.

Produce two raw .prof files, their complete guarded function/module summaries,
the 405 comparison records, launch and final result. Retain exclusive and
cumulative timings separately; cumulative times overlap and must not be summed.
The profiler adds overhead. It does not produce a new isolated benchmark,
replace the completed unprofiled paired timing, profile acquisition or prove
real-time behavior. This diagnostic run does not add independent retained-state
checks; the completed paired replay owns those checks.

Verify the original worker's full input chain before and after profiling and
recheck the completed predecessor afterward. Bind and recheck the new recursive
source closure. Require at least 48 GiB available RAM, 41 GiB artifact headroom
and four physical CPUs. Use one CPU profiling process with the original native
worker; construct no native scene, train no model and change no frozen policy.
Source-only preflight must not admit the original worker, create outputs or
execute the profiling loop. No navigation or hardware claim follows.

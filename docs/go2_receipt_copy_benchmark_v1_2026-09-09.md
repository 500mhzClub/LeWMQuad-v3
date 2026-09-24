# Receipt-copy component benchmark V1

Compare ordinary deepcopy and the separate copy_receipt helper on the saved
new_selection records at fixed common-floor prefix frames20,60,100,959. Warm
both functions once per record, then run eight interleaved repetitions per
function, alternating which runs first. Time only copying; outside the timed
region require identical canonical JSON bytes. Record all timings and input
sizes. Input records were already serialized, so this benchmark does not test
the alias distribution of live objects; separate tests cover aliases, cycles,
NumPy independence, subclasses and custom memo semantics. The helper uses fast
paths only for exact builtin containers and atomic leaves, with standard
deepcopy fallback for other types. It does not replace any imported function.

Exclusive root go2_receipt_copy_benchmark_v1_attempt_001. Bind the completed
common-floor result/artifacts and source closure, plus benchmark/helper/tests
and this protocol, before and after. Minimum4GiB RAM,128MiB output above40GiB
reserve, one CPU process/thread. This brief immutable-input component benchmark
may run beside the native experiment and cache replay; record competition.
No controller integration, model loading/training, native scene, new navigation
outcome or full-loop speed claim. Preserve output/failures, no source changes
after launch. Any integration must be separate and pass full decision replay.

# Combined insertion/query optimization: paired early-prefix controller timing

Prepare the same controlled first256-observation comparison as the completed
packed-owned paired benchmark, replacing only that candidate with
SinglePassLaterFloorController and admitting the actual completed single-pass
full1881-decision replay. Original baseline remains
LaterFloorResolutionRoundTripController. This compares the combined packed
insertion and single-pass query changes with the original controller; it does
not isolate the query change from insertion in a whole-controller comparison.

Run scripts/benchmark_go2_single_pass_controller_pair_v1.py only after full
replay completion, with its actual --replay-result-sha256; use --preflight-only
first. Destination:go2_single_pass_controller_pair_v1_attempt_001. Two independent
fresh model/controller states, same fixed seed/weights/numeric settings,
alternating first arm each observation, private identical public-input copies
outside timing. Require all256 complete decisions equal the recorded original,
input immutability, identical production gzip receipt streams and unchanged
model states. Full source/input/artifact validation before/after is required.

Time complete observe plus the production JSON/gzip/flush receipt writer, with
wall and process CPU measurements, all per-frame results and100ms deadline
misses. Decoding, input copying, comparison, real acquisition and physics are
excluded. Warmup/holds/failures remain in the fixed population. No timing claim
for later map growth or the full native loop. Preserve the result even if the
candidate is slower or every frame still misses the deadline.

One CPU process with two resident controller/model states and one numerical
thread beside the existing native settling scene if resources permit. Inspect
topology/affinity, CPU/GPU/VRAM, RAM, competition and both volumes; require16GiB
available RAM and1GiB output above40GiB reserve. Capacity admissions are not OS
limits. Record shared-machine competition; alternating order does not eliminate
cache, thermal, frequency or scheduling effects. No native source modification,
automatic adoption, training, new simulation, navigation/generalization,
real-time or hardware qualification follows from this bounded benchmark.

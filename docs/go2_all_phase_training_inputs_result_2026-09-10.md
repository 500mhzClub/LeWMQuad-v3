# Expanded causal input validation result

Complete result:
ef48950b7987eaf9310ba8124a00c2e6e13c9b84b7cda6a362ac7ddb6ecd63fb.
Root go2_all_phase_training_inputs_v1_attempt_001.
Status ALL_PHASE_TRAINING_INPUTS_V1_COMPLETE. Original checker80757,
PID2628968/starttick133510189, exited0 in314.06464697094634s.

All4010 available training contexts were independently materialized through
both the past-only inference path and the separate private-training-future
path. Their input tensor identities match exactly. All408 shared original
available training inputs exactly reproduce their authenticated RGB/body/
control hashes, normalized known commands, validity masks and target clocks.
All4800 planned slots remain accounted for, including790 unavailable contexts.

All saved reader scopes were independently checked against the bound target
rows and consumed policy paths:32080 past-packet reads across the two paths,
28444 private training future-packet reads, zero inference future-packet reads.
There are4694 unique consumed policy leaves. The stream opens no native
physics, geometry or contact artifacts; native labels remain training targets.
No geometry-transfer future packet was materialized.

One materialized sample contains1,835,312 tensor bytes; all4010 would occupy
7,359,601,120 bytes (about6.85GiB), excluding Python/process overhead. The checker
used zero sample cache. Cache mutation/eviction tests passed separately. Choose
fitting concurrency from measured capacity and throughput, not this byte count
alone; preserve the live native queue's resource allowance.

All three existing untrained conditions, direct/supervised_rollout/JEPA,
accepted a six-sample batch with finite gradients for every active parameter,
unchanged matched model state and no EMA-target gradients. No optimizer was
created, no parameter updated, and no fitted checkpoint produced. Forty-five
focused tests and the preparatory actual-input checks are recorded in the
preparation document. This result validates data interfaces, not learning or
navigation performance.

Exact output bindings:
- launch.json:b8eb7241b54842dd10d9a55d97752a6ad8f4e4514e8bcf5e32b4551e3da1e090
- tensor_index.json:4b98308745d8b19b79b4ec94329c4c5cfefb387693625bb544196fe770b43baf
- model_contract_checks.json:c1df43a8d08d956a26b5c61fc360a929013b4ee159717a10596903c39f24b8dc

The launch contains two incorrect inherited parent-stage flags:
future_rgb_materialization=false and future_rgb_materialized=false. Its explicit
current checker protocol and private_training_future_materialization=true
required the actual training-only future reads. The unchanged original launch
and result are accompanied by
docs/go2_all_phase_training_inputs_scope_correction_2026-09-10.json,
SHA8fcfdc678b721a84ed53c2545ec55284e2185d7b1e6bc5ca0dc26da4832e2dca.
That correction binds the exact result, artifacts, protocol and source code,
records both original erroneous flags and the verified actual access scope,
and changes neither the scientific definition nor any original artifact.
Admit this correction explicitly alongside the result before fitting; do not
silently interpret the uncorrected metadata as internally consistent.

Independent completion verification checked1123 sources before/after, all
checker outputs, original/expanded target and old-input bindings,4694 consumed
policy leaves, every saved scope against its target row, every shared old input
witness and all three unchanged-parameter contract receipts. It performed6961
hash operations over322,379,839 bytes. This separate check did not repeat full
tensor materialization or the model-gradient calculation. The original checker
completed those operations and its own before/after source/artifact checks.

Final capacity:75,882,504,192 available RAM bytes,668,579,143,680 artifact free
bytes,3.3% sampled CPU utilization, zero GPU utilization. The original maze1
worker remained live in raw audit; no extra native scene was launched.

Next work:
1. Add a combined study view/stream with these4800 training slots and the
   original456 geometry-transfer slots (420 available). Preserve the original
   transfer stream and local-index mapping; never send transfer rows to the
   training-only future reader.
2. Replace the old fit helper's fixed408-context/five-tick-offset assumptions
   in separate source files. Reuse the original model, loss and normalization
   initially; the XY loss already uses0.06m scaling. Add explicit new plan and
   schedule validation without modifying frozen predecessors.
3. Fix matched training schedules, budgets, seeds, input variants and native
   assignments before fitting. Benchmark the relevant workload and assess
   whether a single shared immutable cache or bounded process caches provide
   useful throughput alongside the native queue. No fitting is yet complete.
4. Preserve the original queue/contact waiter. The separately prepared isolated
   recent-reference direct-flow maze3 native pilot remains gated behind their
   authenticated completion. Successful fitting or prediction loss still needs
   prospective physical navigation and the full original goal requirements.

# Primitive-leaf receipt freezing and cached cloning component

The completed tiled-controller profile identifies substantial recursive receipt
handling in late history: the frozen footprint visitor has 1,923,168 calls in ten
profiled decisions. The profile includes profiler overhead and does not measure
an isolated speedup. Its completed verification is bound by the benchmark runner.

`lewm/atomic_leaf_fused_footprint_development.py` is a separate candidate. It moves
the exact primitive-leaf test into the container loops, retaining the original
primitive type tuple, string-key checks, active-container cycle detection,
memoized aliases, original-input fallback and original read-only container types.
Only nonprimitive children invoke the freeze visitor. Cached cloning likewise
recurses only into the two validated frozen-container types. The original cache
query function body is reused with private bindings for these two helpers.
Capacity, query keys, failure checks, scope lifetime and counters remain intact.
Existing source and every running controller remain unchanged.

The focused component tests cover primitive values (including signed zero,
nonfinite floats and Unicode), shared DAGs, independent cloned ownership,
unsupported leaves/subclasses/keys, partial-graph rejection, cycles, fixed mixed
graphs, cache hits/capacity/errors and the actual original recovery functions.
Controller integration and a full-history paired replay are separate required
steps before any controller-speed claim or adoption.

The exclusive benchmark is
`scripts/benchmark_go2_atomic_leaf_fused_footprint_v1.py`, writing
`docs/go2_atomic_leaf_fused_footprint_component_benchmark_2026-09-11.json`.
It uses four predefined synthetic graph workloads: a primitive-heavy list,
nested geometry-shaped containers, a shared DAG and a late unsupported leaf.
Freeze is measured on all four; cached clone only on the three supported graphs.
Each measurement has three warmups and 30 paired repetitions in alternating
order with normal garbage collection. Every timed output is checked outside the
timer for full graph and alias equivalence; source mutation and fallback identity
are checked. Preserve all timings, including any regression. No recorded sensor
data, checkpoint, learned model, physics or independent layout is consumed.

The measurement is shared-host component evidence. It cannot establish whole
controller speedup, lower latency with a profiler removed, deadline compliance,
navigation success or hardware readiness. A positive result justifies a checked
controller composition and paired replay on the existing development history.

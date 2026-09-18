# Reduce repeated support calculations within one contact query

The completed current-controller phase diagnosis measured contact queries at
111.354410764 ms exclusive mean per active observation. Source inspection found
that the inherited contact layers repeatedly call the same original articulated
support calculation with identical measured joints and predicted orientation
within each candidate footprint evaluation. A separately named cache now targets
that duplication; no timing benefit is claimed before the paired benchmark.

SupportQueryCache lives only inside one footprint call. It keys exact float64
posture and direction bytes and shapes, calls the original supports() on a miss,
returns fresh copied receipts, never caches exceptions, and clears all entries
and releases the geometry on exit. At most eight entries are retained; excess
distinct queries use original computation. The supplied URDF geometry is fixed
during the scope. No state persists across candidate actions or observations.

The new memory inherits the verified single-pass memory, wraps only footprint's
geometry argument and records non-policy cache counters. Map/controller
initialization installs this memory; observe, advance, result, tracker,
registration, mission, residual, selector, indices and all contact/clearance rules
remain inherited. No running source or original geometry method was modified.

Validation: 11 focused cache/controller tests passed in 2.84 seconds (89060,
exit 0). Exact support receipts were checked against the complete 27-primitive
robot across measured postures/directions, including independent mutable return
containers, input changes, bounded capacity and original failures. Actual
dual-camera mapping and six distinct complete contact queries matched the
original exactly; each required at least five support requests but only one
calculation. Every scope closed without retained entries. Missing-RGB stopping
and original controller method identities also remained exact.

15 paired-benchmark tests passed in 2.15 seconds (71129, exit 0). They check
complete decision/input equality, alternating order, immediate stopping before
following observations, fresh independent models, active-only timing and the
original verifiers of the completed combined benchmark and phase diagnosis.
All six new files passed an explicit whitespace check.

Frozen new source hashes:

- `lewm/scoped_support_query_cache_development.py`:
  `a191b9cc5b8ad0ddd4f8e37a9733f645d96e76f305d71b4c3cdc2a3a7e86c5b1`.
- `lewm/support_cached_single_pass_controller_development.py`:
  `54f49ff8405e5b600ab38851d4dc45cfb9efcd90b9cc5754f51ce7ffb68e7c63`.
- `scripts/benchmark_go2_scoped_support_cache_v1.py`:
  `cf794c80a206563f699fef06ba3f68d84729f63d06ccf0a1da3e94399e2e6ae4`.
- `lewm/tests/test_scoped_support_query_cache_development.py`:
  `e6d6b5435e7c41cb141c5fa25ebfd8da60a3cd877327c7d0900b153e28896172`.
- `lewm/tests/test_scoped_support_cache_benchmark_development.py`:
  `c7f4ae6d33017e647937502fd57fd37e99e456a985ee247e6af20c62c285ede7`.
- `docs/go2_scoped_support_cache_benchmark_v1_2026-09-09.md`:
  `f53381804cb83f22cf5423a4bb186254d309ba5120961061dbfde5f0ec07a5e0`.

Fresh hardware assessment 54334, exit 0: 16 physical/32 logical CPUs, all 32 in
affinity, CPU 6.5%, 81,219,289,088 RAM bytes available, 75,975,356,416 bytes free
on the artifact volume and 21,358,845,952 on the workspace. Both GPUs idle.
The floor diagnostic and fresh hold replay are the competing CPU jobs; no native
scene runs. Conservative allowances of 16 GiB benchmark + 8 GiB floor + 8 GiB
hold fit. The benchmark refreshes hardware after full admission and every 64
observations; its own output allowance is 64 MiB above the unchanged reserve.

Submitted session 98475 for the complete 514-observation paired replay. Each
variant must match the saved original full decision, preserve public arrays and
model state, and use its own fresh assigned model. The existing combined
controller is the baseline; the candidate adds only this scoped support cache.
Input authentication is pending. Preserve exclusive output
`go2_scoped_support_cache_benchmark_v1_attempt_001` and any failure. No speedup,
real-time operation, native adoption or navigation improvement is yet established.

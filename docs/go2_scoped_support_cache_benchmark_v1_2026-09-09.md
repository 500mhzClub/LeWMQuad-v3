# Scope-local support cache: paired complete-controller benchmark

The completed current phase diagnosis records a mean 111.354 ms per observation
inside six contact queries. Source inspection shows repeated identical calls to
the original articulated support calculation within one footprint evaluation.
Test reuse of those exact results with a cache owned by that one evaluation.

Compare the verified SinglePassReceiptCopiedController with the separately named
SupportCachedSinglePassController on every one of the 514 original learned
maze-2 observations. Both use fresh independent instances of the same assigned
corrected JEPA model and the same unchanged initial controller state. Alternate
which controller runs first on each observation. Both complete decisions must
equal the saved original decision; stop immediately on any mismatch, before
consuming the next observation. Require unchanged public arrays, model states,
absent gradients, actual completed original commands, ordered endpoints and the
full observation population. Keep warmup and terminal timing separate.

The candidate retains all original contact processing, floor rules, forecast,
utility, geometry, indices, copy semantics, tracker, mission and residual code.
Only footprint() wraps its supplied frozen robot geometry in a fresh scoped
support cache. Exact float64 posture/direction shapes and bytes form each key;
no tolerance or quantization. Original supports() computes each cache miss.
Every call returns independent copied receipt containers. Never cache failed
computations. Retain at most eight entries, then use original computations for
uncached queries. Clear entries and release the geometry at scope exit, including
exceptions. No cache survives into the next candidate or observation. The
geometry definition is fixed throughout each scope; its consumers only read it.

Require completed combined benchmark result
`d688f2ed9d30177d2e55fb98e9c9f25d2b035f2258d86449ac8e86615cd13c72`
and phase result
`6ad04101046cdcb9e855a100a446bb0206397354614608f89a03c38d80ecdf76`.
Authenticate their artifact and source bindings with their original verifiers,
including all optimization ancestors and the complete raw-audited input episode.
Reject conflicting inherited source identities. Preserve every frozen attempt.

One CPU replay process with two independent model/controllers, single numerical
threads, 16 GiB memory admission and 64 MiB output allowance above the existing
40 GiB reserve. Assess CPU/affinity, RAM, GPU, both volumes and competing jobs
before submission; refresh admission before output and resources every 64
observations. It may overlap independent CPU analyses with measured headroom.
No native scene, training, source export, cleanup, sealed access or real-robot
action. Retain failures and exclusive output
`go2_scoped_support_cache_benchmark_v1_attempt_001`.

This measures additional controller-compute benefit on one reused development
trajectory. It does not establish acquisition/command/receipt-I/O speed, native
navigation, a calibrated clearance bound, real-time qualification or deployment.
No adoption into a running or frozen native controller is included.

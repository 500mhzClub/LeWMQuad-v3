# Longer-trial resource observations, not dispatch admission

The completed 4,014-observation chained native attempt provides an initial
resource reference. Its root result was reauthenticated at SHA-256
`163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.
Only explicitly bound paths were inspected, with custody and resolved-root
checks before access. No source tree or generated directory was recursively
searched, and no native/controller experiment was rerun.

The metadata sizes of its **24,124** bound artifact files total
**12,402,149,312 bytes (11.5504 GiB)**. This includes per-frame images, depth,
segmentation and receipts as well as collection, audit and launch records.
The size scan did not rehash all those artifacts; their earlier complete hash
verification is recorded in
`docs/go2_measured_plane_chained_native_completed_2026-09-12.md`.
The largest single artifact is the compressed complete decision stream,
609,591,253 bytes.

Simple doubling gives **24,804,298,624 bytes (23.1008 GiB)**. The prepared
collector allowance is 28 GiB. Doubling is an estimate, not an upper bound:
the image content, compressed decisions, retained histories, duration of the
return and subsequent audit outputs can scale differently. The collector's
allowance does not alone bound all parent, persistence and audit artifacts.

The original `resource_monitor.jsonl` was independently rehashed to its bound
SHA-256 `067733f75fccf289b323d6824e97226d42e5a13d5c5977aac3a6f6f1e5f5982e`.
Its **934** samples cover parent elapsed times 0.0025 to 14,941.9368 seconds.
Available system RAM ranges from **66,820,702,208 to 81,690,972,160 bytes**
(62.2316 to 76.0806 GiB). This is sampled system availability, not worker RSS,
an isolated allocation measurement or a proof of peak memory between samples.
Other processes and system caches can contribute to that range.

The physics archive was independently rehashed to
`2eacfb0c9a0c987dbdb756c773320ef66437d7c0373bca463bd4a6fb33bc6e32`.
Reading only its NumPy array headers establishes eleven arrays with 201,400
samples: 47 float64 values and three uint8 values per sample. Their uncompressed
NumPy members total **76,332,008 bytes**, including 1,408 header bytes.
The same schema at 401,400 samples would require **152,132,008 bytes** of
uncompressed NumPy members. This exact arithmetic applies only to those arrays;
it excludes Python sample objects, contacts, raster/feature history, public
packets, decision structures, serialization copies and concurrent audit state.

Before native dispatch, assess those remaining allocations and the full
collection/persistence/audit lifecycle, select explicit headroom and enforce
resource checks for the longer trial. The prospective prefix replay will also
provide a measured retained-state resource trace through the original deadline.
The system currently has ample artifact space, but that observation is not a
substitute for the longer trial's resource admission.

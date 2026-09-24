# Batched measured-sample bounds: source candidate

The completed observation-replan controller profile identified repeated voxel
grouping and bounds construction as a controller cost. The new
`lewm/batched_sample_bounds_development.py` computes the unique voxel grouping
once per insertion and constructs/merges bounds in batches. It inherits the
frozen query implementation, keeps outward rounding and first-witness order,
and still independently copies each witness.

Seven tests in `lewm/tests/test_batched_sample_bounds_development.py` passed in
1.11 seconds. They compare against the frozen implementation across accumulated
clouds, repeated and boundary points, empty input, counts, exact bounds,
box/sphere queries, independently mutable witnesses/bounds, invalid input and
capacity exhaustion. This is a source candidate; no native controller uses it.

An exploratory synthetic timing check used seed 2026091702, ten fixed clouds of
19,200 points uniformly sampled from [-0.3, 0.3] metres, and four fresh paired
index runs with alternating implementation order. Original insertion totals
were 0.550075, 0.544597, 0.541555 and 0.548047 seconds. Batched totals were
0.254346, 0.250200, 0.255322 and 0.243939 seconds. The ratio of medians was 2.166.
This tool-recorded exploratory check ran on one numerical thread while the
separate fits continued; it has no frozen native benchmark attempt or claim of
controller speedup.

Before adoption, compare full recorded controller decisions and failure
behavior under explicit dependency wiring, and measure the uninstrumented
whole loop. Source/test success and this synthetic timing do not establish
real-time operation, maze success or deployment validity. Keep the prepared
short-horizon probe unchanged so its result addresses its declared intervention.

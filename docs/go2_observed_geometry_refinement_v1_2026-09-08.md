# Observed geometry refinement V1: prospective diagnostic specification

Reconstruct the two full-RGB direct-model commitment-pose recordings, preserving
every available map receipt and every unavailable observation. Authenticate the
native, route-loss V2 and inherited source/input identities before and after.
No native execution, learned-weight change, command change or filter waiver.

At each original selection, measure the continuous Euclidean distance from the
actual observed base position to every closed occupied 5-cm square. Search the
same observed-floor entry candidates within 1.25 m, ordered by distance then
cell, retaining the original inflated-grid exclusion for entry cells. A segment
is nominally clear only when its distance from every occupied square exceeds
the unchanged 0.45-m radius plus 1e-12 m. Tangency is blocked. When the start
itself is blocked, no connecting segment can pass. This diagnoses grid
overapproximation; it certifies neither articulated motion nor unknown space.

For every unique first-hit voxel/witness in the V2 candidate records, reconstruct
all original stride-four depth samples in that voxel at its first witness frame.
Retain pixel coordinates, observed initial-frame points and map heights. Classify
a point as a measured floor patch only if all four adjacent pixel quads satisfy
the existing ground-normal and planarity tests and all nine surrounding valid
pixels lie within 10 mm of the original fixed measured floor hypothesis.
Require the exact RGB/depth/frame witness. No scene label, native pose or future
frame enters the classifier. First-witness classification does not characterize
later observations in the same voxel, approve support, or remove a surface veto.

Run the nine focused analytic tests before freezing. Inspect hardware and retain
at least 8 GiB available RAM and 41 GiB artifact free space. Output capacity is
1 GiB. Benchmark 40 frames per case with one and two threads, require identical
results and choose the faster configuration for full reconstruction. Record
every failure in a terminal exclusive attempt; do not overwrite or resume.

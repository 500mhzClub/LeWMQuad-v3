# One auxiliary depth reconstruction per paired session call

The completed chained native run recorded median packet-acquisition time
219.55553600000002 ms, with all 4,014 observations exceeding 100 ms before
controller computation. Source inspection found that the original paired
session builds auxiliary depth, then the dual-camera extension builds it again
while constructing auxiliary RGB. The timing evidence and its limits are in
`docs/go2_measured_plane_chained_logged_timing_decomposition_2026-09-12.md`.
The individual cost of this duplicated reconstruction has not been measured.

The new source
`scripts/single_read_auxiliary_maze_session_development.py`, SHA-256
`3c20a148bc86b98c6bd00b73c6f91516d3e560e9cca494440b55c6acc2238b77`,
composes the existing primary packet method, unchanged auxiliary capture and
existing combined auxiliary RGB/depth packet builder. The combined builder
returns both public packets after reading and validating auxiliary depth once.
The session keeps the original pre-capture observation bound, primary index,
physical pairing, uninterrupted auxiliary history and final RGB validation.
It retains the 4,014-observation bound and changes no mission budget.

`SingleReadAuxiliaryRendererSession` retains the existing extended renderer
wrapper and primary capture methods. Their cooperative method dispatch enters
the new paired method, preserving renderer witnesses, terminal failure latching
and inherited persistence. Raw rendering, file encoding, pixel hashes,
calibration, camera geometry and public schemas are unchanged. No controller,
model, timing boundary or native launch was changed.

The focused test file
`lewm/tests/test_single_read_auxiliary_maze_session_development.py`, SHA-256
`4febbbe40b46fdc028a76fec2c3f0490a0eef16c74352b0c1231fdf6cdd53575`,
passed **20 tests in 2.35 seconds** in tool session 87234. The tests use
synthetic sessions and captures, with real persisted RGB/depth arrays and real
public packet construction and validation. They cover:

- Complete original/candidate packet equality and unchanged primary inputs,
  with two original archive reads versus one candidate read.
- Preserved depth missingness and exclusion of evaluator-only pose fields and
  a diagnostic object array that cannot be loaded with `allow_pickle=False`.
- Fresh packet reconstruction on a repeated call without reacquiring the
  camera or retaining a mutable packet cache.
- Rejection of corrupted depth/pixels/hashes, wrong calibration/frame/clock,
  physical mispairing, primary index drift and auxiliary gaps/extra rows.
- The renderer's budget guard before primary acquisition, including the last
  allowed and first disallowed indices, and its terminal failure latch.
- Paired witness pixel bindings, repeated-call witness preservation and
  rejection of camera-pose drift.

The complete focused test command used the original deterministic single-thread
environment with `python -B -m pytest -q -p no:cacheprovider` and the exact test
path above. These short synthetic checks overlapped the already declared
non-isolated controller timing replay; they are not a performance benchmark.
Afterward all 2,639 sources bound by that live replay were independently
rehashed and remained unchanged. Both new files are outside its source roster.

This is a tested implementation candidate, not native adoption or a measured
speed result. There is no new native launcher or queued execution. Before
adoption, compare complete actual recorded public packets and measure the
packet-construction cost on the completed native population after the current
CPU replay ends. That check must distinguish recorded-file reconstruction from
live camera capture. Any subsequent native trial must retain full raw sensor,
renderer and command audit, so identical simulated acquisitions remain
verifiable. Rendering/encoding costs, controller latency, continuous execution
and hardware sensing remain unresolved.

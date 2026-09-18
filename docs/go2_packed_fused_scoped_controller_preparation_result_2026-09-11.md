# Existing packed-index optimization integrated at empty controller state

The current anchored controller was still using the original measured-bound
insertion. The repository's existing PackedOwnedMeasuredSampleBoundsIndex
already groups voxels once and owns each bounds array independently. The new
PackedFusedScopedController installs that exact implementation into eight
empty persistent indices while preserving the current memory and map types,
their alias, selector, scoped cache, historical patch stores and query methods.

Nineteen component/integration tests passed in 7.01 s, session 30228, exit 0.
They cover actual public observations and articulated robot footprint queries,
cache operation, independent receipts, exact stored values, rejected partial
installation and unchanged failure stops. The state comparator permits only
the two existing patch-store type tags and eight exact index-type tags to
differ; changed bound bytes remain detectable.

Existing historical evidence was authenticated through its original source
and output bindings, including the complete saved decision streams. It was
not rerun:

- Component result `e3f603f7baed193e23b53116fe14f46a94bc6a98ad4b5eebca0af604dbe9a872`.
- Full old-controller replay `4de3195f7976768c7840189e8c5c77227f5842bd090163c5975f2d8d8063b125`.
- Old paired timing result `7a91c2a5588002989f607da26db5ad0e6cedd789500a27c6f6f957a2f952a8c5`.

Preparation and historical-output verification exited 0 in session 96728.
The preparation binds 2,179 source paths:
`docs/go2_packed_fused_scoped_controller_preparation_2026-09-11.json`,
SHA-256 `a33aa0596da7a6cca9326ce722034c83932df95e1f8e1e717c9ac52d19c75108`.
The protocol is `docs/go2_packed_fused_scoped_controller_v1_2026-09-11.md`.

This composition has not undergone a full raw controller replay. The current
fused-receipt comparison is still running and retains the sole full CPU replay
slot. Its completion must be checked before executing the next complete paired
comparison; that comparison's launcher still needs preparation. Historical improvements are not measurements of
the new composition and cannot be added to other speedups. No native command,
navigation improvement, real-time qualification or goal completion follows
from this source preparation.

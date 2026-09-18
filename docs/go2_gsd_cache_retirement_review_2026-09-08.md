# Exact historical GSD cache retirement proposal

No deletion has occurred. Explicit approval is requested for the **8,304 named
ordinary files**, totaling **81.782 GiB allocated**, in
`/home/andrewknowles/.cache/genesis/gsd`. The exact candidate names and metadata
are in `docs/go2_gsd_cache_retirement_proposal_2026-09-08.json`, SHA-256
`bc405303fb8f7d227c71bf7e965d2262ca6ad8f8e817e78512427bbacf8eaa77`.
The candidate population is fixed; this is not recursive directory deletion.

A separate source-bound inspection built the exact current maze scene with zero
physics steps and no navigation or actuator policy. It found 53 native geometries:
one plane, 30 boxes, 17 cylinders and five spheres, with 13 distinct derived GSD
keys. None of those 13 keys currently exists among the 8,304 candidate files.
All 13 are explicitly excluded from retirement even if created later. No geometry
was SDF-preprocessed at construction, and no new GSD cache file was created.
This does not rule out future lazy preprocessing; it ensures such current-scene
keys are not targeted. Existing unrelated cache payloads were not opened.

Inspection root `go2_maze_geometry_cache_identity_v1_attempt_001` completed with
result SHA-256 `f409c5a6211b7443e06b1045be5ae6983332922670cfb2b6ce6738a82e9876ce`.
Launch SHA-256 `f505559a982e4166f54caa1def34343a2c3ca2073ffb1b1243a0e8c2c38fa1ee`
binds 1,440 sources. Metadata before/after had identical SHA-256
`e724a00b28b554ec45bea8000839a7250a4cd523451f1ab384313825fc3b708a`.
Session 18648 completed in 16.947 s after launch. The two identity tests and two
retirement-helper tests passed, including rejection of metadata changes before
any unlink, preservation of current keys and protected-path rejection.

If approved, wait until original native/audit PIDs 2345062 and 2345120 terminate;
then inspect current-user native tasks and open cache files again. Verify every
candidate's inode, size, ownership, single-link status and modification/change
timestamps against the proposal before removing the first file. Recheck each leaf
before removal and journal every removed name at a new exclusive artifact root.
Preserve the cache directory, all unlisted files, all current geometry keys, and
all experiment datasets, models, source and failure records outside this cache.
No protected-directory traversal or cache-payload deserialization is performed.

The tradeoff is loss of these historical preprocessed geometry cache entries.
Genesis regenerates missing entries when those geometries are reused, adding
startup work; exact byte-for-byte regeneration has not been established. This
proposal does not declare every historical simulation independent of the cache.
The prior storage review explicitly reserved cache retirement for a separate
decision. Without approval, keep the cache and continue source/readout work.

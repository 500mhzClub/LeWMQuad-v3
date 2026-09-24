# Current maze geometry-cache identity inspection V1

Prepare storage retirement scope without changing any cache or scientific artifact.
Run `scripts/inspect_go2_maze_geometry_cache_v1.py` once in exclusive
`go2_maze_geometry_cache_identity_v1_attempt_001`, binding the current waypoint
native launch `f7fec194358f3821c036df413c360f355cd758ec7d1ca5c1b7466b99ceea9f4e`
and its source/native closure. Use exactly development maze 0's pack and original
visible-robot scene builder. Build one CPU scene and verify zero physics steps
before and after inspection. Do not load a high-level model or PPO policy, issue
commands, render sensor frames, construct another layout or infer navigation.

For each geometry in the explicitly constructed current scene, derive its exact
GSD key from its native initial vertices/faces and material SDF settings using the
installed Genesis function. Require the ordinary default cache root and 64-hex
`.gsd` basename. Bind only existing files at these source-derived paths with
SHA-256 and byte count. If geometry was preprocessed, its actual loaded path must
match that derived path. Do not trigger lazy preprocessing for absent cache files
or deserialize any cache payload in the inspector. Native scene construction
itself retains its existing cache loading behavior. No arbitrary historical or
protected cache payload is inspected.

Record flat cache metadata before and after, excluding protected names before
access and rejecting unexpected links/directories. Require all preexisting file
metadata (except access time) unchanged. Any new cache leaf created naturally by
the original builder must belong to the current scene's derived key set. Record
it; never remove it. The scene's current collision geometry cache can subsequently
be retained by these identities before a concrete, separately approved cleanup
proposal for other cache files. This inspection authorizes no retirement.

Two focused tests passed in 0.10 s: source-derived key binding/deduplication,
missing-key noncreation, loaded-path mismatch rejection, and protected/unnamed/
symlink rejection. The active native worker 2345120 was directly observed with
both `GS_CACHE_FILE_PATH` and `XDG_CACHE_HOME` unset; the inspection requires the
same default environment. Freeze the installed mesh-key/cache-root/rigid-geometry
source file hashes before build and recheck afterward.

Assess hardware before launch: require 8 GiB available RAM and 64 MiB above the
unchanged 40 GiB artifact reserve. One bounded zero-step scene may coexist with
the current native worker's independent raw-audit stage, which has finished
navigation collection and destroyed its scene. No parallel navigation collection,
GPU computation or training is requested. Preserve inspection failures and all
old raw results. Cache identity evidence does not establish navigation success,
future cache regeneration equivalence, or hardware/real-time qualification.

# Storage follow-up while the waypoint native attempt runs

The original storage-review candidate `/home/andrewknowles/TinyQuadJEPA` is now
absent. No removal was performed in this work. Do not request approval using its
stale 23 GiB estimate. The active waypoint launch contains no literal reference
to that path; accessible process metadata showed no reference. Many system
processes were inaccessible, so that observation is not a universal dependency
proof.

The completed nominal-reentry native attempt's explicitly bound artifacts occupy
1.336112144 GiB for 443 paired observations. NPZ files account for 1,299.01 MiB;
the complete compressed decision stream is only 16.05 MiB. This points to sensor
payload storage, not decision receipts, as the main volume cost. Long full-budget
comparisons will require more headroom; reducing only receipt size will not solve
that problem. Current native collection has its unchanged declared 10+1 GiB
envelope over the 40 GiB reserve and must remain untouched.

Metadata-only inspection of `/home/andrewknowles/.cache/genesis/gsd` found 8,304
ordinary `.gsd` files, 87,812,689,920 allocated bytes, all owned by UID 1000.
The walk explicitly excluded protected paths and symlinks; none were encountered.
No cache payload was opened or changed. Current native launch JSON has no literal
binding to this cache path, but that is not evidence the simulator does not use it.

Inspection of the installed, current Genesis source established that this cache
contains preprocessed geometry signed-distance fields, gradients, closest vertices
and transforms, loaded into collision geometry. It is not merely a compiler cache.
`genesis/utils/mesh.py:get_gsd_path` keys it from geometry vertices/faces and SDF
resolution settings. `genesis/engine/entities/rigid_entity/rigid_geom.py:_preprocess`
loads the file when present and otherwise computes and writes those arrays.
The existing native robot-geometry receipt does not capture each `.gsd` identity.

Do not retire this cache blindly or while the active scene is running. A useful
next read-only step is to prepare a separately bounded source-identical scene
inspection that enumerates and binds only the exact cache identities used by the
current development scene. Preserve those identities before proposing any cache
retirement. Existing cache-retirement/deletion approval has not been established;
no deletion request is ready until exact scope, active users and retention impact
are concrete. All scientific artifacts and failed attempts remain preserved.

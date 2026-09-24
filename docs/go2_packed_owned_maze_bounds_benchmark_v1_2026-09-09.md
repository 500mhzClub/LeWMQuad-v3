# Exact packed voxel grouping and independently owned bounds benchmark

This separate performance candidate reuses the original measured-bound queries
and all insertion semantics. Pack the frozen +/-50m,25mm voxel coordinates
into three biased12-bit fields, preserving lexicographic key order, and group
using one-dimensional np.unique. Preserve the original inverse groups, sample
counts, first-witness order, independent witness copies, outward nextafter
bounds, latest frames, capacity rejection and partial failure behavior.
Each resulting2x3 bound array owns its48 data bytes. The earlier batched
candidate stores views into larger insertion batches; a surviving voxel can
therefore retain unused batch storage. No original candidate is edited.

Ten focused tests pass, including exact grouping at signed/boundary coordinates,
accumulated bounds and queries, input/capacity failures, witness independence,
and absence of retained batch-array backing storage. This is source evidence,
not adoption into the prepared settled-boundary native experiment.

Use exactly the first128 saved public RGB-D observations of the completed ninth
native pilot. Validate their saved current registered pose witnesses and rebuild
the original primary measured clouds with the unchanged body_points(stride=4)
and points@R.T+p operation. No native pose, extra camera, model, training or
simulation. Bind consumed collection manifests/histories/RGB-D files, the full
saved decision stream and completed native result/launch before and after.

Two fresh paired runs over all128 clouds: implementation order original,
previous batched, packed-owned; then reverse that order. Time insertion only.
After each frame, require exact key order, witnesses, counts, latest frames,
bound bytes and representative box/sphere queries for all three indices.
Record every duration and final backing-storage bytes, counting each distinct
array owner once. Verify every cloud remains unchanged. Timing excludes packet
decoding and equivalence checks and cannot establish whole-controller speedup.

One CPU process/one numerical thread beside the already running independent
settling replay. Record topology, affinity, CPU/GPU/VRAM activity, RAM, storage
and competition. Admit4GiB available RAM and64MiB output above40GiB reserve.
Exclusive outputgo2_packed_owned_maze_bounds_benchmark_v1_attempt_001. Resource
allowances are not OS limits; preserve every result/failure. No native adoption,
real-time, navigation or overall-goal success claim follows from this benchmark.

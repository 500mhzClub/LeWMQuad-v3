# Long-tape recording resource analysis — incomplete proof

This is a read-only installed-source analysis and prospective allocation
correction. It is **not** the successful native resource review required by the
launcher. No native scene, checkpoint, training or new dataset was opened.

## Why the previous allocation was insufficiently justified

Even before analyzing contacts, the previous operation allowances summed to
443 × 6 MiB + 128 MiB meshes + 8 MiB setup + 512 MiB deferred = 3,466,592,256
bytes, exceeding the old 3 GiB episode ceiling (3,221,225,472 bytes). Actual
compressed frames may be smaller; that possibility cannot prove a full-tape
worst-case reservation. The new arithmetic test retains this counterexample.

The inherited contact recorder copies all eight native contact fields into a
Python list on each physics sample. It does not keep only disallowed events.
The snapshot concatenates every packet while the original packet arrays remain
alive, and the NPZ writer adds compression-buffer/payload retention. Subsequent
snapshot writes also occur while the concatenated contact dictionary is in
scope. Starting each trial in a fresh process addresses inter-trial retention,
not this within-trial multiplication.

Installed Genesis source was inspected at:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/genesis_rocm_0_4_6_v1/lib/python3.12/site-packages/genesis`.

| Exact relative source | SHA-256 |
| --- | --- |
| `__init__.py` | `35a956c8be7dc836063d2848005896bb7017c29bde758292f6415f047bad395c` |
| `options/solvers.py` | `2e43097685009dd8ddd7dae926ba37be6659ebb93663095944e08edb4489aa47` |
| `engine/solvers/rigid/collider/collider.py` | `3d5863403a98cd738134116cc4768a68eb122c39f11d4d90c25329e60f37a0b9` |
| `engine/entities/rigid_entity/rigid_entity.py` | `05c16043c05acb129f2e54723bc2d619686a1e00cfe2f6419b0e68521c940649` |

The scene constructor supplies no rigid-solver overrides; the initializer does
not override precision. Installed defaults are float32/int32, at most 150
collision pairs and `box_box_detection=False`. Without differentiable physics,
the collider uses five contacts per pair for allocation. Its capacity formula
is `min(max_collision_pairs, n_possible_pairs) * n_contacts_per_pair`, hence a
coarse capacity envelope of 750 rows per sample under these defaults. This is
not a measurement of contacts encountered in either challenge scene.

The public collider slices to the actual contact count, not the entire padded
capacity. However, for `n_envs=1`, the entity wrapper retains the contact rows
with a robot-validity mask; it does not filter them out as the unbatched branch
does. The recorder copies even invalid rows. Self contacts are ignored by the
external-contact classifier. It is therefore invalid to derive a storage bound
from four allowed foot contacts or a small number of disallowed events alone.

## Contact-only allocation calculation

For 22,850 samples × 750 rows, the envelope is 17,137,500 contact rows. Four
int32 index members, three float32 three-vectors and one boolean require 53
bytes per row. Including int64 frame offsets and float64 timestamps gives
908,653,108 uncompressed array bytes. Applying the current writer's per-array
reservation formula (`nbytes + nbytes//10 + 65536`, plus 65536 for the archive)
to all ten members yields 1,000,239,314 bytes, already larger than 512 MiB.

This is a source-derived conservative reservation, not proof that actual NPZ
files exceed 512 MiB or that every capacity row is physically reachable on
every sample. Conversely, an assumed compression ratio cannot certify the old
allowance. Original packet arrays and concatenation alone are approximately
1.82 billion bytes at this envelope, before Python object headers, temporary
copies, compression, physics/sensor histories, renderer/compiler state or the
gait policy. No total peak-memory conclusion follows from this calculation.

## Implemented correction and remaining work

The actual Python method-resolution order was inspected without constructing a
session: `PulseContextSession → FastGyroSession → ObservationSession →
FactorialSession → AttributedSession → GeometryFreePhysicalSample`. In
particular, the effective physics sample has no legacy region-membership
arrays. The two recording wrappers add `phase` and `edge_index`; both remain
uint8. Fixed float64 body/history data, int64 timestamps and boolean masks give
the following numeric allocation calculations for the complete tape:

| Archive | Array members | Uncompressed bytes | Writer reservation bytes |
| --- | ---: | ---: | ---: |
| `physics_trace.npz` |11|8,660,150|10,312,597|
| `ideal_sensor_samples.npz` |7|635,230|1,223,040|
| `policy_histories.npz` |18|3,110,303|4,666,516|
| `fast_gyro_samples.npz` |4|982,550|1,408,485|
| `fast_gyro_histories.npz` |4|971,499|1,396,327|

The totals are 14,359,732 raw bytes and 19,006,965 writer-reservation bytes.
Slow samples number at most 2,285; each of the 443 body histories contains
20 gyro, specific-force and joint samples plus 15 command samples. Each fast
history contains 51 samples. These are source-derived numeric arrays, not a
measurement of native output, a JSON or Python-object bound, or an independently
validated archive-shape contract. A later resource checker should test those
shapes against the actual recorder rather than simply trusting this table.

The unfrozen artifact writer now reserves 2 GiB for deferred recording and a
5 GiB episode ceiling. All declared operation reservations fit that ceiling.
The raw cohort ceiling derives from eight episodes (40 GiB); base-only outputs
have 4 GiB headroom, and the full launcher/stress population has 12 GiB headroom
(52 GiB total). These larger ceilings are not disk allocation, an execution
attempt, or a successful native-recording/memory proof. Actual bytes remain
bounded and authenticated, with physical and infrastructure failures preserved.

Before issuing the genuine resource review:

1. Confirm the effective native options/dtypes and capacity assumptions through
   the frozen construction path, including any native resizing or overflow
   behavior; bind the relevant installed implementation, not just this prose.
2. Bound all physics, body/gyro sample and history arrays, per-frame manifests,
   command/guard/contact-event metadata, geometry/setup/terminal records and
   serialization overhead. Show their combined deferred output fits 2 GiB.
   If not, revise the unfrozen recorder/allocation before consuming an attempt.
3. Account for packet/object retention, snapshot concatenation, NPZ buffers,
   native solver/renderer/compiler and policy memory. If source bounds cannot
   establish the proposed 8 GiB worker allowance, implement a bounded recording
   strategy or a separately justified resource-validation procedure. Do not
   substitute observed available RAM for proof of peak demand.
4. Check the replay/scoring parent's memory too: sequential native workers do
   not bound whole-tape replay or native reconstruction. Confirm reserve and
   scheduling immediately before any later launch, after the original all-12
   collection and 36-fit study finish. Current free space is not a future
   reservation.

Do not truncate contacts, discard invalid rows, alter contact-stop semantics,
reduce the prescribed turn/translation population, or claim a partial tape as
a completed challenge merely to meet an unproved resource allowance.

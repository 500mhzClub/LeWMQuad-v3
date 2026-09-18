# Batched retained floor-patch projection V1

The completed controller profile identifies `RetainedFloorPatches.coverage`
as a recurring cost. The current frame-index cache is already integrated and
does not cache these historical footprint queries. This candidate batches the
projection arithmetic across at most 32 retained observations at a time.

The original append method, stored prefix images, witness history and query
limits remain inherited. Stack only pose arrays and query corners; never copy
or stack the 480×640 historical prefix images during a coverage query. Keep
the original projection arithmetic, near/far depth limits, pixel bounds,
floating-point margins and integer prefix-sum checks. Query stored images in
chronological order and keep the first single-observation complete witness for
each centre. No tiles, partial observations or cameras are combined to create
a positive result. No old evidence is discarded or reclassified.

The batched projection computes rectangles for all remaining centres at a batch
boundary, including centres that may become covered earlier within that batch.
Only still-uncovered centres receive subsequent pixel-prefix queries. All of
these frames are already observed history; this reads no future sensor packet.
The implementation adds no persistent cache, changes no model or nominal
support policy, and makes no ground-support or navigation qualification.

If projection of an unused later frame would raise an arithmetic or metadata
error, retry that batch chronologically with only still-uncovered centres.
Preserve early termination: a later frame must not defeat an earlier complete
witness. An error on a frame actually needed by the chronological calculation
still propagates. Pixel-prefix queries and witness copying are not attempted
until the batch projection succeeds or chronological fallback reaches them.

Before integration, require exact complete outputs against the original on
history/query batch edges, rotated and translated poses, visibility bounds,
mixed earliest witnesses, incomplete pixel patches and invalid queries. Verify
the original append/source identity and output witness independence. Then
measure representative retained-history queries and require a separately frozen
full raw controller replay before using this class in any native controller.

These source changes do not install the candidate in a running or queued
experiment. A component speedup would not prove end-to-end 100-ms execution,
goal-reaching, memory advantage or hardware readiness.

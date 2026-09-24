# Later floor measurements resolve four terminal enclosure queries

Completed provisional read-only diagnosis, session56870. Root
`go2_retained_floor_resolution_diagnosis_v1_attempt_001`; result SHA-256
`3d9a5d3109eff7fb9f8e38acf347ac13a631ef0e643679883eeed3db468b9912`.
Launch `79bcbbcc2b060f35c363c3dc39342b0254c5bc483c02819b84f54d1a54746da7`.
1511 sources and6441 completed-collection artifact bindings were checked before
and after. The native final audit is still required; this does not replace it.

The search examined915 paired observations,143–1057 inclusive. A target requires
a strictly later observation than its last contributing ambiguous sample, all
5cm squares touched by its map XY enclosure covered in one camera observation,
and its complete height interval inside the existing10mm floor band.

| Retained partition | Cell | Last contributing sample | First later complete coverage |
| --- | --- | --- | --- |
| auxiliary | [86,33,-11] |364|auxiliary850|
| primary | [99,40,-10] |143|auxiliary733|
| primary | [87,33,-11] |142|auxiliary850|
| auxiliary | [87,33,-11] |363|auxiliary850|
| primary | [86,34,-11] |361|none through1057|
| auxiliary | [85,34,-11] |357|none through1057|

The later coverage witnesses precede the terminal1057, so they were already
available to the failed controller. Its monotone original unknown-return
partitions do not revise those old classifications when later measurements
cover the same region. This identifies a specific memory-update candidate;
height proximity alone and absent later coverage do not resolve the last two
enclosures. No original return, classification, policy or action was changed.
The query remains conditional on the recorded pose and flat-floor hypotheses;
it is not a calibrated bound, contact/support certificate or navigation result.

Next implementation should preserve the all-return and original partition
indices, adding explicit later-evidence resolution only to nominal-foot contact
queries. Record per-observation primary/auxiliary covered floor-cell sets and
their timestamp/hash witnesses while those sets are actually computed. A query
must enumerate every intersecting ambiguous bound, not just its first hit.
For each bound, conservatively transform all of it into the fixed map and
require both the original height band and a single later measurement covering
every touched floor square. New samples in a cell invalidate earlier resolution
through its latest-sample frame; camera/frame witnesses must remain explicit.
Non-foot hits and any unresolved enclosure remain blocking. Preserve the full
original contact check and report resolved/remaining counts separately; do not
erase or silently relabel history. Source must be separate and tested, then
prospectively replayed to the first changed command before any new native run.
Keep all original model predictions, action set, full nominal horizon and
physical stops. The completed frame-cache implementation can supply the new
variant's computations, but needs its own source and complete-prefix validation.
